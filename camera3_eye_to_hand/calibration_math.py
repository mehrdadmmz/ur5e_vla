"""Pure transform math for fixed camera3 eye-to-hand calibration.

Transform names follow the convention ``T_A_B``: the pose of frame B
expressed in frame A. A point in B is transformed into A with
``p_A = T_A_B @ p_B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CalibrationEstimate:
    """A robust estimate of a fixed transform and its consistency metrics."""

    transform: np.ndarray
    inlier_mask: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.inlier_mask.size)

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    def metrics(self) -> dict[str, float | int]:
        """Return YAML/JSON-friendly residual statistics for inliers."""
        t = self.translation_errors_m[self.inlier_mask]
        r = self.rotation_errors_deg[self.inlier_mask]
        return {
            "num_samples": self.num_samples,
            "num_inliers": self.num_inliers,
            "translation_rmse_m": float(np.sqrt(np.mean(t**2))),
            "translation_median_m": float(np.median(t)),
            "translation_max_m": float(np.max(t)),
            "rotation_rmse_deg": float(np.sqrt(np.mean(r**2))),
            "rotation_median_deg": float(np.median(r)),
            "rotation_max_deg": float(np.max(r)),
        }


def validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """Validate and return a 4x4 rigid transform as float64."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    matrix = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def camera3_in_base(
    T_base_ee: np.ndarray,
    T_ee_camera2: np.ndarray,
    T_camera2_board: np.ndarray,
    T_camera3_board: np.ndarray,
) -> np.ndarray:
    """Compute one candidate fixed wall-camera pose in the robot base.

    Both cameras must observe the same stationary ChArUco board for a sample.
    Camera2 is the wrist camera and camera3 is fixed to the wall.
    """
    return (
        validate_transform(T_base_ee, "T_base_ee")
        @ validate_transform(T_ee_camera2, "T_ee_camera2")
        @ validate_transform(T_camera2_board, "T_camera2_board")
        @ invert_transform(validate_transform(T_camera3_board, "T_camera3_board"))
    )


def camera3_in_camera2(
    T_base_ee: np.ndarray,
    T_ee_camera2: np.ndarray,
    T_base_camera3: np.ndarray,
) -> np.ndarray:
    """Return the dynamic pose of camera3 expressed in wrist-camera2."""
    T_base_camera2 = (
        validate_transform(T_base_ee, "T_base_ee")
        @ validate_transform(T_ee_camera2, "T_ee_camera2")
    )
    return invert_transform(T_base_camera2) @ validate_transform(
        T_base_camera3, "T_base_camera3"
    )


def rotation_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic angular distance between two rotations."""
    relative = Rotation.from_matrix(first).inv() * Rotation.from_matrix(second)
    return float(np.degrees(relative.magnitude()))


def mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average rigid transforms using an SO(3) mean and arithmetic translation."""
    matrices = _stack_transforms(transforms)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(matrices[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.mean(matrices[:, :3, 3], axis=0)
    return result


def estimate_fixed_transform(
    transforms: Sequence[np.ndarray],
    max_translation_deviation_m: float,
    max_rotation_deviation_deg: float,
    min_inliers: int = 3,
) -> CalibrationEstimate:
    """Robustly estimate a fixed transform and reject inconsistent samples.

    A component-wise translation median and a rotation medoid seed the inlier
    selection. The final estimate is refit twice from the accepted samples.
    """
    if max_translation_deviation_m <= 0:
        raise ValueError("max_translation_deviation_m must be positive")
    if max_rotation_deviation_deg <= 0:
        raise ValueError("max_rotation_deviation_deg must be positive")
    if min_inliers < 1:
        raise ValueError("min_inliers must be at least 1")

    matrices = _stack_transforms(transforms)
    if len(matrices) < min_inliers:
        raise ValueError(
            f"Need at least {min_inliers} transforms, received {len(matrices)}"
        )

    seed = np.eye(4, dtype=np.float64)
    seed[:3, :3] = _rotation_medoid(matrices[:, :3, :3])
    seed[:3, 3] = np.median(matrices[:, :3, 3], axis=0)

    inliers = _inlier_mask(
        matrices,
        seed,
        max_translation_deviation_m,
        max_rotation_deviation_deg,
    )

    for _ in range(2):
        if np.count_nonzero(inliers) < min_inliers:
            break
        estimate = mean_transform(matrices[inliers])
        inliers = _inlier_mask(
            matrices,
            estimate,
            max_translation_deviation_m,
            max_rotation_deviation_deg,
        )

    if np.count_nonzero(inliers) < min_inliers:
        raise ValueError(
            "Calibration rejected too many samples: "
            f"{np.count_nonzero(inliers)}/{len(matrices)} inliers remain"
        )

    estimate = mean_transform(matrices[inliers])
    translation_errors, rotation_errors = _errors(matrices, estimate)
    return CalibrationEstimate(
        transform=estimate,
        inlier_mask=inliers,
        translation_errors_m=translation_errors,
        rotation_errors_deg=rotation_errors,
    )


def _stack_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    matrices = [
        validate_transform(item, f"transform[{index}]")
        for index, item in enumerate(transforms)
    ]
    if not matrices:
        raise ValueError("At least one transform is required")
    return np.stack(matrices)


def _rotation_medoid(rotations: np.ndarray) -> np.ndarray:
    count = len(rotations)
    distances = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = rotation_distance_degrees(rotations[i], rotations[j])
            distances[i, j] = distance
            distances[j, i] = distance
    return rotations[int(np.argmin(np.median(distances, axis=1)))]


def _errors(matrices: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation_errors = np.linalg.norm(
        matrices[:, :3, 3] - estimate[:3, 3], axis=1
    )
    estimate_rotation = Rotation.from_matrix(estimate[:3, :3])
    sample_rotations = Rotation.from_matrix(matrices[:, :3, :3])
    rotation_errors = np.degrees((estimate_rotation.inv() * sample_rotations).magnitude())
    return translation_errors, rotation_errors


def _inlier_mask(
    matrices: np.ndarray,
    estimate: np.ndarray,
    max_translation_deviation_m: float,
    max_rotation_deviation_deg: float,
) -> np.ndarray:
    translation_errors, rotation_errors = _errors(matrices, estimate)
    return (translation_errors <= max_translation_deviation_m) & (
        rotation_errors <= max_rotation_deviation_deg
    )
