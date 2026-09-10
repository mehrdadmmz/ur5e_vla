from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_math import (  # noqa: E402
    camera3_in_base,
    camera3_in_camera2,
    estimate_fixed_transform,
    rotation_distance_degrees,
    validate_transform,
)


def make_transform(translation: list[float], rotation_vector: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    transform[:3, 3] = translation
    return transform


class CalibrationMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.T_ee_camera2 = make_transform(
            [-0.03006178, -0.03957967, 0.06249641],
            [-0.0073, -0.0030, 0.0112],
        )
        self.T_base_camera3 = make_transform(
            [0.72, -0.18, 0.91],
            [0.15, -0.55, 2.75],
        )

    def test_camera3_candidate_recovers_exact_fixed_transform(self) -> None:
        rng = np.random.default_rng(12)
        for _ in range(12):
            T_base_ee = make_transform(
                rng.uniform([-0.3, 0.2, 0.2], [0.3, 0.7, 0.7]).tolist(),
                rng.uniform(-1.0, 1.0, size=3).tolist(),
            )
            T_base_board = make_transform(
                rng.uniform([-0.2, 0.2, 0.0], [0.3, 0.7, 0.3]).tolist(),
                rng.uniform(-0.8, 0.8, size=3).tolist(),
            )
            T_base_camera2 = T_base_ee @ self.T_ee_camera2
            T_camera2_board = np.linalg.inv(T_base_camera2) @ T_base_board
            T_camera3_board = np.linalg.inv(self.T_base_camera3) @ T_base_board

            recovered = camera3_in_base(
                T_base_ee,
                self.T_ee_camera2,
                T_camera2_board,
                T_camera3_board,
            )
            np.testing.assert_allclose(recovered, self.T_base_camera3, atol=1e-10)

    def test_dynamic_camera_relative_transform_composes_back_to_base(self) -> None:
        T_base_ee = make_transform([0.1, 0.42, 0.36], [0.3, -0.2, 0.7])
        T_base_camera2 = T_base_ee @ self.T_ee_camera2
        T_camera2_camera3 = camera3_in_camera2(
            T_base_ee,
            self.T_ee_camera2,
            self.T_base_camera3,
        )
        np.testing.assert_allclose(
            T_base_camera2 @ T_camera2_camera3,
            self.T_base_camera3,
            atol=1e-10,
        )

    def test_robust_estimate_rejects_large_outlier(self) -> None:
        rng = np.random.default_rng(3)
        candidates = []
        for _ in range(20):
            sample = np.eye(4, dtype=np.float64)
            rotation_noise = Rotation.from_rotvec(
                rng.normal(0.0, np.radians(0.08), size=3)
            ).as_matrix()
            sample[:3, :3] = self.T_base_camera3[:3, :3] @ rotation_noise
            sample[:3, 3] = self.T_base_camera3[:3, 3] + rng.normal(
                0.0, 0.0005, size=3
            )
            candidates.append(sample)

        outlier = make_transform([1.4, 0.5, 0.2], [-1.2, 0.6, 0.1])
        candidates.append(outlier)
        estimate = estimate_fixed_transform(
            candidates,
            max_translation_deviation_m=0.01,
            max_rotation_deviation_deg=1.0,
            min_inliers=15,
        )

        self.assertEqual(estimate.num_samples, 21)
        self.assertEqual(estimate.num_inliers, 20)
        self.assertFalse(bool(estimate.inlier_mask[-1]))
        self.assertLess(
            np.linalg.norm(
                estimate.transform[:3, 3] - self.T_base_camera3[:3, 3]
            ),
            0.001,
        )
        self.assertLess(
            rotation_distance_degrees(
                estimate.transform[:3, :3], self.T_base_camera3[:3, :3]
            ),
            0.1,
        )

    def test_invalid_transform_is_rejected(self) -> None:
        invalid = np.eye(4)
        invalid[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "not orthonormal"):
            validate_transform(invalid)


if __name__ == "__main__":
    unittest.main()
