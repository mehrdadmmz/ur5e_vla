"""Pure transform math and calibration solvers (no hardware imports).

Transform names follow ``T_A_B``: the pose of frame B expressed in frame A, so
a point in B maps into A with ``p_A = T_A_B @ p_B``.

Solvers:
- ``solve_wrist``: eye-in-hand, finds T_ee_camera for the wrist camera from a
  stationary board seen at many robot poses.
- ``solve_fixed``: eye-to-hand, finds T_base_camera for a fixed camera from a
  board seen at the same time by the (already calibrated) wrist camera.

Both start from a closed-form estimate and then refine every unknown jointly by
minimising ChArUco corner reprojection error in pixels (bundle adjustment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


# =============================================================================
# Transform helpers
# =============================================================================

def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return make_transform(R.T, -R.T @ T[:3, 3])


def to_vector(T: np.ndarray) -> np.ndarray:
    """4x4 transform -> [rotvec(3), translation(3)]."""
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def from_vector(v: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])


def tcp_pose_to_transform(tcp_pose: Sequence[float]) -> np.ndarray:
    """UR [x, y, z, rx, ry, rz] (axis-angle) -> T_base_ee."""
    pose = np.asarray(tcp_pose, dtype=np.float64)
    return make_transform(Rotation.from_rotvec(pose[3:6]).as_matrix(), pose[:3])


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))


def mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(transforms)
    rotation = Rotation.from_matrix(stack[:, :3, :3]).mean().as_matrix()
    return make_transform(rotation, stack[:, :3, 3].mean(axis=0))


def robust_mean_transform(
    transforms: Sequence[np.ndarray],
    max_translation_m: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median/medoid-seeded mean with outlier rejection. Returns (T, inlier_mask)."""
    stack = np.stack(transforms)
    count = len(stack)
    angles = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            angles[i, j] = angles[j, i] = rotation_angle_deg(stack[i, :3, :3], stack[j, :3, :3])
    estimate = make_transform(
        stack[int(np.argmin(np.median(angles, axis=1))), :3, :3],
        np.median(stack[:, :3, 3], axis=0),
    )
    inliers = np.ones(count, dtype=bool)
    for _ in range(3):
        t_err, r_err = transform_errors(stack, estimate)
        inliers = (t_err <= max_translation_m) & (r_err <= max_rotation_deg)
        if not inliers.any():
            break
        estimate = mean_transform(stack[inliers])
    return estimate, inliers


def transform_errors(transforms: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation (m) and rotation (deg) distance of each transform to reference."""
    stack = np.asarray(transforms)
    t_err = np.linalg.norm(stack[:, :3, 3] - reference[:3, 3], axis=1)
    r_err = np.degrees(
        (Rotation.from_matrix(reference[:3, :3]).inv() * Rotation.from_matrix(stack[:, :3, :3])).magnitude()
    )
    return t_err, r_err


def rotation_span_deg(transforms: Sequence[np.ndarray]) -> float:
    """Largest pairwise rotation between poses (a rough pose-diversity measure)."""
    stack = np.stack(transforms)
    span = 0.0
    for i in range(len(stack)):
        for j in range(i + 1, len(stack)):
            span = max(span, rotation_angle_deg(stack[i, :3, :3], stack[j, :3, :3]))
    return span


# =============================================================================
# Board observations
# =============================================================================

@dataclass
class Observation:
    """ChArUco corners seen by one camera at one capture."""

    object_points: np.ndarray  # (N, 3) in board frame, metres
    image_points: np.ndarray   # (N, 2) pixels
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))


def project(T_camera_board: np.ndarray, object_points: np.ndarray,
            K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rvec = Rotation.from_matrix(T_camera_board[:3, :3]).as_rotvec()
    projected, _ = cv2.projectPoints(object_points.reshape(-1, 1, 3), rvec, T_camera_board[:3, 3], K, dist)
    return projected.reshape(-1, 2)


def solve_pnp(obs: Observation, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Board pose in the camera from one observation. Returns (T_camera_board, rms_px)."""
    ok, rvec, tvec = cv2.solvePnP(
        obs.object_points.reshape(-1, 1, 3), obs.image_points.reshape(-1, 1, 2),
        K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    T = make_transform(cv2.Rodrigues(rvec)[0], tvec)
    return T, reprojection_rms(T, obs, K, dist)


def reprojection_rms(T_camera_board: np.ndarray, obs: Observation, K: np.ndarray, dist: np.ndarray) -> float:
    diff = project(T_camera_board, obs.object_points, K, dist) - obs.image_points
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _covariance_sigmas(result, block: slice) -> tuple[float, float]:
    """Approximate 1-sigma (mm, deg) of one 6-DoF block from the solver Jacobian."""
    J = result.jac
    J = J.toarray() if hasattr(J, "toarray") else J
    dof = max(1, J.shape[0] - J.shape[1])
    variance = 2.0 * result.cost / dof
    try:
        cov = np.linalg.pinv(J.T @ J)[block, block] * variance
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    sigma_rot_deg = float(np.degrees(np.sqrt(np.trace(cov[:3, :3]))))
    sigma_trans_mm = float(1000.0 * np.sqrt(np.trace(cov[3:, 3:])))
    return sigma_trans_mm, sigma_rot_deg




def _keep_consistent(per_capture_rms: np.ndarray, inliers: np.ndarray,
                     min_px: float, factor: float) -> np.ndarray:
    """Drop captures whose reprojection error is far above the others."""
    limit = max(min_px, factor * float(np.median(per_capture_rms[inliers])))
    return inliers & (per_capture_rms <= limit)


# =============================================================================
# Wrist camera (eye-in-hand)
# =============================================================================

@dataclass
class WristResult:
    T_ee_camera: np.ndarray
    T_base_board: np.ndarray
    inlier_mask: np.ndarray
    rms_px: float
    per_capture_rms_px: np.ndarray
    # Board position implied by each capture (via its own PnP) vs the solution.
    board_center_errors_mm: np.ndarray
    board_rotation_errors_deg: np.ndarray
    rotation_span_deg: float
    sigma_translation_mm: float
    sigma_rotation_deg: float
    initial_T_ee_camera: np.ndarray
    # Board scale that best matches the robot's metric motion (1.0 = as configured).
    implied_board_scale: float = float("nan")

    def metrics(self) -> dict:
        m = self.inlier_mask
        e = self.board_center_errors_mm[m]
        return {
            "num_captures": int(m.size),
            "num_inliers": int(np.count_nonzero(m)),
            "reprojection_rms_px": self.rms_px,
            "reprojection_worst_capture_px": float(np.max(self.per_capture_rms_px[m])),
            "board_consistency_rms_mm": float(np.sqrt(np.mean(e**2))),
            "board_consistency_max_mm": float(np.max(e)),
            "board_rotation_consistency_max_deg": float(np.max(self.board_rotation_errors_deg[m])),
            "robot_rotation_span_deg": self.rotation_span_deg,
            "approx_sigma_translation_mm": self.sigma_translation_mm,
            "approx_sigma_rotation_deg": self.sigma_rotation_deg,
            "implied_board_scale": self.implied_board_scale,
        }


def hand_eye_park_martin(T_base_ee: Sequence[np.ndarray], T_camera_board: Sequence[np.ndarray],
                         min_pair_rotation_deg: float = 5.0) -> np.ndarray:
    """Closed-form AX = XB over all pose pairs. Returns T_ee_camera."""
    alphas, betas, As, Bs = [], [], [], []
    n = len(T_base_ee)
    for i in range(n):
        for j in range(i + 1, n):
            A = invert(T_base_ee[i]) @ T_base_ee[j]
            B = T_camera_board[i] @ invert(T_camera_board[j])
            a = Rotation.from_matrix(A[:3, :3]).as_rotvec()
            if np.degrees(np.linalg.norm(a)) < min_pair_rotation_deg:
                continue
            alphas.append(a)
            betas.append(Rotation.from_matrix(B[:3, :3]).as_rotvec())
            As.append(A)
            Bs.append(B)
    if len(alphas) < 2:
        raise ValueError("Not enough rotation between poses; tilt/rotate the wrist more")

    # Rotation: alpha = R_X beta, solved with Kabsch.
    H = np.stack(betas).T @ np.stack(alphas)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R_X = Vt.T @ D @ U.T

    # Translation: (R_A - I) t_X = R_X t_B - t_A
    lhs = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    rhs = np.concatenate([R_X @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t_X = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return make_transform(R_X, t_X)


def solve_wrist(
    T_base_ee: Sequence[np.ndarray],
    observations: Sequence[Observation],
    K: np.ndarray,
    dist: np.ndarray,
    board_center: np.ndarray,
    outlier_min_px: float = 1.5,
    outlier_factor: float = 3.0,
) -> WristResult:
    """Board fixed in the base frame, camera on the wrist. Unknowns: X and the board pose."""
    if len(T_base_ee) != len(observations):
        raise ValueError("Each capture needs one robot pose and one observation")
    if len(T_base_ee) < 3:
        raise ValueError("Need at least 3 captures")

    T_cam_board = [solve_pnp(obs, K, dist)[0] for obs in observations]
    X0 = hand_eye_park_martin(T_base_ee, T_cam_board)
    B0 = mean_transform([T @ X0 @ C for T, C in zip(T_base_ee, T_cam_board)])

    def predicted_camera_board(params: np.ndarray, T: np.ndarray) -> np.ndarray:
        return invert(T @ from_vector(params[:6])) @ from_vector(params[6:12])

    inliers = np.ones(len(T_base_ee), dtype=bool)
    params = np.concatenate([to_vector(X0), to_vector(B0)])
    for _ in range(3):
        used = np.flatnonzero(inliers)

        def residuals(p: np.ndarray) -> np.ndarray:
            return np.concatenate([
                (project(predicted_camera_board(p, T_base_ee[i]), observations[i].object_points, K, dist)
                 - observations[i].image_points).ravel()
                for i in used
            ])

        result = least_squares(residuals, params, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000)
        params = result.x
        per_capture = np.array([
            reprojection_rms(predicted_camera_board(params, T), obs, K, dist)
            for T, obs in zip(T_base_ee, observations)
        ])
        kept = _keep_consistent(per_capture, inliers, outlier_min_px, outlier_factor)
        if np.count_nonzero(kept) < 3:
            raise ValueError("Captures disagree too much; check the board did not move")
        if np.array_equal(kept, inliers):
            break
        inliers = kept

    X = from_vector(params[:6])
    B = from_vector(params[6:12])
    sq = [np.sum((project(predicted_camera_board(params, T_base_ee[i]), observations[i].object_points, K, dist)
                  - observations[i].image_points) ** 2, axis=1) for i in np.flatnonzero(inliers)]
    implied = np.stack([T @ X @ C for T, C in zip(T_base_ee, T_cam_board)])
    center_h = np.append(board_center, 1.0)
    center_errors = np.linalg.norm((implied @ center_h)[:, :3] - (B @ center_h)[:3], axis=1) * 1000.0
    _, rotation_errors = transform_errors(implied, B)
    sigma_t, sigma_r = _covariance_sigmas(result, slice(0, 6))

    # Diagnostic only: let the board scale float. The robot moves in metres, so
    # with enough height/tilt variation this checks the printed square size.
    used = np.flatnonzero(inliers)

    def scaled_residuals(p: np.ndarray) -> np.ndarray:
        return np.concatenate([
            (project(predicted_camera_board(p, T_base_ee[i]), observations[i].object_points * p[12], K, dist)
             - observations[i].image_points).ravel()
            for i in used
        ])

    scale_fit = least_squares(scaled_residuals, np.append(params, 1.0), loss="soft_l1", f_scale=1.0,
                              x_scale="jac", max_nfev=2000)

    return WristResult(
        T_ee_camera=X,
        T_base_board=B,
        inlier_mask=inliers,
        rms_px=float(np.sqrt(np.mean(np.concatenate(sq)))),
        per_capture_rms_px=per_capture,
        board_center_errors_mm=center_errors,
        board_rotation_errors_deg=rotation_errors,
        rotation_span_deg=rotation_span_deg([T_base_ee[i] for i in np.flatnonzero(inliers)]),
        sigma_translation_mm=sigma_t,
        sigma_rotation_deg=sigma_r,
        initial_T_ee_camera=X0,
        implied_board_scale=float(scale_fit.x[12]),
    )


# =============================================================================
# Fixed camera (eye-to-hand through the wrist camera)
# =============================================================================

@dataclass
class FixedCapture:
    T_base_ee: np.ndarray
    wrist: Observation
    fixed: Observation


@dataclass
class FixedResult:
    T_base_camera: np.ndarray
    inlier_mask: np.ndarray
    wrist_rms_px: float
    fixed_rms_px: float
    per_capture_wrist_rms_px: np.ndarray
    per_capture_fixed_rms_px: np.ndarray
    # Where the two cameras each put the board centre (base frame), per capture.
    board_disagreement_mm: np.ndarray
    candidate_translation_errors_mm: np.ndarray
    candidate_rotation_errors_deg: np.ndarray
    sigma_translation_mm: float
    sigma_rotation_deg: float
    initial_T_base_camera: np.ndarray

    def metrics(self) -> dict:
        m = self.inlier_mask
        d = self.board_disagreement_mm[m]
        return {
            "num_captures": int(m.size),
            "num_inliers": int(np.count_nonzero(m)),
            "wrist_reprojection_rms_px": self.wrist_rms_px,
            "fixed_reprojection_rms_px": self.fixed_rms_px,
            "board_disagreement_rms_mm": float(np.sqrt(np.mean(d**2))),
            "board_disagreement_max_mm": float(np.max(d)),
            "candidate_translation_rmse_mm": float(np.sqrt(np.mean(self.candidate_translation_errors_mm[m] ** 2))),
            "candidate_rotation_rmse_deg": float(np.sqrt(np.mean(self.candidate_rotation_errors_deg[m] ** 2))),
            "approx_sigma_translation_mm": self.sigma_translation_mm,
            "approx_sigma_rotation_deg": self.sigma_rotation_deg,
        }


def solve_fixed(
    T_ee_wrist: np.ndarray,
    captures: Sequence[FixedCapture],
    K_wrist: np.ndarray, dist_wrist: np.ndarray,
    K_fixed: np.ndarray, dist_fixed: np.ndarray,
    board_center: np.ndarray,
    max_translation_m: float = 0.03,
    max_rotation_deg: float = 5.0,
    outlier_min_px: float = 1.5,
    outlier_factor: float = 3.0,
) -> FixedResult:
    """Board moves between captures; both cameras see it at each one.
    Unknowns: the fixed camera pose F and one board pose per capture."""
    if len(captures) < 3:
        raise ValueError("Need at least 3 captures")

    T_c2_board = [solve_pnp(c.wrist, K_wrist, dist_wrist)[0] for c in captures]
    T_cf_board = [solve_pnp(c.fixed, K_fixed, dist_fixed)[0] for c in captures]
    wrist_boards = [c.T_base_ee @ T_ee_wrist @ Tw for c, Tw in zip(captures, T_c2_board)]
    candidates = np.stack([B @ invert(Tf) for B, Tf in zip(wrist_boards, T_cf_board)])
    F0, inliers = robust_mean_transform(candidates, max_translation_m, max_rotation_deg)
    if np.count_nonzero(inliers) < 3:
        raise ValueError(f"Too many outlier captures: {np.count_nonzero(inliers)}/{len(captures)} consistent")

    T_c2_base = [invert(c.T_base_ee @ T_ee_wrist) for c in captures]

    def capture_residuals(F: np.ndarray, B: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        c = captures[i]
        rw = project(T_c2_base[i] @ B, c.wrist.object_points, K_wrist, dist_wrist) - c.wrist.image_points
        rf = project(invert(F) @ B, c.fixed.object_points, K_fixed, dist_fixed) - c.fixed.image_points
        return rw, rf

    F = F0
    boards = {i: wrist_boards[i] for i in range(len(captures))}
    for _ in range(3):
        used = list(np.flatnonzero(inliers))
        rows = [2 * (len(captures[i].wrist.object_points) + len(captures[i].fixed.object_points)) for i in used]
        sparsity = lil_matrix((sum(rows), 6 + 6 * len(used)), dtype=int)
        row = 0
        for k, i in enumerate(used):
            n_wrist = 2 * len(captures[i].wrist.object_points)
            sparsity[row:row + rows[k], 6 + 6 * k:12 + 6 * k] = 1
            sparsity[row + n_wrist:row + rows[k], :6] = 1
            row += rows[k]

        def residuals(p: np.ndarray) -> np.ndarray:
            F_p = from_vector(p[:6])
            out = []
            for k, i in enumerate(used):
                rw, rf = capture_residuals(F_p, from_vector(p[6 + 6 * k:12 + 6 * k]), i)
                out += [rw.ravel(), rf.ravel()]
            return np.concatenate(out)

        x0 = np.concatenate([to_vector(F)] + [to_vector(boards[i]) for i in used])
        result = least_squares(residuals, x0, jac_sparsity=sparsity,
                               loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=2000)
        F = from_vector(result.x[:6])
        for k, i in enumerate(used):
            boards[i] = from_vector(result.x[6 + 6 * k:12 + 6 * k])

        combined = np.array([
            np.sqrt(np.mean(np.concatenate([np.sum(r**2, axis=1) for r in capture_residuals(F, boards[i], i)])))
            for i in range(len(captures))
        ])
        kept = _keep_consistent(combined, inliers, outlier_min_px, outlier_factor)
        if np.count_nonzero(kept) < 3:
            raise ValueError("Captures disagree too much; check nothing moved during capture")
        if np.array_equal(kept, inliers):
            break
        inliers = kept

    wrist_rms, fixed_rms, disagreement, wrist_sq, fixed_sq = [], [], [], [], []
    center_h = np.append(board_center, 1.0)
    for i in range(len(captures)):
        rw, rf = capture_residuals(F, boards[i], i)
        wrist_rms.append(np.sqrt(np.mean(np.sum(rw**2, axis=1))))
        fixed_rms.append(np.sqrt(np.mean(np.sum(rf**2, axis=1))))
        if inliers[i]:
            wrist_sq.append(np.sum(rw**2, axis=1))
            fixed_sq.append(np.sum(rf**2, axis=1))
        from_wrist = (wrist_boards[i] @ center_h)[:3]
        from_fixed = (F @ T_cf_board[i] @ center_h)[:3]
        disagreement.append(np.linalg.norm(from_wrist - from_fixed) * 1000.0)

    t_err, r_err = transform_errors(candidates, F)
    sigma_t, sigma_r = _covariance_sigmas(result, slice(0, 6))
    return FixedResult(
        T_base_camera=F,
        inlier_mask=inliers,
        wrist_rms_px=float(np.sqrt(np.mean(np.concatenate(wrist_sq)))),
        fixed_rms_px=float(np.sqrt(np.mean(np.concatenate(fixed_sq)))),
        per_capture_wrist_rms_px=np.array(wrist_rms),
        per_capture_fixed_rms_px=np.array(fixed_rms),
        board_disagreement_mm=np.array(disagreement),
        candidate_translation_errors_mm=t_err * 1000.0,
        candidate_rotation_errors_deg=r_err,
        sigma_translation_mm=sigma_t,
        sigma_rotation_deg=sigma_r,
        initial_T_base_camera=F0,
    )
