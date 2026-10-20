"""Offline checks of the calibration solvers on synthetic data (no hardware).

    .venv/bin/python -m unittest discover -s calibration/tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import (  # noqa: E402
    FixedCapture, Observation, invert, make_transform, project, rotation_angle_deg,
    solve_fixed, solve_wrist,
)

K = np.array([[610.0, 0.0, 326.0], [0.0, 610.0, 245.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


def grid(nx: int, ny: int, square: float) -> np.ndarray:
    return np.array([[x * square, y * square, 0.0] for y in range(1, ny) for x in range(1, nx)])


def look_at(eye, target, roll_deg=0.0) -> np.ndarray:
    """Camera pose (z forward, y down) at eye looking at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z]) @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    return make_transform(R, eye)


def observe(T_cam_board, points, rng, noise_px=0.2) -> Observation:
    pixels = project(T_cam_board, points, K, DIST) + rng.normal(0, noise_px, (len(points), 2))
    assert np.all((pixels > 0) & (pixels < [640, 480])), "synthetic board left the image"
    return Observation(points, pixels)


class WristSolverTest(unittest.TestCase):
    def test_recovers_hand_eye(self):
        rng = np.random.default_rng(0)
        points = grid(4, 4, 0.032)
        center = points.mean(axis=0)
        X = make_transform(Rotation.from_euler("xyz", [1.0, -2.0, 0.7], degrees=True).as_matrix(),
                           [-0.030, -0.040, 0.062])
        B = make_transform(np.eye(3), [0.0, 0.45, 0.0])
        target = (B @ np.append(center, 1))[:3]

        poses, observations = [], []
        for k in range(20):
            eye = target + np.array([*rng.uniform(-0.15, 0.15, 2), rng.uniform(0.28, 0.42)])
            T_base_cam = look_at(eye, target + rng.uniform(-0.02, 0.02, 3), rng.uniform(-60, 60))
            poses.append(T_base_cam @ invert(X))
            obs = observe(invert(T_base_cam) @ B, points, rng)
            if k == 7:  # robot pose read while the arm was still settling
                poses[-1] = poses[-1] @ make_transform(np.eye(3), [0.01, 0.0, 0.0])
            observations.append(obs)

        result = solve_wrist(poses, observations, K, DIST, center)
        self.assertLess(np.linalg.norm(result.T_ee_camera[:3, 3] - X[:3, 3]), 0.001)
        self.assertLess(rotation_angle_deg(result.T_ee_camera[:3, :3], X[:3, :3]), 0.1)
        self.assertLess(result.rms_px, 0.4)
        self.assertLess(result.metrics()["board_consistency_rms_mm"], 2.0)
        self.assertFalse(result.inlier_mask[7])
        self.assertEqual(int(result.inlier_mask.sum()), 19)


class FixedSolverTest(unittest.TestCase):
    def test_recovers_fixed_camera_and_rejects_outlier(self):
        rng = np.random.default_rng(1)
        points = grid(7, 5, 0.025)
        center = points.mean(axis=0)
        X = make_transform(Rotation.from_euler("xyz", [0.5, 0.3, -0.4], degrees=True).as_matrix(),
                           [-0.030, -0.040, 0.062])
        F = look_at(np.array([0.06, 1.06, 0.58]), np.array([0.0, 0.55, 0.05]))

        captures = []
        for k in range(20):
            spot = np.array([rng.uniform(-0.12, 0.12), rng.uniform(0.45, 0.65), rng.uniform(0.0, 0.08)])
            # Board faces up and toward camera F, with some random tilt.
            R_face = look_at(spot, (F[:3, 3] + np.array([0, 0, 0.6])) / 1, 0)[:3, :3] @ Rotation.from_euler(
                "xyz", [180 + rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-30, 30)], degrees=True
            ).as_matrix()
            B = make_transform(R_face, spot - R_face @ center)
            T_base_cam2 = look_at(spot + np.array([*rng.uniform(-0.1, 0.1, 2), rng.uniform(0.3, 0.4)]), spot,
                                  rng.uniform(-40, 40))
            T_base_ee = T_base_cam2 @ invert(X)
            wrist = observe(invert(T_base_cam2) @ B, points, rng)
            fixed = observe(invert(F) @ B, points, rng)
            if k == 5:  # board bumped between the two camera reads
                fixed = Observation(points, fixed.image_points + 15.0)
            captures.append(FixedCapture(T_base_ee, wrist, fixed))

        result = solve_fixed(X, captures, K, DIST, K, DIST, center)
        self.assertLess(np.linalg.norm(result.T_base_camera[:3, 3] - F[:3, 3]), 0.003)
        self.assertLess(rotation_angle_deg(result.T_base_camera[:3, :3], F[:3, :3]), 0.2)
        self.assertFalse(result.inlier_mask[5])
        self.assertEqual(int(result.inlier_mask.sum()), 19)


if __name__ == "__main__":
    unittest.main()
