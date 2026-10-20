#!/usr/bin/env python3
"""Step 4: check that all calibrated cameras agree on where the board is.

Put the 7x5 board somewhere all cameras can see it. Each camera estimates the
board in the robot base frame through its own calibration; the numbers on
screen are how far apart those estimates are. Coloured crosses show where the
OTHER cameras think the board corners are (should sit on the real corners).

    .venv/bin/python calibration/validate.py
    .venv/bin/python calibration/validate.py --cameras camera2,camera3

Keys: s log a stationary measurement | q quit (prints a summary)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import invert, project, rotation_angle_deg, solve_pnp  # noqa: E402
from hardware import (  # noqa: E402
    CYAN, GREEN, RED, RESULTS_DIR,
    Board, RobotReader, capture_stationary, latest_result_file, load_config, open_cameras, put_text,
)

RESULT_FILES = {"camera1": "T_base_camera1.npy", "camera2": "T_ee_camera2.npy", "camera3": "T_base_camera3.npy"}
CROSS_COLORS = {"camera1": (255, 0, 255), "camera2": (0, 255, 255), "camera3": (255, 128, 0)}  # BGR


def load_calibrations(names: list[str], overrides: dict[str, str]) -> dict[str, np.ndarray]:
    calibrations = {}
    for name in names:
        path = Path(overrides[name]) if name in overrides else latest_result_file(name, RESULT_FILES[name])
        if path is None or not path.is_file():
            raise SystemExit(f"No calibration for {name}; run its step first or pass --{name} PATH")
        print(f"{name}: {path}")
        calibrations[name] = np.load(path)
    return calibrations


def board_in_base(name: str, T_cam_board: np.ndarray, calib: np.ndarray, T_base_ee: np.ndarray) -> np.ndarray:
    if name == "camera2":
        return T_base_ee @ calib @ T_cam_board
    return calib @ T_cam_board


def T_camera_base(name: str, calib: np.ndarray, T_base_ee: np.ndarray) -> np.ndarray:
    return invert(T_base_ee @ calib) if name == "camera2" else invert(calib)


def pairwise(boards: dict[str, np.ndarray], center: np.ndarray) -> list[tuple[str, float, float]]:
    c = np.append(center, 1.0)
    rows = []
    for a, b in itertools.combinations(sorted(boards), 2):
        dist_mm = float(np.linalg.norm((boards[a] @ c)[:3] - (boards[b] @ c)[:3]) * 1000.0)
        rows.append((f"{a}-{b}", dist_mm, rotation_angle_deg(boards[a][:3, :3], boards[b][:3, :3])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check camera calibrations")
    parser.add_argument("--cameras", default="camera1,camera2,camera3")
    parser.add_argument("--config", default=None)
    for name in RESULT_FILES:
        parser.add_argument(f"--{name}", help=f"calibration .npy for {name} (default: newest new run, else three_cam_calibration/)")
    args = parser.parse_args()
    config = load_config(args.config)
    names = [n.strip() for n in args.cameras.split(",") if n.strip()]
    overrides = {n: getattr(args, n) for n in RESULT_FILES if getattr(args, n)}
    calibrations = load_calibrations(names, overrides)

    board = Board(config["board"], config["capture"]["minimum_corners"])
    robot = RobotReader(config["robot"]["ip"])
    cameras = open_cameras(config, names)
    log_path = RESULTS_DIR / "validation" / f"validation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    logged: list[list[tuple[str, float, float]]] = []
    title = "Step 4: calibration cross-check"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print(__doc__)

    try:
        while True:
            T_base_ee = robot.T_base_ee()
            images = {n: cam.frame() for n, cam in cameras.items()}
            boards: dict[str, np.ndarray] = {}
            tiles: dict[str, np.ndarray] = {}
            for name, cam in cameras.items():
                det = board.detect(images[name])
                img, status, ok = board.annotate(images[name], det, cam.K, cam.dist)
                put_text(img, status, 0, GREEN if ok else RED)
                put_text(img, name, -1, CYAN)
                if ok:
                    T_cam_board, _ = solve_pnp(board.observation(det.ids, det.corners), cam.K, cam.dist)
                    boards[name] = board_in_base(name, T_cam_board, calibrations[name], T_base_ee)
                    p = (boards[name] @ np.append(board.center, 1.0))[:3] * 1000.0
                    put_text(img, f"board xyz mm: {p[0]:.0f} {p[1]:.0f} {p[2]:.0f}", 1, CYAN, 0.5)
                tiles[name] = img

            # Draw where the other cameras put the board corners.
            for name, cam in cameras.items():
                T_c_base = T_camera_base(name, calibrations[name], T_base_ee)
                for other, T_base_board in boards.items():
                    if other == name:
                        continue
                    T_c_board = T_c_base @ T_base_board
                    if T_c_board[2, 3] <= 0:
                        continue
                    for u, v in project(T_c_board, board.corner_points, cam.K, cam.dist):
                        if 0 <= u < tiles[name].shape[1] and 0 <= v < tiles[name].shape[0]:
                            cv2.drawMarker(tiles[name], (int(u), int(v)), CROSS_COLORS[other], cv2.MARKER_CROSS, 10, 1)

            view = np.hstack([tiles[n] for n in names])
            for row, (pair, mm, deg) in enumerate(pairwise(boards, board.center)):
                put_text(view, f"{pair}: {mm:5.1f} mm  {deg:4.2f} deg", 2 + row, GREEN if mm < 5 else RED)
            put_text(view, f"crosses: magenta=camera1 yellow=camera2 blue=camera3 | logged {len(logged)}",
                     -2, CYAN, 0.5)
            cv2.imshow(title, view)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("s"):
                capture, message = capture_stationary(cameras, robot, board, config["capture"])
                if capture is None:
                    print("FAIL " + message)
                    continue
                measured = {}
                for name, cam in cameras.items():
                    T_cam_board, _ = solve_pnp(capture.observations[name], cam.K, cam.dist)
                    measured[name] = board_in_base(name, T_cam_board, calibrations[name], capture.T_base_ee)
                rows = pairwise(measured, board.center)
                logged.append(rows)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                new_file = not log_path.exists()
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if new_file:
                        writer.writerow(["sample", "pair", "distance_mm", "rotation_deg"])
                    for pair, mm, deg in rows:
                        writer.writerow([len(logged), pair, f"{mm:.2f}", f"{deg:.3f}"])
                print(f"#{len(logged)}  " + "  |  ".join(f"{p}: {mm:.1f} mm, {d:.2f} deg" for p, mm, d in rows))
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        for cam in cameras.values():
            cam.stop()
        robot.close()

    if logged:
        print(f"\nSummary over {len(logged)} board positions (log: {log_path}):")
        for i, (pair, _, _) in enumerate(logged[0]):
            mm = np.array([rows[i][1] for rows in logged])
            deg = np.array([rows[i][2] for rows in logged])
            print(f"  {pair}: mean {mm.mean():.1f} mm, max {mm.max():.1f} mm | mean {deg.mean():.2f} deg, max {deg.max():.2f} deg")


if __name__ == "__main__":
    main()
