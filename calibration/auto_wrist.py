#!/usr/bin/env python3
"""Step 1 (automatic): the robot moves the wrist camera around a fixed board.

THIS SCRIPT MOVES THE ROBOT. Keep a hand near the e-stop.

1. Tape the 7x5 board flat on the table.
2. Freedrive the robot so the wrist camera looks at the board from ~30 cm,
   roughly straight down, with free space all around. This pose is "home".
3. Plan and check only (no motion):
       .venv/bin/python calibration/auto_wrist.py --dry-run
4. Put the pendant in Remote Control mode, then run:
       .venv/bin/python calibration/auto_wrist.py

The script finds the board from home, plans poses on a dome around it (several
tilts, directions, distances and wrist twists, always aimed at the board),
and rejects any pose that would bring the gripper near the table, go far from
home, need a large joint swing, or fails the controller's IK/safety check. It
moves slowly in straight lines and returns home between poses.
Press Esc or q in the camera window to stop the arm immediately.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_wrist  # noqa: E402
from geometry import invert, make_transform, project, rotation_angle_deg, solve_pnp, to_vector  # noqa: E402
from hardware import (  # noqa: E402
    CYAN, GREEN, RED, RESULTS_DIR, YELLOW,
    Board, Capture, RobotReader, camera_intrinsics, capture_stationary, latest_result_file,
    load_config, open_cameras, put_text, save_captures,
)

CAMERA = "camera2"


class Abort(Exception):
    pass


@dataclass
class PlannedPose:
    label: str
    T_base_ee: np.ndarray
    clearance_m: float          # lowest gripper point above the board along the path from home
    distance_from_home_m: float
    rotation_from_home_deg: float
    problem: str = ""           # empty = OK so far


def ur_pose(T: np.ndarray) -> list[float]:
    """4x4 -> UR [x, y, z, rx, ry, rz]."""
    v = to_vector(T)
    return [float(x) for x in np.concatenate([v[3:], v[:3]])]


def rough_wrist_calibration(config: dict) -> tuple[np.ndarray, str]:
    """Any reasonable T_ee_camera2 is enough to aim the camera."""
    path = latest_result_file(CAMERA, "T_ee_camera2.npy")
    if path is None:
        path = (Path(__file__).resolve().parent / config["previous_results"]["camera2"]).resolve()
    return np.load(path), str(path)


def lowest_gripper_point(T_base_ee: np.ndarray, gripper_length: float) -> float:
    """Lowest z of the flange and the fingertip centre."""
    flange = T_base_ee[:3, 3]
    tip = (T_base_ee @ np.array([0.0, 0.0, gripper_length, 1.0]))[:3]
    return float(min(flange[2], tip[2]))


def path_clearance(T_home: np.ndarray, T_target: np.ndarray, board_z: float, gripper_length: float) -> float:
    """Minimum gripper height above the board along a straight-line move."""
    slerp = Slerp([0, 1], Rotation.from_matrix([T_home[:3, :3], T_target[:3, :3]]))
    lowest = np.inf
    for s in np.linspace(0.0, 1.0, 11):
        T = make_transform(slerp([s]).as_matrix()[0], (1 - s) * T_home[:3, 3] + s * T_target[:3, 3])
        lowest = min(lowest, lowest_gripper_point(T, gripper_length))
    return float(lowest - board_z)


def plan_poses(T_base_board: np.ndarray, board: Board, T_home_ee: np.ndarray, X: np.ndarray,
               K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int], cfg: dict) -> list[PlannedPose]:
    """Camera poses on a dome around the board centre, all looking at it."""
    board_center = (T_base_board @ np.append(board.center, 1.0))[:3]
    T_home_cam = T_home_ee @ X
    x_ref = T_home_cam[:3, 0]
    radii = cfg["radii_m"]
    twist = float(cfg["twist_deg"])
    n_az = int(cfg["azimuths"])

    specs = [(0.0, 0.0, radii[0], -twist), (0.0, 0.0, radii[-1], twist)]
    for j, tilt in enumerate(cfg["tilts_deg"]):
        for k in range(n_az):
            azimuth = 360.0 * k / n_az + j * 180.0 / n_az
            specs.append((float(tilt), azimuth, radii[(k + j) % len(radii)], twist if (k // 2 + j) % 2 else -twist))

    planned = []
    for tilt, azimuth, radius, tw in specs:
        t, a = np.radians(tilt), np.radians(azimuth)
        d = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])  # board -> camera
        z = -d
        x = x_ref - np.dot(x_ref, z) * z
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        x /= np.linalg.norm(x)
        x = np.cos(np.radians(tw)) * x + np.sin(np.radians(tw)) * np.cross(z, x)
        T_cam = make_transform(np.column_stack([x, np.cross(z, x), z]), board_center + radius * d)
        T_ee = T_cam @ invert(X)

        pose = PlannedPose(
            label=f"tilt {tilt:4.0f} dir {azimuth:5.1f} r {radius * 100:3.0f}cm twist {tw:+4.0f}",
            T_base_ee=T_ee,
            clearance_m=path_clearance(T_home_ee, T_ee, board_center[2], cfg["gripper_length_m"]),
            distance_from_home_m=float(np.linalg.norm(T_ee[:3, 3] - T_home_ee[:3, 3])),
            rotation_from_home_deg=rotation_angle_deg(T_home_ee[:3, :3], T_ee[:3, :3]),
        )
        pixels = project(invert(T_cam) @ T_base_board, board.corner_points, K, dist)
        margin = 20
        in_view = np.all((pixels >= margin) & (pixels <= np.array(image_size) - margin))
        if not in_view:
            pose.problem = "board would leave the image"
        elif pose.clearance_m < cfg["min_clearance_m"]:
            pose.problem = f"gripper only {pose.clearance_m * 100:.0f} cm above board"
        elif pose.distance_from_home_m > cfg["max_distance_from_home_m"]:
            pose.problem = f"{pose.distance_from_home_m * 100:.0f} cm from home"
        elif pose.rotation_from_home_deg > cfg["max_rotation_from_home_deg"]:
            pose.problem = f"{pose.rotation_from_home_deg:.0f} deg from home orientation"
        planned.append(pose)
    return planned


def print_plan(planned: list[PlannedPose]) -> None:
    print(f"\n  #   {'pose':44s} clearance  from home   status")
    for i, p in enumerate(planned, start=1):
        print(f"  {i:<3d} {p.label:44s} {p.clearance_m * 100:6.1f}cm  {p.distance_from_home_m * 100:5.1f}cm "
              f"{p.rotation_from_home_deg:4.0f}deg  {p.problem or 'OK'}")
    ok = sum(1 for p in planned if not p.problem)
    print(f"\n  {ok}/{len(planned)} poses usable")


def find_board(cameras, robot, board, config, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Board pose in base (via rough calibration), plus home pose and joints."""
    capture, message = capture_stationary(cameras, robot, board, config["capture"])
    if capture is None:
        raise SystemExit(f"Cannot see the board from home: {message}")
    cam = cameras[CAMERA]
    T_cam_board, _ = solve_pnp(capture.observations[CAMERA], cam.K, cam.dist)
    T_base_board = capture.T_base_ee @ X @ T_cam_board
    normal_tilt = np.degrees(np.arccos(abs(T_base_board[2, 2])))
    if normal_tilt > 15:
        raise SystemExit(f"Board is tilted {normal_tilt:.0f} deg; lay it flat on the table")
    center = (T_base_board @ np.append(board.center, 1.0))[:3]
    print(f"Board centre in base (approx): {np.round(center, 3)} m, "
          f"{np.linalg.norm(T_cam_board[:3, 3]) * 100:.0f} cm from the camera")
    return T_base_board, capture.T_base_ee, np.asarray(robot.receiver.getActualQ())


def preview(cam, board, title: str, lines: list[str]) -> int:
    image = cam.frame()
    img, status, ok = board.annotate(image, board.detect(image), cam.K, cam.dist)
    put_text(img, status, 0, GREEN if ok else RED)
    for row, text in enumerate(lines, start=1):
        put_text(img, text, row, YELLOW)
    put_text(img, "Esc / q = STOP", -1, RED)
    cv2.imshow(title, img)
    return cv2.waitKey(10) & 0xFF


def move_l(rtde_c, robot: RobotReader, T: np.ndarray, cfg: dict, cam, board, title: str, label: str) -> None:
    target = ur_pose(T)
    rtde_c.moveL(target, cfg["speed"], cfg["acceleration"], True)
    time.sleep(0.1)
    while rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning():
        if preview(cam, board, title, [label]) in (27, ord("q")):
            rtde_c.stopL(1.0)
            raise Abort("stopped by user")
        if robot.receiver.getSafetyMode() != 1:
            raise Abort(f"robot safety mode {robot.receiver.getSafetyMode()} (protective stop?)")
    reached = robot.T_base_ee()
    error = np.linalg.norm(reached[:3, 3] - T[:3, 3])
    if error > 0.003:
        raise Abort(f"robot did not reach the target ({error * 1000:.1f} mm away)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic wrist camera2 hand-eye collection (MOVES THE ROBOT)")
    parser.add_argument("--dry-run", action="store_true", help="plan and check only; the robot does not move")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["auto_wrist"]

    X, X_source = rough_wrist_calibration(config)
    print(f"Aiming with rough calibration: {X_source}")
    board = Board(config["board"], config["capture"]["minimum_corners"])
    robot = RobotReader(config["robot"]["ip"])
    cameras = open_cameras(config, [CAMERA])
    cam = cameras[CAMERA]
    title = "Automatic wrist calibration"
    rtde_c = None
    captures: list[Capture] = []
    autosave = RESULTS_DIR / CAMERA / "unsaved_session.npz"
    autosave.parent.mkdir(parents=True, exist_ok=True)

    try:
        if robot.speed() > config["capture"]["max_tcp_speed"]:
            raise SystemExit("Robot is moving; stop it at the home pose first")
        T_base_board, T_home, q_home = find_board(cameras, robot, board, config, X)
        image_size = (cam.intrinsics["width"], cam.intrinsics["height"])
        planned = plan_poses(T_base_board, board, T_home, X, cam.K, cam.dist, image_size, cfg)

        if not args.dry_run:
            import rtde_control
            print("Connecting robot control (pendant must be in Remote Control mode)...")
            rtde_c = rtde_control.RTDEControlInterface(config["robot"]["ip"])
            max_joint = np.radians(cfg["max_joint_change_deg"])
            for p in planned:
                if p.problem:
                    continue
                pose = ur_pose(p.T_base_ee)
                if not rtde_c.isPoseWithinSafetyLimits(pose):
                    p.problem = "outside robot safety limits"
                elif not rtde_c.getInverseKinematicsHasSolution(pose, list(q_home)):
                    p.problem = "unreachable (no IK)"
                else:
                    q = np.asarray(rtde_c.getInverseKinematics(pose, list(q_home)))
                    if np.max(np.abs(q - q_home)) > max_joint:
                        p.problem = f"joint swing {np.degrees(np.max(np.abs(q - q_home))):.0f} deg"

        print_plan(planned)
        usable = [p for p in planned if not p.problem]
        if args.dry_run:
            print("\nDry run: the robot did not move. Without --dry-run the controller also checks"
                  " reachability and safety limits.")
            return
        if len(usable) < config["wrist"]["minimum_captures"]:
            raise SystemExit("Too few usable poses; move home closer to the middle of free space and retry")

        input(f"\nRobot will visit {len(usable)} poses at {cfg['speed'] * 100:.0f} cm/s, returning home between "
              "each.\nHand near the e-stop. Press Enter to start (Ctrl+C to cancel)... ")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        for i, p in enumerate(usable, start=1):
            label = f"{i}/{len(usable)} {p.label}"
            print(f"\n[{label}]")
            move_l(rtde_c, robot, p.T_base_ee, cfg, cam, board, title, label)
            settle_until = time.time() + cfg["settle_s"]
            while time.time() < settle_until:
                if preview(cam, board, title, [label, "settling"]) in (27, ord("q")):
                    raise Abort("stopped by user")
            capture, message = capture_stationary(cameras, robot, board, config["capture"])
            print(("  OK   " if capture else "  FAIL ") + message)
            if capture:
                captures.append(capture)
                save_captures(autosave, captures, camera_intrinsics(cameras), board)
            move_l(rtde_c, robot, T_home, cfg, cam, board, title, f"{label} -> home")
        print(f"\nCollected {len(captures)} captures; robot is back home.")
    except Abort as reason:
        print(f"\nSTOPPED: {reason}. The robot stays where it is; Freedrive it back if needed.")
    finally:
        cv2.destroyAllWindows()
        if rtde_c is not None:
            try:
                rtde_c.stopScript()
                rtde_c.disconnect()
            except Exception:
                pass
        intrinsics = camera_intrinsics(cameras)
        for c in cameras.values():
            c.stop()
        robot.close()

    if len(captures) >= 3:
        result = calibrate_wrist.solve(captures, intrinsics, board)
        calibrate_wrist.report(result, captures, config)
        if len(captures) >= config["wrist"]["minimum_captures"]:
            calibrate_wrist.save(result, captures, intrinsics, board, config, "auto_wrist.py")
        else:
            print(f"Not saved automatically (fewer than {config['wrist']['minimum_captures']} captures). "
                  f"Captures are in {autosave}; solve with calibrate_wrist.py --from-captures.")


if __name__ == "__main__":
    main()
