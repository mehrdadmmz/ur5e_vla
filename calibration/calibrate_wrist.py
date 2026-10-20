#!/usr/bin/env python3
"""Step 1: wrist camera2 hand-eye calibration (eye-in-hand) -> T_ee_camera2.

Tape the 7x5 board flat on the table (all edges); it must not move during the
session. Move the robot (Freedrive is easiest) so the camera is 25-40 cm from the
board, looking at it from many directions: tilt 20-35 deg from straight down to
the front/back/left/right, at a few heights, with some wrist twist. Press 's'
at each pose while the robot is stopped. This script never moves the robot.

    .venv/bin/python calibration/calibrate_wrist.py

Re-solve saved captures without hardware, e.g. dropping captures 3 and 7:

    .venv/bin/python calibration/calibrate_wrist.py \
        --from-captures calibration/results/camera2/result_1/captures.npz --exclude 3,7 --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import rotation_angle_deg, solve_wrist  # noqa: E402
from hardware import (  # noqa: E402
    Board, Capture, Intrinsics, compare_to_previous, load_captures, load_config,
    next_result_dir, parse_exclude, save_captures, timestamp, transform_summary, write_yaml,
)
from session import run_session  # noqa: E402

CAMERA = "camera2"


def solve(captures: list[Capture], intrinsics: Intrinsics, board: Board):
    K, dist = intrinsics[CAMERA]
    return solve_wrist(
        [c.T_base_ee for c in captures],
        [c.observations[CAMERA] for c in captures],
        K, dist, board.center,
    )


def report(result, captures: list[Capture], config: dict) -> str:
    m = result.metrics()
    t = result.T_ee_camera[:3, 3] * 1000.0
    print("\n" + "=" * 64)
    print("T_ee_camera2 (wrist camera pose in the TCP/flange frame):")
    print(np.array2string(result.T_ee_camera, precision=6, suppress_small=True))
    print(f"translation (mm): [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
    print(f"\n  captures used           : {m['num_inliers']}/{m['num_captures']}")
    print(f"  reprojection RMS        : {m['reprojection_rms_px']:.3f} px   (good < 0.5)")
    print(f"  board consistency RMS   : {m['board_consistency_rms_mm']:.2f} mm  (good < 2)")
    print(f"  board consistency max   : {m['board_consistency_max_mm']:.2f} mm")
    print(f"  tilt coverage           : {tilt_summary(captures)}  (want 20+ deg toward all of +x -x +y -y)")
    print(f"  approx 1-sigma          : {m['approx_sigma_translation_mm']:.2f} mm, "
          f"{m['approx_sigma_rotation_deg']:.3f} deg")
    square_mm = config["board"]["square_length_m"] * 1000.0
    print(f"  board size check        : robot motion implies {square_mm * result.implied_board_scale:.2f} mm squares "
          f"(config {square_mm:.2f}); trust this only with good tilt/height variety")
    change = compare_to_previous(config, CAMERA, result.T_ee_camera)
    if change:
        print(f"  vs previous calibration : {change['translation_change_mm']:.1f} mm, "
              f"{change['rotation_change_deg']:.2f} deg")

    print("\n  #    reproj px   board err mm")
    for k, (rms, err) in enumerate(zip(result.per_capture_rms_px, result.board_center_errors_mm), start=1):
        flag = ""
        if not result.inlier_mask[k - 1]:
            flag = "   <-- outlier (not used)"
        elif err > 5.0:
            flag = "   <-- suspicious, consider --exclude"
        print(f"  {k:<4d} {rms:9.3f}   {err:12.2f}{flag}")
    print("=" * 64)
    return f"{m['reprojection_rms_px']:.2f}px, board {m['board_consistency_rms_mm']:.1f}mm"


def save(result, captures: list[Capture], intrinsics: Intrinsics, board: Board, config: dict, source: str) -> Path:
    out = next_result_dir(CAMERA)
    np.save(out / "T_ee_camera2.npy", result.T_ee_camera)
    save_captures(out / "captures.npz", captures, intrinsics, board)
    K, dist = intrinsics[CAMERA]
    write_yaml(out / "report.yaml", {
        "transform_name": "T_ee_camera2",
        "convention": "p_ee = T_ee_camera2 @ p_camera2; ee = UR TCP (pendant TCP 'TCP', zero offset = flange)",
        **transform_summary(result.T_ee_camera),
        "metrics": result.metrics(),
        "change_vs_previous": compare_to_previous(config, CAMERA, result.T_ee_camera),
        "inlier_mask": result.inlier_mask,
        "per_capture_reprojection_px": result.per_capture_rms_px,
        "per_capture_board_error_mm": result.board_center_errors_mm,
        "T_base_board": result.T_base_board,
        "board": board.cfg,
        "camera": {"name": CAMERA, "serial": config["cameras"][CAMERA]["serial"], "K": K, "dist": dist},
        "source": source,
        "created_at": timestamp(),
    })
    print(f"\nSaved to {out}")
    return out


def pose_hint(config: dict):
    def hint(captures: list[Capture], T_now: np.ndarray) -> str:
        if not captures:
            return ""
        for k, c in enumerate(captures, start=1):
            if (np.linalg.norm(c.T_base_ee[:3, 3] - T_now[:3, 3]) < config["wrist"]["duplicate_translation_m"]
                    and rotation_angle_deg(c.T_base_ee[:3, :3], T_now[:3, :3]) < config["wrist"]["duplicate_rotation_deg"]):
                return f"hint: almost same pose as #{k} (s still works; tilt/twist more for better data)"
        covered = {direction for c in captures for direction, tilt in [tilt_direction(c.T_base_ee)] if tilt >= 20.0}
        missing = [d for d in TILT_DIRECTIONS if d not in covered]
        _, tilt_now = tilt_direction(T_now)
        return f"tilt now {tilt_now:.0f} deg | still need a 20+ deg tilt toward: {' '.join(missing) or 'none - good'}"
    return hint


TILT_DIRECTIONS = ("+x", "-x", "+y", "-y")


def tilt_summary(captures: list[Capture]) -> str:
    best = {d: 0.0 for d in TILT_DIRECTIONS}
    for c in captures:
        direction, tilt = tilt_direction(c.T_base_ee)
        best[direction] = max(best[direction], tilt)
    return "  ".join(f"{d}:{best[d]:.0f}deg" for d in TILT_DIRECTIONS)


def tilt_direction(T_base_ee: np.ndarray) -> tuple[str, float]:
    """Which base-frame direction the tool axis leans toward, and by how much (deg from straight down)."""
    z = T_base_ee[:3, 2]
    tilt = float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))
    if abs(z[0]) >= abs(z[1]):
        return ("+x" if z[0] > 0 else "-x"), tilt
    return ("+y" if z[1] > 0 else "-y"), tilt


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrist camera2 hand-eye calibration")
    parser.add_argument("--config", default=None, help="default: calibration/config.yaml")
    parser.add_argument("--from-captures", help="solve a saved captures.npz instead of using hardware")
    parser.add_argument("--exclude", help="with --from-captures: capture numbers to drop, e.g. 3,7")
    parser.add_argument("--save", action="store_true", help="with --from-captures: save as a new result")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.from_captures:
        captures, intrinsics, board = load_captures(Path(args.from_captures))
        excluded = parse_exclude(args.exclude)
        captures = [c for k, c in enumerate(captures, start=1) if k not in excluded]
        print(f"Loaded {len(captures)} captures (excluded: {excluded or 'none'})")
        result = solve(captures, intrinsics, board)
        report(result, captures, config)
        if args.save:
            save(result, captures, intrinsics, board, config, f"{args.from_captures} excluding {excluded}")
        return

    board = Board(config["board"], config["capture"]["minimum_corners"])
    run_session(
        config,
        title="Step 1: wrist camera2 hand-eye",
        camera_names=[CAMERA],
        board=board,
        autosave_name=CAMERA,
        minimum_captures=int(config["wrist"]["minimum_captures"]),
        solve=lambda caps, intr: solve(caps, intr, board),
        report=lambda result, caps: report(result, caps, config),
        save=lambda result, caps, intr: save(result, caps, intr, board, config, "interactive session"),
        hint=pose_hint(config),
    )


if __name__ == "__main__":
    main()
