#!/usr/bin/env python3
"""Steps 2/3: fixed camera calibration through the wrist camera -> T_base_cameraN.

Place the 7x5 board where BOTH the wrist camera2 and the fixed camera see it,
keep everything still, press 's'. Move the board (and optionally the robot) to
a new position/angle before every capture. This script never moves the robot.

    .venv/bin/python calibration/calibrate_fixed.py --camera camera3
    .venv/bin/python calibration/calibrate_fixed.py --camera camera1

By default the wrist result is the newest new run in calibration/results/camera2/,
or the accepted one in calibration/three_cam_calibration/ if there is none.

Re-solve saved captures without hardware:

    .venv/bin/python calibration/calibrate_fixed.py --camera camera3 \
        --from-captures calibration/results/camera3/result_1/captures.npz --exclude 4 --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import FixedCapture, solve_fixed  # noqa: E402
from hardware import (  # noqa: E402
    Board, Capture, Intrinsics, compare_to_previous, latest_result_file, load_captures,
    load_config, next_result_dir, parse_exclude, save_captures, timestamp,
    transform_summary, write_yaml,
)
from session import run_session  # noqa: E402

WRIST = "camera2"
FIXED_CAMERAS = ("camera1", "camera3")


def load_wrist(path_arg: str | None) -> tuple[np.ndarray, Path]:
    path = Path(path_arg) if path_arg else latest_result_file(WRIST, "T_ee_camera2.npy")
    if path is None or not path.is_file():
        raise SystemExit("No wrist calibration found. Run step 1 (calibrate_wrist.py) first, or pass --wrist.")
    print(f"Using wrist calibration: {path}")
    return np.load(path), path.resolve()


def solve(captures: list[Capture], intrinsics: Intrinsics, camera: str, T_ee_wrist: np.ndarray,
          board: Board, config: dict):
    K_w, d_w = intrinsics[WRIST]
    K_f, d_f = intrinsics[camera]
    return solve_fixed(
        T_ee_wrist,
        [FixedCapture(c.T_base_ee, c.observations[WRIST], c.observations[camera]) for c in captures],
        K_w, d_w, K_f, d_f, board.center,
        max_translation_m=float(config["fixed"]["max_translation_deviation_m"]),
        max_rotation_deg=float(config["fixed"]["max_rotation_deviation_deg"]),
    )


def report(result, camera: str, config: dict) -> str:
    m = result.metrics()
    t = result.T_base_camera[:3, 3]
    print("\n" + "=" * 70)
    print(f"T_base_{camera} ({camera} pose in the robot base frame):")
    print(np.array2string(result.T_base_camera, precision=6, suppress_small=True))
    print(f"position (m): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"\n  captures used           : {m['num_inliers']}/{m['num_captures']}")
    print(f"  reprojection RMS        : wrist {m['wrist_reprojection_rms_px']:.3f} px, "
          f"{camera} {m['fixed_reprojection_rms_px']:.3f} px")
    print(f"  board disagreement RMS  : {m['board_disagreement_rms_mm']:.2f} mm  (good < 3)")
    print(f"  board disagreement max  : {m['board_disagreement_max_mm']:.2f} mm")
    print(f"  approx 1-sigma          : {m['approx_sigma_translation_mm']:.2f} mm, "
          f"{m['approx_sigma_rotation_deg']:.3f} deg")
    change = compare_to_previous(config, camera, result.T_base_camera)
    if change:
        print(f"  vs previous calibration : {change['translation_change_mm']:.1f} mm, "
              f"{change['rotation_change_deg']:.2f} deg")

    print(f"\n  #    wrist px  {camera} px  board disagree mm")
    for k in range(len(result.inlier_mask)):
        flag = "" if result.inlier_mask[k] else "   <-- outlier (not used)"
        if result.inlier_mask[k] and result.board_disagreement_mm[k] > 8.0:
            flag = "   <-- suspicious, consider --exclude"
        print(f"  {k + 1:<4d} {result.per_capture_wrist_rms_px[k]:8.3f}  {result.per_capture_fixed_rms_px[k]:9.3f}"
              f"  {result.board_disagreement_mm[k]:16.2f}{flag}")
    print("=" * 70)
    return f"{camera} {m['fixed_reprojection_rms_px']:.2f}px, disagreement {m['board_disagreement_rms_mm']:.1f}mm"


def save(result, captures: list[Capture], intrinsics: Intrinsics, camera: str, T_ee_wrist: np.ndarray,
         wrist_path: Path, board: Board, config: dict, source: str) -> Path:
    out = next_result_dir(camera)
    np.save(out / f"T_base_{camera}.npy", result.T_base_camera)
    save_captures(out / "captures.npz", captures, intrinsics, board)
    K, dist = intrinsics[camera]
    write_yaml(out / "report.yaml", {
        "transform_name": f"T_base_{camera}",
        "convention": f"p_base = T_base_{camera} @ p_{camera} (colour optical frame)",
        **transform_summary(result.T_base_camera),
        "metrics": result.metrics(),
        "change_vs_previous": compare_to_previous(config, camera, result.T_base_camera),
        "inlier_mask": result.inlier_mask,
        "per_capture_board_disagreement_mm": result.board_disagreement_mm,
        "wrist_calibration": {"file": str(wrist_path), "T_ee_camera2": T_ee_wrist},
        "board": board.cfg,
        "camera": {"name": camera, "serial": config["cameras"][camera]["serial"], "K": K, "dist": dist},
        "source": source,
        "created_at": timestamp(),
    })
    print(f"\nSaved to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed camera calibration through the wrist camera")
    parser.add_argument("--camera", required=True, choices=FIXED_CAMERAS)
    parser.add_argument("--wrist", help="T_ee_camera2 .npy (default: newest new run, else three_cam_calibration/)")
    parser.add_argument("--config", default=None, help="default: calibration/config.yaml")
    parser.add_argument("--from-captures", help="solve a saved captures.npz instead of using hardware")
    parser.add_argument("--exclude", help="with --from-captures: capture numbers to drop, e.g. 3,7")
    parser.add_argument("--save", action="store_true", help="with --from-captures: save as a new result")
    args = parser.parse_args()
    config = load_config(args.config)
    camera = args.camera
    T_ee_wrist, wrist_path = load_wrist(args.wrist)

    if args.from_captures:
        captures, intrinsics, board = load_captures(Path(args.from_captures))
        excluded = parse_exclude(args.exclude)
        captures = [c for k, c in enumerate(captures, start=1) if k not in excluded]
        print(f"Loaded {len(captures)} captures (excluded: {excluded or 'none'})")
        result = solve(captures, intrinsics, camera, T_ee_wrist, board, config)
        report(result, camera, config)
        if args.save:
            save(result, captures, intrinsics, camera, T_ee_wrist, wrist_path, board, config,
                 f"{args.from_captures} excluding {excluded}")
        return

    board = Board(config["board"], config["capture"]["minimum_corners"])
    run_session(
        config,
        title=f"{camera} calibration (left: wrist camera2, right: {camera})",
        camera_names=[WRIST, camera],
        board=board,
        autosave_name=camera,
        minimum_captures=int(config["fixed"]["minimum_captures"]),
        solve=lambda caps, intr: solve(caps, intr, camera, T_ee_wrist, board, config),
        report=lambda result, caps: report(result, camera, config),
        save=lambda result, caps, intr: save(result, caps, intr, camera, T_ee_wrist, wrist_path,
                                             board, config, "interactive session"),
        hint=lambda caps, T_now: "move the board to a new spot/angle before each capture",
    )


if __name__ == "__main__":
    main()
