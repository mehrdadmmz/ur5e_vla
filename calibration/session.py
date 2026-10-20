"""Shared keyboard loop for the interactive calibration scripts.

Keys: s capture | d delete last | c calculate | w calculate + save | q quit
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from hardware import (
    CYAN, GREEN, RED, RESULTS_DIR, YELLOW,
    Board, Capture, RobotReader, camera_intrinsics, capture_stationary, open_cameras,
    put_text, save_captures,
)


def run_session(
    config: dict,
    title: str,
    camera_names: list[str],
    board: Board,
    autosave_name: str,
    minimum_captures: int,
    solve: Callable[[list[Capture], dict], Any],
    report: Callable[[Any, list[Capture]], str],
    save: Callable[[Any, list[Capture], dict], None],
    hint: Callable[[list[Capture], np.ndarray], str] | None = None,
) -> None:
    """``solve(captures, intrinsics)`` -> result; ``report(result, captures)`` prints and
    returns a one-line summary; ``save(result, captures, intrinsics)`` writes it."""
    robot = RobotReader(config["robot"]["ip"])
    try:
        cameras = open_cameras(config, camera_names)
    except Exception:
        robot.close()
        raise
    intrinsics = camera_intrinsics(cameras)
    settings = config["capture"]
    autosave = RESULTS_DIR / autosave_name / "unsaved_session.npz"
    autosave.parent.mkdir(parents=True, exist_ok=True)

    captures: list[Capture] = []
    message, message_ok = "position robot/board, wait until still, press 's'", True
    unsaved = False
    quit_armed = False
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print(f"\n{title}\nBoard: {board.describe()}")
    print("Keys: s capture | d delete last | c calculate | w calculate + save | q quit")
    print("This script only reads the robot; jog it from the pendant.\n")

    try:
        while True:
            T_now = robot.T_base_ee()
            moving = robot.speed() > float(settings["max_tcp_speed"])
            tiles = []
            for name, cam in cameras.items():
                image = cam.frame()
                img, status, ok = board.annotate(image, board.detect(image), cam.K, cam.dist)
                put_text(img, f"{name} ({config['cameras'][name]['role']})", -1, CYAN, 0.55)
                put_text(img, status, 0, GREEN if ok else RED)
                tiles.append(img)
            view = np.hstack(tiles)
            put_text(view, "ROBOT MOVING" if moving else "robot stopped", 1, RED if moving else GREEN)
            extra = hint(captures, T_now) if hint else ""
            if extra:
                put_text(view, extra, 2, YELLOW)
            put_text(view, f"captures {len(captures)} (need {minimum_captures})", -3, CYAN)
            put_text(view, message, -2, GREEN if message_ok else RED, 0.55)
            cv2.imshow(title, view)
            key = cv2.waitKey(10) & 0xFF
            if key == 255:
                continue
            if key != ord("q"):
                quit_armed = False

            if key == ord("s"):
                capture, message = capture_stationary(cameras, robot, board, settings)
                message_ok = capture is not None
                if capture:
                    captures.append(capture)
                    message = f"#{len(captures)} {message}"
                    save_captures(autosave, captures, intrinsics, board)
                    unsaved = True
                print(("OK   " if message_ok else "FAIL ") + message)
            elif key == ord("d"):
                if captures:
                    captures.pop()
                    unsaved = True
                    if captures:
                        save_captures(autosave, captures, intrinsics, board)
                message, message_ok = f"deleted last; {len(captures)} left", True
                print(message)
            elif key in (ord("c"), ord("w")):
                if len(captures) < 3:
                    message, message_ok = "need at least 3 captures", False
                    continue
                try:
                    result = solve(captures, intrinsics)
                except ValueError as error:
                    message, message_ok = f"cannot solve: {error}", False
                    print(message)
                    continue
                message, message_ok = report(result, captures) + " (details in terminal)", True
                if key == ord("w"):
                    if len(captures) < minimum_captures:
                        message, message_ok = f"not saved: need {minimum_captures} captures", False
                        print(message)
                        continue
                    save(result, captures, intrinsics)
                    unsaved = False
                    message = "saved - see terminal for the folder"
            elif key == ord("q"):
                if unsaved and not quit_armed:
                    message, message_ok = "unsaved captures! press q again to quit", False
                    quit_armed = True
                    continue
                break
    finally:
        cv2.destroyAllWindows()
        for cam in cameras.values():
            cam.stop()
        robot.close()
        if unsaved and captures:
            print(f"\nUnsaved captures are kept in {autosave}")
            print("Solve them later with --from-captures.")
