"""Cameras, ChArUco detection, robot pose reading and stationary capture.

The robot is only read through ``rtde_receive``: nothing here can command
motion. Jog the robot from the pendant between captures.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
import yaml
from cv2 import aruco
from scipy.spatial.transform import Rotation

from geometry import (
    Observation,
    mean_transform,
    rotation_angle_deg,
    solve_pnp,
    tcp_pose_to_transform,
)

# drawFrameAxes warns every frame when the axis arrows leave the image; harmless.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

CALIBRATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CALIBRATION_DIR / "results"           # new runs
ACCEPTED_DIR = CALIBRATION_DIR / "three_cam_calibration"  # calibration in use


# =============================================================================
# Config and result folders
# =============================================================================

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else CALIBRATION_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_result_dir(camera_name: str) -> Path:
    """results/<camera>/result_1, result_2, ... (never overwrites)."""
    root = RESULTS_DIR / camera_name
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[7:]) for p in root.glob("result_*") if p.name[7:].isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        path = root / f"result_{number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def latest_result_file(camera_name: str, filename: str) -> Path | None:
    """Newest new run in results/<camera>/, else the accepted calibration."""
    root = RESULTS_DIR / camera_name
    candidates = [
        (int(p.parent.name[7:]), p)
        for p in root.glob(f"result_*/{filename}")
        if p.parent.name[7:].isdigit()
    ]
    if candidates:
        return max(candidates)[1]
    accepted = ACCEPTED_DIR / filename
    return accepted if accepted.is_file() else None


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(data), f, sort_keys=False)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def transform_summary(T: np.ndarray) -> dict:
    rot = Rotation.from_matrix(T[:3, :3])
    return {
        "matrix": T,
        "translation_m": T[:3, 3],
        "quaternion_xyzw": rot.as_quat(),
        "euler_xyz_deg": rot.as_euler("xyz", degrees=True),
    }


def compare_to_previous(config: dict, camera_name: str, T: np.ndarray) -> dict | None:
    """How far a new result is from the earlier one listed in config.previous_results."""
    previous = (config.get("previous_results") or {}).get(camera_name)
    if not previous:
        return None
    path = (CALIBRATION_DIR / previous).resolve()
    if not path.is_file():
        return None
    T_old = np.load(path)
    return {
        "previous_file": str(path),
        "translation_change_mm": float(np.linalg.norm(T[:3, 3] - T_old[:3, 3]) * 1000.0),
        "rotation_change_deg": rotation_angle_deg(T_old[:3, :3], T[:3, :3]),
    }


def parse_exclude(text: str | None) -> list[int]:
    """'3,7' -> [3, 7] (1-based capture numbers, as printed in the report)."""
    return [int(x) for x in text.split(",") if x.strip()] if text else []


# =============================================================================
# Cameras
# =============================================================================

class ColorCamera:
    """One RealSense RGB stream with its factory colour intrinsics."""

    def __init__(self, name: str, serial: str, resolution: Sequence[int], fps: int):
        self.name = name
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, int(resolution[0]), int(resolution[1]), rs.format.rgb8, int(fps))
        profile = self.pipeline.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        self.dist = np.asarray(intr.coeffs, dtype=np.float64)
        self.intrinsics = {
            "serial": serial, "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "distortion_model": str(intr.model), "coefficients": list(intr.coeffs),
        }

    def frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"No colour frame from {self.name} ({self.serial})")
        return np.asanyarray(color.get_data()).copy()

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def open_cameras(config: dict, names: Sequence[str]) -> dict[str, ColorCamera]:
    stream = config["stream"]
    cameras: dict[str, ColorCamera] = {}
    try:
        for name in names:
            serial = str(config["cameras"][name]["serial"])
            print(f"Opening {name} ({config['cameras'][name]['role']}): {serial}")
            cameras[name] = ColorCamera(name, serial, stream["resolution"], stream["fps"])
    except Exception:
        for cam in cameras.values():
            cam.stop()
        raise
    return cameras


# =============================================================================
# ChArUco board
# =============================================================================

@dataclass
class Detection:
    ids: np.ndarray | None        # (N,)
    corners: np.ndarray | None    # (N, 2)
    marker_corners: Any
    marker_ids: Any

    @property
    def count(self) -> int:
        return 0 if self.ids is None else len(self.ids)


class Board:
    def __init__(self, cfg: dict, minimum_corners: int):
        self.cfg = dict(cfg)
        self.squares = (int(cfg["squares_x"]), int(cfg["squares_y"]))
        self.square_length = float(cfg["square_length_m"])
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, f"DICT_{str(cfg['dictionary']).upper()}"))
        self.board = aruco.CharucoBoard(self.squares, self.square_length, float(cfg["marker_length_m"]), dictionary)
        # Finer thresholding and more pixels per marker cell: needed for small,
        # distant boards (camera3 ~60 cm away finds 22-24 corners instead of 13-18).
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 4
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.15
        self.detector = aruco.CharucoDetector(self.board, aruco.CharucoParameters(), params)
        self.corner_points = np.asarray(self.board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        self.center = self.corner_points.mean(axis=0)
        max_corners = (self.squares[0] - 1) * (self.squares[1] - 1)
        self.minimum_corners = min(int(minimum_corners), max_corners)

    def describe(self) -> str:
        c = self.cfg
        return (f"{c['squares_x']}x{c['squares_y']} ChArUco, DICT_{c['dictionary']}, "
                f"{c['square_length_m'] * 1000:.1f} mm squares")

    def detect(self, image_rgb: np.ndarray) -> Detection:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return Detection(None, None, marker_corners, marker_ids)
        return Detection(
            np.asarray(ids, dtype=np.int32).reshape(-1),
            np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            marker_corners, marker_ids,
        )

    def observation(self, ids: np.ndarray, corners: np.ndarray) -> Observation:
        return Observation(self.corner_points[ids], corners, ids)

    def annotate(self, image_rgb: np.ndarray, det: Detection, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, str, bool]:
        """Draw the detection; returns (BGR image, status text, ok)."""
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if det.marker_ids is not None and len(det.marker_ids) > 0:
            aruco.drawDetectedMarkers(img, det.marker_corners, det.marker_ids)
        if det.count < self.minimum_corners:
            return img, f"need {self.minimum_corners} corners, found {det.count}", False
        aruco.drawDetectedCornersCharuco(
            img, det.corners.astype(np.float32).reshape(-1, 1, 2), det.ids.reshape(-1, 1)
        )
        try:
            T, rms = solve_pnp(self.observation(det.ids, det.corners), K, dist)
        except (ValueError, cv2.error):
            return img, "pose failed", False
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        cv2.drawFrameAxes(img, K, dist, rvec, T[:3, 3], self.square_length * 2)
        dist_m = np.linalg.norm(T[:3, 3])
        return img, f"{det.count} corners | {rms:.2f}px | {dist_m * 100:.0f}cm", True


def average_detections(dets: Sequence[Detection], min_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Average each corner id over frames; keep ids seen in >= min_fraction of frames."""
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for det in dets:
        for i, c in zip(det.ids, det.corners):
            sums[int(i)] = sums.get(int(i), 0) + c
            counts[int(i)] = counts.get(int(i), 0) + 1
    keep = sorted(i for i, n in counts.items() if n >= min_fraction * len(dets))
    ids = np.array(keep, dtype=np.int32)
    corners = np.array([sums[i] / counts[i] for i in keep], dtype=np.float64).reshape(-1, 2)
    return ids, corners


# =============================================================================
# Robot (read-only)
# =============================================================================

class RobotReader:
    def __init__(self, ip: str):
        print(f"Connecting to robot (read-only) at {ip}...")
        self.receiver = rtde_receive.RTDEReceiveInterface(ip)

    def T_base_ee(self) -> np.ndarray:
        return tcp_pose_to_transform(self.receiver.getActualTCPPose())

    def speed(self) -> float:
        """Linear TCP speed (m/s) plus angular speed (rad/s), a combined moving check."""
        v = np.asarray(self.receiver.getActualTCPSpeed(), dtype=np.float64)
        return float(np.linalg.norm(v[:3]) + np.linalg.norm(v[3:]))

    def close(self) -> None:
        try:
            self.receiver.disconnect()
        except Exception:
            pass


# =============================================================================
# Stationary capture (all named cameras must see the board)
# =============================================================================

@dataclass
class Capture:
    T_base_ee: np.ndarray
    observations: dict[str, Observation]
    valid_frames: int


def capture_stationary(
    cameras: dict[str, ColorCamera],
    robot: RobotReader,
    board: Board,
    settings: dict,
) -> tuple[Capture | None, str]:
    """Average several frames at one stationary pose. Robot and board must not move."""
    if robot.speed() > float(settings["max_tcp_speed"]):
        return None, "robot is still moving; wait until it stops"

    # Drop frames that may have been queued while things were still moving.
    for _ in range(int(settings["flush_frames"])):
        for cam in cameras.values():
            cam.frame()

    requested = int(settings["samples_per_capture"])
    need = max(1, math.ceil(requested * float(settings["minimum_valid_fraction"])))
    max_motion_m = float(settings["max_robot_motion_m"])
    max_motion_deg = float(settings["max_robot_motion_deg"])

    poses: list[np.ndarray] = []
    per_camera: dict[str, list[Detection]] = {name: [] for name in cameras}
    rejected = {"robot_motion": 0, "board_not_seen": 0}
    for _ in range(requested):
        before = robot.T_base_ee()
        images = {name: cam.frame() for name, cam in cameras.items()}
        after = robot.T_base_ee()
        if (np.linalg.norm(after[:3, 3] - before[:3, 3]) > max_motion_m
                or rotation_angle_deg(before[:3, :3], after[:3, :3]) > max_motion_deg):
            rejected["robot_motion"] += 1
            continue
        dets = {name: board.detect(img) for name, img in images.items()}
        if any(d.count < board.minimum_corners for d in dets.values()):
            rejected["board_not_seen"] += 1
            continue
        poses.append(before)
        for name, d in dets.items():
            per_camera[name].append(d)
        time.sleep(float(settings["sample_delay_s"]))

    if len(poses) < need:
        return None, f"only {len(poses)}/{requested} good frames (need {need}); rejected {rejected}"

    observations: dict[str, Observation] = {}
    max_rms = float(settings["max_reprojection_error_px"])
    for name, cam in cameras.items():
        ids, corners = average_detections(per_camera[name])
        if len(ids) < board.minimum_corners:
            return None, f"{name}: board corners flickered; only {len(ids)} stable corners"
        obs = board.observation(ids, corners)
        try:
            _, rms = solve_pnp(obs, cam.K, cam.dist)
        except (ValueError, cv2.error) as error:
            return None, f"{name}: pose failed ({error})"
        if rms > max_rms:
            return None, f"{name}: reprojection {rms:.2f}px > {max_rms}px (blur or board not flat?)"
        observations[name] = obs

    return Capture(mean_transform(poses), observations, len(poses)), f"captured using {len(poses)}/{requested} frames"


Intrinsics = dict[str, tuple[np.ndarray, np.ndarray]]  # camera name -> (K, dist)


def camera_intrinsics(cameras: dict[str, ColorCamera]) -> Intrinsics:
    return {name: (cam.K, cam.dist) for name, cam in cameras.items()}


def save_captures(path: Path, captures: Sequence[Capture], intrinsics: Intrinsics, board: Board) -> None:
    data: dict[str, Any] = {
        "T_base_ee": np.stack([c.T_base_ee for c in captures]),
        "valid_frames": np.array([c.valid_frames for c in captures]),
        "board_config": json.dumps(board.cfg),
        "camera_names": np.array(list(intrinsics)),
    }
    for name, (K, dist) in intrinsics.items():
        data[f"{name}_K"] = K
        data[f"{name}_dist"] = dist
        data[f"{name}_capture_index"] = np.concatenate(
            [np.full(len(c.observations[name].ids), k) for k, c in enumerate(captures)])
        data[f"{name}_ids"] = np.concatenate([c.observations[name].ids for c in captures])
        data[f"{name}_image_points"] = np.concatenate([c.observations[name].image_points for c in captures])
    np.savez(path, **data)


def load_captures(path: Path) -> tuple[list[Capture], Intrinsics, Board]:
    """Returns (captures, {camera: (K, dist)}, board) from a saved captures.npz."""
    data = np.load(path, allow_pickle=False)
    board = Board(json.loads(str(data["board_config"])), minimum_corners=6)
    names = [str(n) for n in data["camera_names"]]
    intrinsics = {n: (data[f"{n}_K"], data[f"{n}_dist"]) for n in names}
    captures = []
    for k, T in enumerate(data["T_base_ee"]):
        obs = {}
        for n in names:
            mask = data[f"{n}_capture_index"] == k
            obs[n] = board.observation(data[f"{n}_ids"][mask], data[f"{n}_image_points"][mask])
        captures.append(Capture(T, obs, int(data["valid_frames"][k])))
    return captures, intrinsics, board


# =============================================================================
# Preview helpers
# =============================================================================

def put_text(img: np.ndarray, text: str, row: int, color=(255, 255, 255), scale: float = 0.6) -> None:
    """Row 0, 1, ... from the top; row -1, -2, ... from the bottom."""
    y = 28 + 26 * row if row >= 0 else img.shape[0] - 14 + 26 * (row + 1)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


GREEN, RED, YELLOW, CYAN = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 0)
