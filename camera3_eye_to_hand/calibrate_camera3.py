#!/usr/bin/env python3
"""Interactive camera3 eye-to-hand calibration using wrist camera2 as reference.

This program opens two RealSense color streams and reads the UR5e TCP pose. It
does not import ``rtde_control`` and cannot command robot motion.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
import yaml
from cv2 import aruco
from scipy.spatial.transform import Rotation

from calibration_math import (
    CalibrationEstimate,
    camera3_in_base,
    estimate_fixed_transform,
    mean_transform,
    rotation_distance_degrees,
    validate_transform,
)


@dataclass(frozen=True)
class BoardDetection:
    T_camera_board: np.ndarray | None
    annotated_rgb: np.ndarray
    num_corners: int
    reprojection_error_px: float | None


@dataclass(frozen=True)
class CaptureRecord:
    T_base_camera3: np.ndarray
    T_base_ee: np.ndarray
    T_camera2_board: np.ndarray
    T_camera3_board: np.ndarray
    camera2_reprojection_error_px: float
    camera3_reprojection_error_px: float
    valid_frames: int


class RealSenseColorCamera:
    """One RealSense RGB stream with its factory color intrinsics."""

    def __init__(self, role: str, serial: str, resolution: tuple[int, int], fps: int):
        self.role = role
        self.serial = serial
        self.width, self.height = resolution
        self.pipeline = rs.pipeline()

        stream_config = rs.config()
        stream_config.enable_device(serial)
        stream_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            fps,
        )
        profile = self.pipeline.start(stream_config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()

        self.camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
        self.intrinsics = {
            "serial": serial,
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
            "distortion_model": str(intrinsics.model),
            "coefficients": [float(value) for value in intrinsics.coeffs],
        }
        self._stopped = False

    def frame(self, timeout_ms: int = 1500) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"No RGB frame received from {self.role} ({self.serial})")
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        if not self._stopped:
            self.pipeline.stop()
            self._stopped = True


class DualCameraRig:
    """Camera2 and camera3 capture with explicit role names."""

    def __init__(self, camera2_config: dict[str, Any], camera3_config: dict[str, Any]):
        self.camera2: RealSenseColorCamera | None = None
        self.camera3: RealSenseColorCamera | None = None
        try:
            self.camera2 = _open_camera("wrist camera2", camera2_config)
            self.camera3 = _open_camera("wall camera3", camera3_config)
            print("Warming up both cameras...")
            for _ in range(30):
                self.frames()
            print("Both cameras are ready.")
        except Exception:
            self.stop()
            raise

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self.camera2 is None or self.camera3 is None:
            raise RuntimeError("Both cameras must be initialized")
        # These calls are sequential, not hardware synchronized. The robot and
        # board must remain stationary throughout each saved capture.
        return self.camera2.frame(), self.camera3.frame()

    def stop(self) -> None:
        if self.camera3 is not None:
            self.camera3.stop()
        if self.camera2 is not None:
            self.camera2.stop()


class CharucoPoseDetector:
    """Detect a configured ChArUco board and estimate T_camera_board."""

    def __init__(self, config: dict[str, Any]):
        self.squares_x = int(config["squares_x"])
        self.squares_y = int(config["squares_y"])
        self.square_length = float(config["square_length_m"])
        self.marker_length = float(config["marker_length_m"])
        self.minimum_corners = int(config.get("minimum_corners", 4))
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("ChArUco board must have at least 2x2 squares")
        if not 0.0 < self.marker_length < self.square_length:
            raise ValueError(
                "ChArUco marker_length_m must be positive and smaller than "
                "square_length_m"
            )
        maximum_corners = (self.squares_x - 1) * (self.squares_y - 1)
        if not 6 <= self.minimum_corners <= maximum_corners:
            raise ValueError(
                f"minimum_corners must be between 6 and {maximum_corners} "
                "for the iterative pose solver"
            )

        dictionary_name = str(config["dictionary"]).upper()
        dictionary_attribute = f"DICT_{dictionary_name}"
        if not hasattr(aruco, dictionary_attribute):
            raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
        self.dictionary = aruco.getPredefinedDictionary(
            getattr(aruco, dictionary_attribute)
        )

        if hasattr(aruco, "CharucoBoard"):
            self.board = aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.dictionary,
            )
        else:
            self.board = aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length,
                self.marker_length,
                self.dictionary,
            )

        self.charuco_detector = (
            aruco.CharucoDetector(self.board)
            if hasattr(aruco, "CharucoDetector")
            else None
        )
        self.detector_parameters = (
            aruco.DetectorParameters()
            if hasattr(aruco, "DetectorParameters")
            else aruco.DetectorParameters_create()
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> BoardDetection:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        annotated = image_rgb.copy()

        if self.charuco_detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self.charuco_detector.detectBoard(gray)
            )
        else:
            marker_corners, marker_ids, _ = aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.detector_parameters,
            )
            if marker_ids is None:
                charuco_corners, charuco_ids = None, None
            else:
                _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    gray,
                    self.board,
                )

        if marker_ids is not None and marker_corners is not None:
            aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

        corner_count = 0 if charuco_corners is None else len(charuco_corners)
        if charuco_ids is None or corner_count < self.minimum_corners:
            _put_status(
                annotated,
                f"Need {self.minimum_corners} corners; found {corner_count}",
                good=False,
            )
            return BoardDetection(None, annotated, corner_count, None)

        object_points, image_points = self._matched_points(
            charuco_corners, charuco_ids
        )
        if len(object_points) < 6:
            _put_status(
                annotated,
                f"Pose needs 6 corners; found {len(object_points)}",
                good=False,
            )
            return BoardDetection(None, annotated, corner_count, None)

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as error:
            # A partial or degenerate board view should reject only this frame,
            # never terminate the full interactive calibration session.
            print(f"Warning: ChArUco pose estimation rejected a frame: {error}")
            _put_status(annotated, "Pose estimation rejected", good=False)
            return BoardDetection(None, annotated, corner_count, None)
        if not success:
            _put_status(annotated, "solvePnP failed", good=False)
            return BoardDetection(None, annotated, corner_count, None)

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec).reshape(3)

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        differences = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        reprojection_error = float(
            np.sqrt(np.mean(np.sum(differences**2, axis=1)))
        )

        # OpenCV 5 returns ChArUco arrays as (N, 2) and (N,), while the
        # drawing helper still requires the older (N, 1, 2)/(N, 1) layout.
        drawing_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(
            -1, 1, 2
        )
        drawing_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
        aruco.drawDetectedCornersCharuco(
            annotated, drawing_corners, drawing_ids
        )
        cv2.drawFrameAxes(
            annotated,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            self.square_length * 2.0,
        )
        _put_status(
            annotated,
            f"{corner_count} corners | reproj {reprojection_error:.2f}px",
            good=True,
        )
        return BoardDetection(
            transform,
            annotated,
            corner_count,
            reprojection_error,
        )

    def _matched_points(
        self, charuco_corners: np.ndarray, charuco_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.board, "matchImagePoints"):
            object_points, image_points = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )
        else:
            board_points = np.asarray(self.board.chessboardCorners)
            object_points = board_points[np.asarray(charuco_ids).reshape(-1)]
            image_points = charuco_corners
        return (
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        )


class RobotPoseReader:
    """Read-only UR RTDE connection."""

    def __init__(self, robot_ip: str):
        print(f"Connecting to UR5e state receiver at {robot_ip}...")
        self.receiver = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Robot state receiver connected (no motion-control interface opened).")

    def T_base_ee(self) -> np.ndarray:
        tcp_pose = self.receiver.getActualTCPPose()
        if tcp_pose is None or len(tcp_pose) != 6:
            raise RuntimeError("UR5e returned an invalid TCP pose")
        rotation, _ = cv2.Rodrigues(np.asarray(tcp_pose[3:6], dtype=np.float64))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tcp_pose[:3], dtype=np.float64)
        return transform

    def close(self) -> None:
        disconnect = getattr(self.receiver, "disconnect", None)
        if callable(disconnect):
            disconnect()


class CalibrationSession:
    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path
        self.config_directory = config_path.parent
        self.settings = config["calibration"]

        camera2_config = config["cameras"]["wrist_camera2"]
        camera3_config = config["cameras"]["wall_camera3"]
        if camera2_config["serial"] == camera3_config["serial"]:
            raise ValueError("Camera2 and camera3 serial numbers must be different")

        wrist_path = _resolve_from_config(
            self.config_directory, self.settings["wrist_extrinsic_file"]
        )
        if not wrist_path.is_file():
            raise FileNotFoundError(f"Wrist calibration not found: {wrist_path}")
        self.wrist_calibration_path = wrist_path
        self.T_ee_camera2 = validate_transform(
            np.load(wrist_path, allow_pickle=False), "T_ee_camera2"
        )

        self.detector = CharucoPoseDetector(config["charuco"])
        self.robot: RobotPoseReader | None = None
        self.cameras: DualCameraRig | None = None
        self.records: list[CaptureRecord] = []
        self.last_estimate: CalibrationEstimate | None = None

    def open(self) -> None:
        self.robot = RobotPoseReader(str(self.config["robot"]["ip"]))
        try:
            self.cameras = DualCameraRig(
                self.config["cameras"]["wrist_camera2"],
                self.config["cameras"]["wall_camera3"],
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.cameras is not None:
            self.cameras.stop()
            self.cameras = None
        if self.robot is not None:
            self.robot.close()
            self.robot = None

    def preview(self) -> tuple[BoardDetection, BoardDetection]:
        if self.cameras is None:
            raise RuntimeError("Cameras are not open")
        image2, image3 = self.cameras.frames()
        camera2 = self.cameras.camera2
        camera3 = self.cameras.camera3
        assert camera2 is not None and camera3 is not None
        detection2 = self.detector.detect(
            image2, camera2.camera_matrix, camera2.dist_coeffs
        )
        detection3 = self.detector.detect(
            image3, camera3.camera_matrix, camera3.dist_coeffs
        )
        return detection2, detection3

    def capture_stationary_pose(self) -> tuple[bool, str]:
        if self.cameras is None or self.robot is None:
            raise RuntimeError("Calibration hardware is not open")

        requested_samples = int(self.settings.get("samples_per_capture", 10))
        minimum_valid = max(
            1,
            math.ceil(
                requested_samples
                * float(self.settings.get("minimum_valid_fraction", 0.6))
            ),
        )
        delay_s = float(self.settings.get("sample_delay_s", 0.05))
        max_reprojection = float(
            self.settings.get("max_reprojection_error_px", 2.0)
        )
        max_robot_translation = float(
            self.settings.get("max_robot_motion_m", 0.0005)
        )
        max_robot_rotation = float(
            self.settings.get("max_robot_motion_deg", 0.1)
        )

        candidates: list[np.ndarray] = []
        robot_poses: list[np.ndarray] = []
        camera2_board_poses: list[np.ndarray] = []
        camera3_board_poses: list[np.ndarray] = []
        reprojection2: list[float] = []
        reprojection3: list[float] = []
        rejection_counts = {
            "detection": 0,
            "reprojection": 0,
            "motion": 0,
        }

        camera2 = self.cameras.camera2
        camera3 = self.cameras.camera3
        assert camera2 is not None and camera3 is not None

        for _ in range(requested_samples):
            T_before = self.robot.T_base_ee()
            image2, image3 = self.cameras.frames()
            T_after = self.robot.T_base_ee()

            robot_translation = float(
                np.linalg.norm(T_after[:3, 3] - T_before[:3, 3])
            )
            robot_rotation = rotation_distance_degrees(
                T_before[:3, :3], T_after[:3, :3]
            )
            if (
                robot_translation > max_robot_translation
                or robot_rotation > max_robot_rotation
            ):
                rejection_counts["motion"] += 1
                continue

            detection2 = self.detector.detect(
                image2, camera2.camera_matrix, camera2.dist_coeffs
            )
            detection3 = self.detector.detect(
                image3, camera3.camera_matrix, camera3.dist_coeffs
            )
            if (
                detection2.T_camera_board is None
                or detection3.T_camera_board is None
                or detection2.reprojection_error_px is None
                or detection3.reprojection_error_px is None
            ):
                rejection_counts["detection"] += 1
                continue
            if (
                detection2.reprojection_error_px > max_reprojection
                or detection3.reprojection_error_px > max_reprojection
            ):
                rejection_counts["reprojection"] += 1
                continue

            candidate = camera3_in_base(
                T_before,
                self.T_ee_camera2,
                detection2.T_camera_board,
                detection3.T_camera_board,
            )
            candidates.append(candidate)
            robot_poses.append(T_before)
            camera2_board_poses.append(detection2.T_camera_board)
            camera3_board_poses.append(detection3.T_camera_board)
            reprojection2.append(detection2.reprojection_error_px)
            reprojection3.append(detection3.reprojection_error_px)
            if delay_s > 0:
                time.sleep(delay_s)

        if len(candidates) < minimum_valid:
            return (
                False,
                f"Only {len(candidates)}/{requested_samples} valid frames; "
                f"need {minimum_valid}. Rejections: {rejection_counts}",
            )

        self.records.append(
            CaptureRecord(
                T_base_camera3=mean_transform(candidates),
                T_base_ee=mean_transform(robot_poses),
                T_camera2_board=mean_transform(camera2_board_poses),
                T_camera3_board=mean_transform(camera3_board_poses),
                camera2_reprojection_error_px=float(np.mean(reprojection2)),
                camera3_reprojection_error_px=float(np.mean(reprojection3)),
                valid_frames=len(candidates),
            )
        )
        self.last_estimate = None
        return (
            True,
            f"Saved board pose {len(self.records)} using "
            f"{len(candidates)}/{requested_samples} valid frames",
        )

    def delete_last(self) -> str:
        if not self.records:
            return "No saved board pose to delete"
        self.records.pop()
        self.last_estimate = None
        return f"Deleted last board pose; {len(self.records)} remain"

    def calculate(self) -> CalibrationEstimate:
        minimum_captures = int(self.settings.get("minimum_captures", 15))
        if len(self.records) < minimum_captures:
            raise ValueError(
                f"Need at least {minimum_captures} board poses; "
                f"currently have {len(self.records)}"
            )
        self.last_estimate = estimate_fixed_transform(
            [record.T_base_camera3 for record in self.records],
            max_translation_deviation_m=float(
                self.settings.get("max_translation_deviation_m", 0.03)
            ),
            max_rotation_deviation_deg=float(
                self.settings.get("max_rotation_deviation_deg", 5.0)
            ),
            min_inliers=minimum_captures,
        )
        return self.last_estimate

    def save(self) -> tuple[Path, Path, Path]:
        estimate = self.calculate()
        output_root = _resolve_from_config(
            self.config_directory, self.settings.get("output_dir", "results")
        )
        output_directory = _create_next_result_directory(output_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"camera3_to_base_{stamp}"
        matrix_path = output_directory / f"{stem}.npy"
        report_path = output_directory / f"{stem}.yaml"
        captures_path = output_directory / f"{stem}_captures.npz"

        np.save(matrix_path, estimate.transform)
        np.savez_compressed(
            captures_path,
            T_ee_camera2=self.T_ee_camera2,
            T_base_camera3_candidates=np.stack(
                [record.T_base_camera3 for record in self.records]
            ),
            T_base_ee=np.stack([record.T_base_ee for record in self.records]),
            T_camera2_board=np.stack(
                [record.T_camera2_board for record in self.records]
            ),
            T_camera3_board=np.stack(
                [record.T_camera3_board for record in self.records]
            ),
            camera2_reprojection_error_px=np.asarray(
                [record.camera2_reprojection_error_px for record in self.records]
            ),
            camera3_reprojection_error_px=np.asarray(
                [record.camera3_reprojection_error_px for record in self.records]
            ),
            valid_frames=np.asarray(
                [record.valid_frames for record in self.records], dtype=np.int32
            ),
            inlier_mask=estimate.inlier_mask,
        )

        camera2 = self.cameras.camera2 if self.cameras is not None else None
        camera3 = self.cameras.camera3 if self.cameras is not None else None
        quaternion = Rotation.from_matrix(estimate.transform[:3, :3]).as_quat()
        report = {
            "format_version": 1,
            "transform_name": "T_base_camera3",
            "transform_convention": (
                "Pose of camera3 expressed in robot base; p_base = "
                "T_base_camera3 @ p_camera3"
            ),
            "matrix": estimate.transform.tolist(),
            "translation_m": estimate.transform[:3, 3].tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "metrics": estimate.metrics(),
            "sources": {
                "config": str(self.config_path),
                "wrist_calibration": str(self.wrist_calibration_path),
                "T_ee_camera2": self.T_ee_camera2.tolist(),
                "wrist_camera2_serial": str(
                    self.config["cameras"]["wrist_camera2"]["serial"]
                ),
                "wall_camera3_serial": str(
                    self.config["cameras"]["wall_camera3"]["serial"]
                ),
            },
            "charuco": self.config["charuco"],
            "camera_intrinsics": {
                "wrist_camera2": camera2.intrinsics if camera2 else None,
                "wall_camera3": camera3.intrinsics if camera3 else None,
            },
            "created_at": datetime.now().astimezone().isoformat(),
        }
        with report_path.open("w", encoding="utf-8") as output:
            yaml.safe_dump(report, output, sort_keys=False)

        return matrix_path, report_path, captures_path


def _open_camera(role: str, config: dict[str, Any]) -> RealSenseColorCamera:
    resolution = config.get("resolution", [640, 480])
    if len(resolution) != 2:
        raise ValueError(f"{role} resolution must be [width, height]")
    serial = str(config["serial"])
    print(f"Opening {role}: {serial}")
    return RealSenseColorCamera(
        role=role,
        serial=serial,
        resolution=(int(resolution[0]), int(resolution[1])),
        fps=int(config.get("fps", 30)),
    )


def _put_status(image: np.ndarray, message: str, good: bool) -> None:
    color = (0, 255, 0) if good else (255, 0, 0)
    cv2.putText(
        image,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _resolve_from_config(config_directory: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_directory / path).resolve()


def _create_next_result_directory(output_root: Path) -> Path:
    """Create result_1, result_2, ... without overwriting an earlier run."""
    output_root.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for child in output_root.iterdir():
        if not child.is_dir() or not child.name.startswith("result_"):
            continue
        suffix = child.name.removeprefix("result_")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))

    next_number = max(existing_numbers, default=0) + 1
    while True:
        result_directory = output_root / f"result_{next_number}"
        try:
            result_directory.mkdir(exist_ok=False)
            return result_directory
        except FileExistsError:
            next_number += 1


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    for key in ("robot", "cameras", "charuco", "calibration"):
        if key not in config:
            raise ValueError(f"Configuration is missing '{key}'")
    for role in ("wrist_camera2", "wall_camera3"):
        if role not in config["cameras"]:
            raise ValueError(f"Configuration is missing cameras.{role}")
        if "serial" not in config["cameras"][role]:
            raise ValueError(f"Configuration is missing cameras.{role}.serial")
    return config


def _show_estimate(estimate: CalibrationEstimate) -> None:
    print("\nT_base_camera3:")
    print(np.array2string(estimate.transform, precision=8, suppress_small=True))
    print("\nConsistency metrics:")
    for key, value in estimate.metrics().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def _combined_preview(
    detection2: BoardDetection,
    detection3: BoardDetection,
    capture_count: int,
) -> np.ndarray:
    left = cv2.cvtColor(detection2.annotated_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(detection3.annotated_rgb, cv2.COLOR_RGB2BGR)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    cv2.putText(
        left,
        "camera2 / wrist reference",
        (10, left.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        "camera3 / fixed wall target",
        (10, right.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    combined = np.hstack([left, right])
    cv2.putText(
        combined,
        f"Saved board poses: {capture_count}",
        (combined.shape[1] // 2 - 125, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined


def run_interactive(session: CalibrationSession) -> None:
    session.open()
    window_name = "Camera3 eye-to-hand calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nThis collector cannot move the robot.")
    print("Keep the robot and board stationary whenever saving a pose.")
    print("Controls:")
    print("  s - capture the current stationary board pose")
    print("  d - delete the last saved pose")
    print("  c - calculate and print calibration")
    print("  w - calculate and save calibration + raw captures")
    print("  q - quit without automatically saving")

    try:
        while True:
            detection2, detection3 = session.preview()
            cv2.imshow(
                window_name,
                _combined_preview(detection2, detection3, len(session.records)),
            )
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s"):
                print("\nCapturing; do not move the robot or board...")
                success, message = session.capture_stationary_pose()
                print(("OK: " if success else "FAILED: ") + message)
            elif key == ord("d"):
                print(session.delete_last())
            elif key == ord("c"):
                try:
                    _show_estimate(session.calculate())
                except ValueError as error:
                    print(f"Cannot calculate: {error}")
            elif key == ord("w"):
                try:
                    _show_estimate(session.calculate())
                    paths = session.save()
                    print("\nSaved:")
                    for path in paths:
                        print(f"  {path}")
                except ValueError as error:
                    print(f"Cannot save: {error}")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate fixed camera3 in the UR5e base frame using calibrated "
            "wrist camera2 and a shared ChArUco board"
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration path (default: config.yaml beside this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_path = Path(args.config).expanduser()
    if not requested_path.is_file():
        requested_path = Path(__file__).resolve().parent / args.config
    config_path = requested_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {args.config}")

    config = _load_config(config_path)
    session = CalibrationSession(config, config_path)
    try:
        run_interactive(session)
    except KeyboardInterrupt:
        print("\nInterrupted; no calibration was automatically saved.")
        session.close()


if __name__ == "__main__":
    main()
