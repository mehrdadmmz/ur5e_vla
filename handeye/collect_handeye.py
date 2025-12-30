#!/usr/bin/env python3
"""
Hand-Eye Calibration Data Collection for UR5e with RealSense Camera.

This script collects pose pairs for hand-eye calibration:
- Robot end-effector poses (from RTDE)
- Camera-to-marker poses (from ArUco detection)

Usage:
    python collect_handeye.py                    # Interactive mode
    python collect_handeye.py --config my.yaml   # Custom config
    python collect_handeye.py --auto             # Auto-collect at predefined poses

Controls (interactive mode):
    s - Save current pose pair
    d - Delete last pose pair
    c - Run calibration with collected data
    v - Visualize current detection
    q - Quit and save
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import cv2
from cv2 import aruco
from datetime import datetime
from typing import List, Tuple, Optional

import pyrealsense2 as rs
import rtde_receive

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handeye_calibration import calibrate_hand_eye


# =============================================================================
# Camera Utilities
# =============================================================================

class RealsenseCamera:
    """RealSense camera interface for hand-eye calibration."""

    def __init__(self, serial: str, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        """
        Initialize RealSense camera.

        Args:
            serial: Camera serial number
            resolution: (width, height)
            fps: Frames per second
        """
        self.serial = serial
        self.width, self.height = resolution

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, fps)

        # Start pipeline and get intrinsics
        profile = self.pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()

        # Build camera matrix
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients
        self.dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

        # Warmup
        print(f"Camera {serial} initialized, warming up...")
        for _ in range(30):
            self.get_frame()
        print(f"Camera ready. Intrinsics:\n  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        print(f"  cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the camera."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"Camera error: {e}")
        return None

    def close(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()


# =============================================================================
# ArUco Detection
# =============================================================================

class ArucoDetector:
    """ArUco marker detector for camera pose estimation."""

    # Predefined ArUco dictionaries
    DICT_MAP = {
        "4x4_50": aruco.DICT_4X4_50,
        "4x4_100": aruco.DICT_4X4_100,
        "4x4_250": aruco.DICT_4X4_250,
        "5x5_50": aruco.DICT_5X5_50,
        "5x5_100": aruco.DICT_5X5_100,
        "5x5_250": aruco.DICT_5X5_250,
        "6x6_50": aruco.DICT_6X6_50,
        "6x6_100": aruco.DICT_6X6_100,
        "6x6_250": aruco.DICT_6X6_250,
        "original": aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self, marker_size: float, marker_id: int = 0,
                 dict_type: str = "4x4_50"):
        """
        Initialize ArUco detector.

        Args:
            marker_size: Physical size of marker in meters
            marker_id: ID of the marker to detect (use -1 for any)
            dict_type: ArUco dictionary type
        """
        self.marker_size = marker_size
        self.marker_id = marker_id

        if dict_type not in self.DICT_MAP:
            raise ValueError(f"Unknown dict_type: {dict_type}. Use one of {list(self.DICT_MAP.keys())}")

        self.aruco_dict = aruco.getPredefinedDictionary(self.DICT_MAP[dict_type])

        # Handle both old and new OpenCV ArUco API
        # OpenCV 4.7+ uses ArucoDetector class, older versions use module functions
        self._use_new_api = hasattr(aruco, 'ArucoDetector')

        if self._use_new_api:
            self.parameters = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            # Old API (OpenCV < 4.7)
            self.parameters = aruco.DetectorParameters_create()

        print(f"ArUco detector initialized: dict={dict_type}, size={marker_size}m, id={marker_id}")

    def detect(self, image: np.ndarray, camera_matrix: np.ndarray,
               dist_coeffs: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect ArUco marker and estimate pose.

        Args:
            image: RGB image
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            Tuple of (T_cam_marker as 4x4 matrix or None, annotated image)
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect markers - handle both old and new API
        if self._use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # Create annotated image
        annotated = image.copy()

        if ids is None or len(ids) == 0:
            cv2.putText(annotated, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return None, annotated

        # Find the target marker
        target_idx = None
        if self.marker_id == -1:
            # Use first detected marker
            target_idx = 0
        else:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.marker_id:
                    target_idx = i
                    break

        if target_idx is None:
            cv2.putText(annotated, f"Marker {self.marker_id} not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            aruco.drawDetectedMarkers(annotated, corners, ids)
            return None, annotated

        # Estimate pose for the target marker
        # Use solvePnP directly (works with all OpenCV versions)
        marker_corners = corners[target_idx].reshape(4, 2)

        # Define 3D marker corners (marker coordinate frame)
        half_size = self.marker_size / 2
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, marker_corners, camera_matrix, dist_coeffs
        )

        if not success:
            cv2.putText(annotated, "Pose estimation failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return None, annotated

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Build 4x4 transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = R
        T_cam_marker[:3, 3] = tvec

        # Draw detection
        aruco.drawDetectedMarkers(annotated, corners, ids)
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, self.marker_size * 0.5)

        # Add text info
        cv2.putText(annotated, f"ID: {ids[target_idx][0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Pos: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return T_cam_marker, annotated


# =============================================================================
# Robot Interface
# =============================================================================

def get_robot_pose_matrix(rtde_r) -> np.ndarray:
    """
    Get current robot TCP pose as 4x4 transformation matrix.

    Args:
        rtde_r: RTDE receive interface

    Returns:
        4x4 homogeneous transformation matrix (base -> end-effector)
    """
    # Get TCP pose [x, y, z, rx, ry, rz] (axis-angle format)
    tcp_pose = rtde_r.getActualTCPPose()

    # Extract position and axis-angle rotation
    position = np.array(tcp_pose[:3])
    axis_angle = np.array(tcp_pose[3:6])

    # Convert axis-angle to rotation matrix
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        R = np.eye(3)
    else:
        axis = axis_angle / angle
        # Rodrigues formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position

    return T


# =============================================================================
# Data Collection
# =============================================================================

class HandEyeDataCollector:
    """Collects and manages hand-eye calibration data."""

    def __init__(self, config: dict):
        """
        Initialize data collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize robot
        robot_ip = config['robot']['ip']
        print(f"Connecting to robot at {robot_ip}...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Robot connected.")

        # Initialize camera
        camera_cfg = config['camera']
        self.camera = RealsenseCamera(
            serial=camera_cfg['serial'],
            resolution=tuple(camera_cfg['resolution']),
            fps=camera_cfg.get('fps', 30)
        )

        # Initialize ArUco detector
        aruco_cfg = config['aruco']
        self.detector = ArucoDetector(
            marker_size=aruco_cfg['marker_size'],
            marker_id=aruco_cfg.get('marker_id', 0),
            dict_type=aruco_cfg.get('dict_type', '4x4_50')
        )

        # Data storage
        self.robot_poses: List[np.ndarray] = []
        self.camera_poses: List[np.ndarray] = []
        self.timestamps: List[float] = []

        # Output directory
        self.output_dir = config.get('output_dir', 'handeye_data')
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_pose_pair(self, num_samples: int = 5, delay: float = 0.1) -> Tuple[bool, str]:
        """
        Capture a pose pair (robot + camera).

        Averages multiple samples for stability.

        Args:
            num_samples: Number of samples to average
            delay: Delay between samples in seconds

        Returns:
            Tuple of (success, message)
        """
        robot_matrices = []
        camera_matrices = []

        for i in range(num_samples):
            # Get robot pose
            T_robot = get_robot_pose_matrix(self.rtde_r)

            # Get camera image and detect marker
            frame = self.camera.get_frame()
            if frame is None:
                return False, "Failed to capture camera frame"

            T_camera, _ = self.detector.detect(
                frame, self.camera.camera_matrix, self.camera.dist_coeffs
            )

            if T_camera is None:
                return False, "Marker not detected in frame"

            robot_matrices.append(T_robot)
            camera_matrices.append(T_camera)

            if i < num_samples - 1:
                time.sleep(delay)

        # Average the poses (simple averaging for position, use last for rotation)
        # For more accuracy, could use quaternion averaging
        T_robot_avg = robot_matrices[-1].copy()
        T_robot_avg[:3, 3] = np.mean([m[:3, 3] for m in robot_matrices], axis=0)

        T_camera_avg = camera_matrices[-1].copy()
        T_camera_avg[:3, 3] = np.mean([m[:3, 3] for m in camera_matrices], axis=0)

        # Store
        self.robot_poses.append(T_robot_avg)
        self.camera_poses.append(T_camera_avg)
        self.timestamps.append(time.time())

        return True, f"Pose {len(self.robot_poses)} captured"

    def delete_last_pose(self) -> str:
        """Delete the last captured pose pair."""
        if len(self.robot_poses) == 0:
            return "No poses to delete"

        self.robot_poses.pop()
        self.camera_poses.pop()
        self.timestamps.pop()

        return f"Deleted. {len(self.robot_poses)} poses remaining"

    def get_preview(self) -> np.ndarray:
        """Get current camera view with marker detection overlay."""
        frame = self.camera.get_frame()
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        _, annotated = self.detector.detect(
            frame, self.camera.camera_matrix, self.camera.dist_coeffs
        )

        # Add pose count
        cv2.putText(annotated, f"Poses: {len(self.robot_poses)}", (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated

    def run_calibration(self, verbose: bool = True,
                        max_iterations: int = 2000,
                        num_restarts: int = 5) -> Optional[np.ndarray]:
        """
        Run hand-eye calibration with collected data.

        Args:
            verbose: Print progress information
            max_iterations: Maximum iterations per optimization run
            num_restarts: Number of optimization restarts with perturbation

        Returns:
            4x4 calibration matrix or None if failed
        """
        if len(self.robot_poses) < 3:
            print(f"Need at least 3 poses, have {len(self.robot_poses)}")
            return None

        try:
            X = calibrate_hand_eye(
                self.robot_poses,
                self.camera_poses,
                verbose=verbose,
                max_iterations=max_iterations,
                num_restarts=num_restarts
            )
            return X
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None

    def save_data(self, filename: str = None):
        """Save collected pose pairs to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"poses_{timestamp}.npz")

        np.savez(
            filename,
            robot_poses=np.array(self.robot_poses),
            camera_poses=np.array(self.camera_poses),
            timestamps=np.array(self.timestamps),
            camera_matrix=self.camera.camera_matrix,
            dist_coeffs=self.camera.dist_coeffs
        )
        print(f"Saved {len(self.robot_poses)} pose pairs to {filename}")
        return filename

    def load_data(self, filename: str):
        """Load pose pairs from file."""
        data = np.load(filename)
        self.robot_poses = list(data['robot_poses'])
        self.camera_poses = list(data['camera_poses'])
        self.timestamps = list(data['timestamps'])
        print(f"Loaded {len(self.robot_poses)} pose pairs from {filename}")

    def save_calibration(self, X: np.ndarray, filename: str = None):
        """Save calibration result."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"calibration_{timestamp}.npy")

        np.save(filename, X)
        print(f"Saved calibration to {filename}")

        # Also save as readable text
        txt_filename = filename.replace('.npy', '.txt')
        with open(txt_filename, 'w') as f:
            f.write("Hand-Eye Calibration Result\n")
            f.write("=" * 40 + "\n\n")
            f.write("4x4 Transformation Matrix (end-effector -> camera):\n")
            f.write(np.array2string(X, precision=8, suppress_small=True))
            f.write("\n\n")

            # Extract translation and rotation
            from scipy.spatial.transform import Rotation
            t = X[:3, 3]
            rot = Rotation.from_matrix(X[:3, :3])
            q = rot.as_quat()  # xyzw
            euler = rot.as_euler('xyz', degrees=True)

            f.write(f"Translation (x, y, z): [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")
            f.write(f"Rotation quaternion (x, y, z, w): [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n")
            f.write(f"Rotation euler (deg, xyz): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]\n")

        print(f"Saved readable format to {txt_filename}")
        return filename

    def close(self):
        """Clean up resources."""
        self.camera.close()


# =============================================================================
# Interactive Collection Mode
# =============================================================================

def run_interactive(collector: HandEyeDataCollector):
    """Run interactive data collection with keyboard controls."""
    print("\n" + "=" * 50)
    print("Hand-Eye Calibration Data Collection")
    print("=" * 50)
    print("Controls:")
    print("  s - Save current pose pair")
    print("  d - Delete last pose pair")
    print("  c - Run calibration")
    print("  q - Quit and save")
    print("=" * 50 + "\n")

    cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get preview
            preview = collector.get_preview()

            # Convert RGB to BGR for OpenCV display
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand-Eye Calibration", preview_bgr)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                # Save pose
                print("Capturing pose (hold still)...")
                success, msg = collector.capture_pose_pair(num_samples=10, delay=0.05)
                if success:
                    print(f"  {msg}")
                else:
                    print(f"  Failed: {msg}")

            elif key == ord('d'):
                # Delete last
                msg = collector.delete_last_pose()
                print(f"  {msg}")

            elif key == ord('c'):
                # Run calibration
                if len(collector.robot_poses) < 3:
                    print(f"  Need at least 3 poses (have {len(collector.robot_poses)})")
                else:
                    print("\n" + "="*50)
                    print("Running extended calibration optimization...")
                    print("  - 2000 iterations per run")
                    print("  - 5 restarts with perturbation")
                    print("  - BFGS + L-BFGS-B refinement")
                    print("This may take a minute...")
                    print("="*50 + "\n")
                    X = collector.run_calibration(
                        verbose=True,
                        max_iterations=2000,
                        num_restarts=5
                    )
                    if X is not None:
                        print("\nCalibration successful!")
                        save = input("Save calibration? (y/n): ").strip().lower()
                        if save == 'y':
                            collector.save_calibration(X)

            elif key == ord('q'):
                # Quit
                break

    finally:
        cv2.destroyAllWindows()

    # Save data on exit
    if len(collector.robot_poses) > 0:
        save = input(f"\nSave {len(collector.robot_poses)} poses before exit? (y/n): ").strip().lower()
        if save == 'y':
            collector.save_data()


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hand-Eye Calibration Data Collection")
    parser.add_argument('--config', default='handeye_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing pose data from .npz file')
    parser.add_argument('--calibrate-only', action='store_true',
                        help='Only run calibration on loaded data (no robot/camera needed)')
    args = parser.parse_args()

    # Handle calibrate-only mode
    if args.calibrate_only:
        if args.load is None:
            print("Error: --calibrate-only requires --load <file.npz>")
            return

        print(f"Loading data from {args.load}")
        data = np.load(args.load)
        robot_poses = list(data['robot_poses'])
        camera_poses = list(data['camera_poses'])

        print(f"Loaded {len(robot_poses)} pose pairs")
        print("\nRunning calibration...")

        X = calibrate_hand_eye(robot_poses, camera_poses, verbose=True)

        if X is not None:
            output_file = args.load.replace('.npz', '_calibration.npy')
            np.save(output_file, X)
            print(f"\nSaved calibration to {output_file}")
        return

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try in handeye directory
        config_path = os.path.join(os.path.dirname(__file__), args.config)

    if not os.path.exists(config_path):
        print(f"Config file not found: {args.config}")
        print("Creating default config...")
        create_default_config(args.config)
        print(f"Please edit {args.config} and run again.")
        return

    config = load_config(config_path)

    # Initialize collector
    collector = HandEyeDataCollector(config)

    # Load existing data if specified
    if args.load:
        collector.load_data(args.load)

    try:
        run_interactive(collector)
    finally:
        collector.close()


def create_default_config(path: str):
    """Create a default configuration file."""
    config = {
        'robot': {
            'ip': '192.168.56.101',
        },
        'camera': {
            'serial': '337322072205',  # Your RealSense serial
            'resolution': [640, 480],
            'fps': 30,
        },
        'aruco': {
            'marker_size': 0.05,       # Marker size in meters (5cm)
            'marker_id': 0,            # ID of marker to detect (-1 for any)
            'dict_type': '4x4_50',     # ArUco dictionary type
        },
        'output_dir': 'handeye_data',
    }

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created default config at {path}")


if __name__ == '__main__':
    main()
