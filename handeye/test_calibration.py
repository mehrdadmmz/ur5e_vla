#!/usr/bin/env python3
"""
Test Hand-Eye Calibration by grabbing an ArUco-marked block.

This script:
1. Detects the ArUco marker on a block
2. Transforms marker pose to robot base frame using calibration
3. Moves robot to the block and grabs it

Usage:
    python test_calibration.py
    python test_calibration.py --calibration handeye_data/calibration_xxx.npy
"""

import os
import sys
import time
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import pyrealsense2 as rs
import rtde_control
import rtde_receive

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collect'))

from collect_handeye import RealsenseCamera, ArucoDetector, get_robot_pose_matrix

# Try to import gripper
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collect'))
    import robotiq_gripper
    HAS_GRIPPER = True
except ImportError:
    HAS_GRIPPER = False
    print("Warning: robotiq_gripper not available, gripper commands will be skipped")


class CalibrationTester:
    """Test hand-eye calibration by picking up a marked block."""

    def __init__(self, config: dict, calibration_file: str):
        """
        Initialize tester.

        Args:
            config: Configuration dictionary
            calibration_file: Path to calibration .npy file
        """
        self.config = config

        # Load calibration (end-effector -> camera transform)
        self.X = np.load(calibration_file)
        print(f"Loaded calibration from {calibration_file}")
        print(f"X (ee -> camera):\n{self.X}")

        # Robot connection
        robot_ip = config['robot']['ip']
        print(f"\nConnecting to robot at {robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        self.speed = config['robot'].get('speed', 0.2)
        self.acceleration = config['robot'].get('acceleration', 0.2)
        print("Robot connected.")

        # Gripper
        if HAS_GRIPPER:
            gripper_port = config.get('gripper', {}).get('port', 63352)
            print(f"Connecting to gripper on port {gripper_port}...")
            self.gripper = robotiq_gripper.RobotiqGripper()
            self.gripper.connect(robot_ip, gripper_port)
            self.gripper.activate()
            print("Gripper connected.")
        else:
            self.gripper = None

        # Camera
        camera_cfg = config['camera']
        self.camera = RealsenseCamera(
            serial=camera_cfg['serial'],
            resolution=tuple(camera_cfg['resolution']),
            fps=camera_cfg.get('fps', 30)
        )

        # ArUco detector
        aruco_cfg = config['aruco']
        self.detector = ArucoDetector(
            marker_size=aruco_cfg['marker_size'],
            marker_id=aruco_cfg.get('marker_id', 0),
            dict_type=aruco_cfg.get('dict_type', '5x5_50')
        )

        # Grasp parameters
        self.approach_height = 0.10  # 10cm above block
        self.grasp_height = 0.02     # 2cm above table (adjust based on block height)

    def get_marker_pose_in_base(self) -> tuple:
        """
        Detect marker and transform to robot base frame.

        Returns:
            (T_base_marker, annotated_image) or (None, image) if not detected
        """
        # Get camera frame
        frame = self.camera.get_frame()
        if frame is None:
            return None, np.zeros((480, 640, 3), dtype=np.uint8)

        # Detect marker (camera -> marker)
        T_cam_marker, annotated = self.detector.detect(
            frame, self.camera.camera_matrix, self.camera.dist_coeffs
        )

        if T_cam_marker is None:
            return None, annotated

        # Get current robot pose (base -> end-effector)
        T_base_ee = get_robot_pose_matrix(self.rtde_r)

        # Transform: base -> ee -> camera -> marker
        # T_base_marker = T_base_ee @ X @ T_cam_marker
        T_base_marker = T_base_ee @ self.X @ T_cam_marker

        # Add marker position to annotated image
        pos = T_base_marker[:3, 3]
        cv2.putText(annotated, f"Base pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return T_base_marker, annotated

    def quaternion_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to axis-angle."""
        rot = Rotation.from_quat(quat)
        return rot.as_rotvec()

    def move_to_pose(self, position: np.ndarray, orientation: np.ndarray = None,
                     speed: float = None) -> bool:
        """
        Move robot to target pose.

        Args:
            position: [x, y, z] in base frame
            orientation: [qx, qy, qz, qw] or None to keep current
            speed: Override speed

        Returns:
            True if successful
        """
        if speed is None:
            speed = self.speed

        # Get current orientation if not specified
        if orientation is None:
            current_pose = self.rtde_r.getActualTCPPose()
            axis_angle = np.array(current_pose[3:6])
        else:
            axis_angle = self.quaternion_to_axis_angle(orientation)

        target_pose = np.concatenate([position, axis_angle])

        # Check IK
        current_joints = self.rtde_r.getActualQ()
        if not self.rtde_c.getInverseKinematicsHasSolution(target_pose, current_joints):
            print(f"IK failed for position {position}")
            return False

        # Move
        target_joints = self.rtde_c.getInverseKinematics(target_pose, current_joints)
        self.rtde_c.moveJ(target_joints, speed, self.acceleration)
        return True

    def open_gripper(self):
        """Open the gripper."""
        if self.gripper:
            self.gripper.move(0, 100, 100)  # position, speed, force
            time.sleep(0.5)

    def close_gripper(self):
        """Close the gripper."""
        if self.gripper:
            self.gripper.move(255, 100, 100)  # position, speed, force
            time.sleep(0.5)

    def pick_block(self, T_base_marker: np.ndarray) -> bool:
        """
        Pick up the block at the marker position.

        Args:
            T_base_marker: 4x4 transform of marker in base frame

        Returns:
            True if successful
        """
        marker_pos = T_base_marker[:3, 3]
        print(f"\nMarker position in base frame: {marker_pos}")

        # Approach position (above the block)
        approach_pos = marker_pos.copy()
        approach_pos[2] += self.approach_height

        # Grasp position
        grasp_pos = marker_pos.copy()
        grasp_pos[2] = self.grasp_height  # Fixed height above table

        print(f"Approach position: {approach_pos}")
        print(f"Grasp position: {grasp_pos}")

        # Open gripper
        print("Opening gripper...")
        self.open_gripper()

        # Move to approach position
        print("Moving to approach position...")
        if not self.move_to_pose(approach_pos):
            return False
        time.sleep(0.5)

        # Move down to grasp
        print("Moving to grasp position...")
        if not self.move_to_pose(grasp_pos, speed=0.1):
            return False
        time.sleep(0.3)

        # Close gripper
        print("Closing gripper...")
        self.close_gripper()

        # Lift up
        print("Lifting...")
        if not self.move_to_pose(approach_pos):
            return False

        print("Pick complete!")
        return True

    def run_interactive(self):
        """Run interactive test mode."""
        print("\n" + "=" * 50)
        print("Hand-Eye Calibration Test")
        print("=" * 50)
        print("Controls:")
        print("  p - Pick up the detected block")
        print("  o - Open gripper")
        print("  c - Close gripper")
        print("  r - Release (open) and return to view position")
        print("  q - Quit")
        print("=" * 50 + "\n")

        cv2.namedWindow("Calibration Test", cv2.WINDOW_NORMAL)

        last_marker_pose = None

        try:
            while True:
                # Get marker pose
                T_base_marker, annotated = self.get_marker_pose_in_base()

                if T_base_marker is not None:
                    last_marker_pose = T_base_marker

                # Show image
                cv2.imshow("Calibration Test", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(30) & 0xFF

                if key == ord('p'):
                    # Pick block
                    if last_marker_pose is not None:
                        print("\nAttempting to pick block...")
                        self.pick_block(last_marker_pose)
                    else:
                        print("No marker detected!")

                elif key == ord('o'):
                    print("Opening gripper...")
                    self.open_gripper()

                elif key == ord('c'):
                    print("Closing gripper...")
                    self.close_gripper()

                elif key == ord('r'):
                    print("Releasing and returning...")
                    self.open_gripper()

                elif key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()

    def close(self):
        """Clean up."""
        self.camera.close()
        self.rtde_c.stopScript()


def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_latest_calibration(output_dir: str = "handeye_data") -> str:
    """Find the most recent calibration file."""
    import glob
    files = glob.glob(os.path.join(output_dir, "calibration_*.npy"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test hand-eye calibration")
    parser.add_argument('--config', default='handeye_config.yaml', help='Config file')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Calibration .npy file (default: latest)')
    parser.add_argument('--approach-height', type=float, default=0.10,
                        help='Height above block for approach (meters)')
    parser.add_argument('--grasp-height', type=float, default=0.02,
                        help='Height above table for grasping (meters)')

    args = parser.parse_args()

    # Find calibration file
    calibration_file = args.calibration
    if calibration_file is None:
        calibration_file = find_latest_calibration()
        if calibration_file is None:
            print("No calibration file found. Run calibration first.")
            return
        print(f"Using latest calibration: {calibration_file}")

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)

    config = load_config(config_path)

    # Add gripper config if not present
    if 'gripper' not in config:
        config['gripper'] = {'port': 63352}

    # Initialize tester
    tester = CalibrationTester(config, calibration_file)
    tester.approach_height = args.approach_height
    tester.grasp_height = args.grasp_height

    try:
        tester.run_interactive()
    finally:
        tester.close()


if __name__ == '__main__':
    main()
