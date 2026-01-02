#!/usr/bin/env python3
"""
Automatic Hand-Eye Calibration Data Collection for UR5e.

Automatically moves the robot to different poses and collects calibration data.
All poses face toward the marker position for consistent detection.

Usage:
    python auto_collect_handeye.py
    python auto_collect_handeye.py --config handeye_config.yaml
    python auto_collect_handeye.py --poses 100
"""

import os
import sys
import time
import yaml
import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation

import rtde_control
import rtde_receive

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collect'))

from handeye_calibration import calibrate_hand_eye
from collect_handeye import RealsenseCamera, ArucoDetector, CharucoDetector, create_detector, get_robot_pose_matrix


# =============================================================================
# Pose Generation - Always facing marker
# =============================================================================

def look_at_rotation(eye_pos: np.ndarray, target_pos: np.ndarray) -> Rotation:
    """
    Compute rotation so that Z-axis points from eye to target (camera looks at target).

    Args:
        eye_pos: Position of camera/end-effector [x, y, z]
        target_pos: Position to look at [x, y, z]

    Returns:
        Rotation object
    """
    # Direction from eye to target (this will be the camera's Z-axis / optical axis)
    forward = target_pos - eye_pos
    forward = forward / np.linalg.norm(forward)

    # Choose an up vector (world Z-up, but handle edge case)
    world_up = np.array([0, 0, 1])

    # If looking straight down, use Y as up
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 1, 0])

    # Compute right vector
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (camera convention: Z forward, X right, Y down)
    # For gripper: we want Z pointing down toward marker
    R = np.column_stack([right, -up, forward])

    return Rotation.from_matrix(R)


def generate_calibration_poses(
    marker_position: np.ndarray,
    init_height: float = 0.50,
    num_poses: int = 50,
    position_range: float = 0.08,
    height_range: float = 0.05,
    min_height: float = 0.35,
) -> List[np.ndarray]:
    """
    Generate diverse poses around marker, all facing the marker.

    Args:
        marker_position: [x, y, z] of the ArUco marker in base frame
        init_height: Starting height above marker
        num_poses: Number of poses to generate
        position_range: XY position variation in meters
        height_range: Height variation in meters
        min_height: Minimum height above table

    Returns:
        List of poses [x, y, z, qx, qy, qz, qw]
    """
    poses = []

    for i in range(num_poses):
        t = i / max(num_poses - 1, 1)

        # Generate position in a spiral pattern around the marker
        angle = t * 2 * np.pi * 4  # 4 revolutions
        radius = position_range * (0.2 + 0.8 * abs(np.sin(t * np.pi * 3)))

        # XY offset from marker
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)

        # Height variation
        dz = height_range * np.sin(t * np.pi * 5)

        # Camera/EE position
        ee_pos = np.array([
            marker_position[0] + dx,
            marker_position[1] + dy,
            max(init_height + dz, min_height)
        ])

        # Compute rotation to look at marker
        rot = look_at_rotation(ee_pos, marker_position)
        quat = rot.as_quat()  # [qx, qy, qz, qw]

        pose = np.concatenate([ee_pos, quat])
        poses.append(pose)

    return poses


# =============================================================================
# Auto Collection
# =============================================================================

class AutoHandEyeCollector:
    """Automated hand-eye calibration data collection."""

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config

        # Robot connection
        robot_ip = config['robot']['ip']
        print(f"Connecting to robot at {robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Robot connected.")

        # Robot motion parameters
        self.speed = config['robot'].get('speed', 0.3)
        self.acceleration = config['robot'].get('acceleration', 0.3)

        # Camera
        camera_cfg = config['camera']
        self.camera = RealsenseCamera(
            serial=camera_cfg['serial'],
            resolution=tuple(camera_cfg['resolution']),
            fps=camera_cfg.get('fps', 30)
        )

        # Detector (ArUco or ChArUco based on config)
        self.detector = create_detector(config)

        # Marker position
        self.marker_position = np.array(config.get('marker_position', [0.0, 0.45, 0.0]))

        # Data storage
        self.robot_poses: List[np.ndarray] = []
        self.camera_poses: List[np.ndarray] = []

        # Output directory
        self.output_dir = config.get('output_dir', 'handeye_data')
        os.makedirs(self.output_dir, exist_ok=True)

        # Calibration result file (will be overwritten)
        self.calibration_file = os.path.join(self.output_dir, "calibration_latest.npy")
        self.poses_file = os.path.join(self.output_dir, "poses_latest.npz")

    def quaternion_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to axis-angle [rx, ry, rz]."""
        rot = Rotation.from_quat(quat)
        return rot.as_rotvec()

    def move_to_pose(self, pose: np.ndarray) -> bool:
        """
        Move robot to target pose.

        Args:
            pose: [x, y, z, qx, qy, qz, qw]

        Returns:
            True if successful
        """
        target_xyz = pose[:3]
        target_quat = pose[3:]  # xyzw

        # Convert to axis-angle for RTDE
        target_axis_angle = self.quaternion_to_axis_angle(target_quat)
        target_pose = np.concatenate([target_xyz, target_axis_angle])

        # Check IK solution exists
        current_joints = self.rtde_r.getActualQ()
        if not self.rtde_c.getInverseKinematicsHasSolution(target_pose, current_joints):
            return False

        # Move
        target_joints = self.rtde_c.getInverseKinematics(target_pose, current_joints)
        self.rtde_c.moveJ(target_joints, self.speed, self.acceleration)

        return True

    def wait_until_stopped(self, timeout: float = 5.0, threshold: float = 0.001):
        """Wait until robot has stopped moving."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            speed = np.linalg.norm(self.rtde_r.getActualTCPSpeed())
            if speed < threshold:
                return True
            time.sleep(0.05)
        return False

    def capture_pose_pair(self, num_samples: int = 20, delay: float = 0.05) -> Tuple[bool, str]:
        """
        Capture a pose pair at current position with averaging.

        Args:
            num_samples: Number of samples to average
            delay: Delay between samples

        Returns:
            (success, message)
        """
        robot_matrices = []
        camera_matrices = []

        for i in range(num_samples):
            # Get robot pose
            T_robot = get_robot_pose_matrix(self.rtde_r)

            # Get camera frame and detect marker
            frame = self.camera.get_frame()
            if frame is None:
                return False, "Camera frame failed"

            T_camera, _ = self.detector.detect(
                frame, self.camera.camera_matrix, self.camera.dist_coeffs
            )

            if T_camera is None:
                return False, "Marker not detected"

            robot_matrices.append(T_robot)
            camera_matrices.append(T_camera)

            time.sleep(delay)

        # Average poses (use mean for translation, last for rotation)
        T_robot_avg = robot_matrices[-1].copy()
        T_robot_avg[:3, 3] = np.mean([m[:3, 3] for m in robot_matrices], axis=0)

        T_camera_avg = camera_matrices[-1].copy()
        T_camera_avg[:3, 3] = np.mean([m[:3, 3] for m in camera_matrices], axis=0)

        self.robot_poses.append(T_robot_avg)
        self.camera_poses.append(T_camera_avg)

        return True, f"Captured pose {len(self.robot_poses)}"

    def check_marker_visible(self) -> bool:
        """Check if marker is currently visible."""
        frame = self.camera.get_frame()
        if frame is None:
            return False

        T_camera, _ = self.detector.detect(
            frame, self.camera.camera_matrix, self.camera.dist_coeffs
        )
        return T_camera is not None

    def show_preview(self, wait_ms: int = 1) -> int:
        """Show camera preview. Returns key pressed."""
        frame = self.camera.get_frame()
        if frame is not None:
            _, annotated = self.detector.detect(
                frame, self.camera.camera_matrix, self.camera.dist_coeffs
            )
            cv2.putText(annotated, f"Poses: {len(self.robot_poses)}",
                        (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Hand-Eye Calibration", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        return cv2.waitKey(wait_ms) & 0xFF

    def save_data(self):
        """Save collected data (overwrites latest)."""
        np.savez(
            self.poses_file,
            robot_poses=np.array(self.robot_poses),
            camera_poses=np.array(self.camera_poses),
            camera_matrix=self.camera.camera_matrix,
            dist_coeffs=self.camera.dist_coeffs
        )
        print(f"Saved {len(self.robot_poses)} poses to {self.poses_file}")

    def run_calibration(self, verbose: bool = True,
                        max_iterations: int = 2000,
                        num_restarts: int = 5) -> Optional[np.ndarray]:
        """Run calibration with collected data.

        Args:
            verbose: Print progress information
            max_iterations: Maximum iterations per optimization run
            num_restarts: Number of optimization restarts with perturbation
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

    def save_calibration(self, X: np.ndarray):
        """Save calibration result (overwrites latest)."""
        np.save(self.calibration_file, X)
        print(f"Saved calibration to {self.calibration_file}")

        # Also save readable text
        txt_file = self.calibration_file.replace('.npy', '.txt')
        with open(txt_file, 'w') as f:
            f.write("Hand-Eye Calibration Result\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of poses used: {len(self.robot_poses)}\n\n")
            f.write("4x4 Transformation Matrix (end-effector -> camera):\n")
            f.write(np.array2string(X, precision=8, suppress_small=True))
            f.write("\n\n")

            t = X[:3, 3]
            rot = Rotation.from_matrix(X[:3, :3])
            q = rot.as_quat()
            euler = rot.as_euler('xyz', degrees=True)

            f.write(f"Translation (x, y, z) meters: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")
            f.write(f"Rotation quaternion (x, y, z, w): [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n")
            f.write(f"Rotation euler (degrees, xyz): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]\n")

        print(f"Saved readable format to {txt_file}")

    def run_auto_collection(
        self,
        init_pose: np.ndarray,
        num_poses: int = 50,
        position_range: float = 0.15,
        height_range: float = 0.15,
        settle_time: float = 1.0,
        capture_samples: int = 20,
        save_interval: int = 10,
    ) -> bool:
        """
        Run automatic data collection.

        Args:
            init_pose: Starting pose [x, y, z, qx, qy, qz, qw]
            num_poses: Number of poses to collect
            position_range: XY position variation in meters
            height_range: Height variation in meters
            settle_time: Time to wait after moving (seconds)
            capture_samples: Number of samples to average per pose
            save_interval: Save and run calibration every N poses

        Returns:
            True if successful
        """
        print(f"\n{'='*50}")
        print("Automatic Hand-Eye Calibration Collection")
        print(f"{'='*50}")
        print(f"Marker position: {self.marker_position}")
        print(f"Target poses: {num_poses}")
        print(f"Position range: +/- {position_range*100:.1f} cm")
        print(f"Height range: +/- {height_range*100:.1f} cm")
        print(f"Settle time: {settle_time} seconds")
        print(f"Samples per pose: {capture_samples}")
        print(f"Save interval: every {save_interval} poses")
        print(f"{'='*50}\n")

        # Get initial height from init_pose
        init_height = init_pose[2]

        # Generate poses that all face the marker
        print("Generating calibration poses (all facing marker)...")
        poses = generate_calibration_poses(
            marker_position=self.marker_position,
            init_height=init_height,
            num_poses=num_poses,
            position_range=position_range,
            height_range=height_range,
        )

        # Add initial pose at beginning
        poses.insert(0, init_pose)

        cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)

        successful_poses = 0
        failed_poses = 0

        try:
            for i, pose in enumerate(poses):
                print(f"\n[{i+1}/{len(poses)}] Moving to pose...")

                # Show current view
                self.show_preview(1)

                # Move to pose
                if not self.move_to_pose(pose):
                    print(f"  IK failed, skipping")
                    failed_poses += 1
                    continue

                # Wait for robot to stop completely
                print(f"  Waiting for robot to stop...")
                self.wait_until_stopped(timeout=3.0)

                # Additional settle time for vibration to stop
                print(f"  Settling for {settle_time}s...")
                time.sleep(settle_time)

                # Show preview and check for abort
                key = self.show_preview(100)
                if key == ord('q'):
                    print("\nAborted by user")
                    break

                # Check marker visibility
                if not self.check_marker_visible():
                    print(f"  Marker not visible, skipping")
                    failed_poses += 1
                    continue

                # Capture pose pair with averaging
                print(f"  Capturing {capture_samples} samples...")
                success, msg = self.capture_pose_pair(num_samples=capture_samples, delay=0.05)
                if success:
                    print(f"  {msg}")
                    successful_poses += 1

                    # Save and calibrate at intervals
                    if successful_poses % save_interval == 0 and successful_poses >= 5:
                        print(f"\n--- Checkpoint at {successful_poses} poses ---")
                        self.save_data()

                        print("Running intermediate calibration...")
                        X = self.run_calibration(verbose=True)
                        if X is not None:
                            self.save_calibration(X)
                        print("--- Checkpoint complete ---\n")

                else:
                    print(f"  Failed: {msg}")
                    failed_poses += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"Collection complete!")
        print(f"  Successful: {successful_poses}")
        print(f"  Failed: {failed_poses}")
        print(f"{'='*50}")

        # Return to initial pose
        print("\nReturning to initial pose...")
        self.move_to_pose(init_pose)

        return successful_poses >= 3

    def close(self):
        """Clean up."""
        self.camera.close()
        self.rtde_c.stopScript()


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Automatic Hand-Eye Calibration")
    parser.add_argument('--config', default='handeye_config.yaml', help='Config file')
    parser.add_argument('--poses', type=int, default=50, help='Number of poses to collect')
    parser.add_argument('--position-range', type=float, default=0.08, help='XY position range (meters)')
    parser.add_argument('--height-range', type=float, default=0.05, help='Height range (meters)')
    parser.add_argument('--settle-time', type=float, default=1.0, help='Settle time after move (seconds)')
    parser.add_argument('--samples', type=int, default=20, help='Samples per pose for averaging')
    parser.add_argument('--save-interval', type=int, default=10, help='Save/calibrate every N poses')

    args = parser.parse_args()

    # Default initial pose: gripper facing down, above marker
    # [x, y, z, qx, qy, qz, qw]
    init_pose = np.array([0.00, 0.45, 0.50, 1.0, 0.0, 0.0, 0.0])

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)

    if not os.path.exists(config_path):
        print(f"Config file not found: {args.config}")
        return

    config = load_config(config_path)

    # Add robot motion parameters if not present
    if 'speed' not in config['robot']:
        config['robot']['speed'] = 0.3
    if 'acceleration' not in config['robot']:
        config['robot']['acceleration'] = 0.3

    # Initialize collector
    collector = AutoHandEyeCollector(config)

    try:
        # First move to init pose and verify marker is visible
        print("Moving to initial pose...")
        if not collector.move_to_pose(init_pose):
            print("ERROR: Cannot reach initial pose (IK failed)")
            return

        collector.wait_until_stopped()
        time.sleep(1.0)

        if not collector.check_marker_visible():
            print("ERROR: Marker not visible at initial pose!")
            print("Please adjust the initial pose or marker position.")

            cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
            print("Press 'q' to quit...")
            while True:
                key = collector.show_preview(100)
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
            return

        print("Marker detected at initial pose!")

        # Run automatic collection
        success = collector.run_auto_collection(
            init_pose=init_pose,
            num_poses=args.poses,
            position_range=args.position_range,
            height_range=args.height_range,
            settle_time=args.settle_time,
            capture_samples=args.samples,
            save_interval=args.save_interval,
        )

        if success and len(collector.robot_poses) >= 3:
            # Final save
            collector.save_data()

            # Final calibration
            print("\nRunning final calibration...")
            X = collector.run_calibration(verbose=True)

            if X is not None:
                collector.save_calibration(X)
                print("\n" + "="*50)
                print("CALIBRATION COMPLETE!")
                print(f"Result saved to: {collector.calibration_file}")
                print("="*50)
            else:
                print("\nCalibration failed. Data saved for manual inspection.")
        else:
            print("\nNot enough poses collected for calibration.")
            if len(collector.robot_poses) > 0:
                collector.save_data()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.close()


if __name__ == '__main__':
    main()
