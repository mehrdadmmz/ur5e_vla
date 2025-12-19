"""
Pure Python data recorder for UR5e robot.
Records robot state and RealSense camera images in parallel.
"""

import time
import os
import csv
import threading
import numpy as np
import cv2
import pyrealsense2 as rs

from utils import axis_angle_to_quaternion


class RealsenseCapture:
    """Handles RealSense camera capture."""

    # RealSense D435 supported fps values
    SUPPORTED_FPS = [6, 15, 30, 60]

    def __init__(self, camera_serials, resolution=(640, 480), fps=30):
        self.pipelines = []
        self.num_cameras = len(camera_serials)
        width, height = resolution

        # Find closest supported fps (always >= requested)
        camera_fps = min([f for f in self.SUPPORTED_FPS if f >= fps], default=30)
        if camera_fps != fps:
            print(f"Note: Using {camera_fps} fps for cameras (requested {fps})")

        for serial in camera_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, camera_fps)
            pipeline.start(config)
            self.pipelines.append(pipeline)

        # Warmup: wait for streams to start and auto-exposure to stabilize
        print(f"Initialized {self.num_cameras} camera(s) at {camera_fps} fps, warming up...")
        time.sleep(0.5)  # Give cameras time to start streaming
        for _ in range(10):
            self.get_frames(timeout_ms=2000)
        print(f"Cameras ready")

    def get_frames(self, timeout_ms=1000):
        """Capture frames from all cameras."""
        images = []
        for pipeline in self.pipelines:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                color_frame = frameset.get_color_frame()
                if color_frame:
                    images.append(np.asanyarray(color_frame.get_data()))
                else:
                    images.append(None)
            except Exception:
                images.append(None)
        return images

    def close(self):
        """Stop all camera pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()


class DataRecorder:
    """Records robot state and camera images in parallel threads."""

    def __init__(self, rtde_r, gripper, camera_serials, hz=20, camera_resolution=(640, 480)):
        """
        Args:
            rtde_r: RTDE receive interface
            gripper: RobotiqGripper instance
            camera_serials: List of RealSense camera serial numbers
            hz: Recording frequency in Hz (also used for camera fps)
            camera_resolution: (width, height) tuple
        """
        self.rtde_r = rtde_r
        self.gripper = gripper
        self.hz = hz
        self.num_cameras = len(camera_serials)
        self.recording = False
        self.record_thread = None

        # Setup cameras (use hz for camera fps)
        self.cameras = RealsenseCapture(camera_serials, resolution=camera_resolution, fps=hz)

        # Data storage
        self.output_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

    def start(self, output_dir):
        """Start recording to the specified directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create folder for each camera
        for i in range(self.num_cameras):
            os.makedirs(os.path.join(output_dir, f"camera{i+1}"), exist_ok=True)

        # Build CSV header dynamically
        header = [
            "frame_idx", "timestamp",
            "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
            "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
            "gripper_pos"
        ]
        # Add camera columns
        for i in range(self.num_cameras):
            header.append(f"camera{i+1}_file")

        # Setup CSV
        self.csv_file = open(os.path.join(output_dir, "state.csv"), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(header)

        self.frame_idx = 0
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        print(f"Recording started: {output_dir} ({self.num_cameras} cameras)")

    def stop(self):
        """Stop recording."""
        if not self.recording:
            return
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        print(f"Recording stopped. {self.frame_idx} frames saved.")

    def _record_loop(self):
        """Main recording loop running in background thread."""
        interval = 1.0 / self.hz

        while self.recording:
            loop_start = time.time()

            # Capture timestamp at start of frame capture
            timestamp = time.time()

            # Capture all data in parallel using threads
            results = {}
            threads = [
                threading.Thread(target=self._capture_robot_state, args=(results,)),
                threading.Thread(target=self._capture_images, args=(results,)),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Use timestamp as filename for offline alignment
            filename = f"{timestamp:.6f}.png"

            # Save images for each camera
            camera_files = []
            images = results.get("images", [])
            for i in range(self.num_cameras):
                if i < len(images) and images[i] is not None:
                    cv2.imwrite(
                        os.path.join(self.output_dir, f"camera{i+1}", filename),
                        cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
                    )
                camera_files.append(filename)

            # Write CSV row
            joints = results.get("joints", [0]*6)
            tcp_pos = results.get("tcp_pos", [0]*3)
            tcp_quat = results.get("tcp_quat", [0, 0, 0, 1])
            gripper_pos = results.get("gripper_pos", 0)

            row = [
                self.frame_idx, timestamp,
                *joints,
                *tcp_pos,
                *tcp_quat,
                gripper_pos,
                *camera_files
            ]
            self.csv_writer.writerow(row)

            self.frame_idx += 1

            # Maintain recording rate
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _capture_robot_state(self, results):
        """Capture robot joint positions, TCP pose (as quaternion), and gripper position."""
        results["joints"] = list(self.rtde_r.getActualQ())

        # Get TCP pose [x, y, z, rx, ry, rz] and convert orientation to quaternion
        tcp_pose = self.rtde_r.getActualTCPPose()
        results["tcp_pos"] = list(tcp_pose[:3])

        # Convert axis-angle [rx, ry, rz] to quaternion [qx, qy, qz, qw]
        axis_angle = np.array(tcp_pose[3:6])
        quat = axis_angle_to_quaternion(axis_angle)
        results["tcp_quat"] = list(quat)

        results["gripper_pos"] = self.gripper.get_current_position()

    def _capture_images(self, results):
        """Capture images from all cameras."""
        # Use shorter timeout during recording to maintain rate
        results["images"] = self.cameras.get_frames(timeout_ms=100)

    def test(self):
        """Test all recording components. Raises exception if something fails."""
        print("Testing recording components...")
        errors = []

        # Test robot state
        try:
            joints = self.rtde_r.getActualQ()
            if joints is None or len(joints) != 6:
                errors.append("Robot: Failed to read joint positions")
            else:
                print(f"  Robot joints: OK")
        except Exception as e:
            errors.append(f"Robot: {e}")

        # Test gripper
        try:
            gripper_pos = self.gripper.get_current_position()
            if gripper_pos is None:
                errors.append("Gripper: Failed to read position")
            else:
                print(f"  Gripper: OK (pos={gripper_pos})")
        except Exception as e:
            errors.append(f"Gripper: {e}")

        # Test cameras
        try:
            images = self.cameras.get_frames()
            for i, img in enumerate(images):
                if img is None:
                    errors.append(f"Camera {i+1}: No frame received")
                else:
                    print(f"  Camera {i+1}: OK ({img.shape})")
        except Exception as e:
            errors.append(f"Cameras: {e}")

        if errors:
            error_msg = "Recording test failed:\n  " + "\n  ".join(errors)
            raise RuntimeError(error_msg)

        print("All recording components OK")
        return True

    def close(self):
        """Cleanup cameras."""
        self.cameras.close()
