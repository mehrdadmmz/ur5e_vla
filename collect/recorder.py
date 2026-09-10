"""
Pure Python data recorder for UR5e robot.

Records robot state (joints, TCP pose, gripper) and RealSense camera frames
in parallel. Per-episode output layout:

    <ep_dir>/
        state.csv               # per-frame robot state + image filenames
        camera{i}/*.png         # RGB frames (one PNG per frame, timestamp name)
        camera{i}_depth/*.png   # 16-bit depth (only if record_depth=True)
        intrinsics.json         # camera intrinsics (only if record_depth=True)

Depth PNGs are raw uint16 (RealSense "depth units"). Multiply by the per-camera
`depth_scale` in intrinsics.json to get meters. Depth is spatially aligned to
the color frame (rs.align(color)), so depth[v,u] corresponds to color[v,u].
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

    def __init__(self, camera_serials, resolution=(640, 480), fps=30, record_depth=False):
        self.pipelines = []
        self.aligners = []          # rs.align(color) per camera when depth is on
        self.intrinsics = []        # per-camera dict (see get_intrinsics)
        self.num_cameras = len(camera_serials)
        self.record_depth = record_depth
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
            if record_depth:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, camera_fps)
            profile = pipeline.start(config)
            self.pipelines.append(pipeline)

            if record_depth:
                # Align depth into color's frame so depth[v,u] corresponds to
                # color[v,u]. Point-cloud backprojection uses color intrinsics.
                self.aligners.append(rs.align(rs.stream.color))
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_profile.get_intrinsics()
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                self.intrinsics.append({
                    "serial": serial,
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.ppx,
                    "cy": intr.ppy,
                    "depth_scale": depth_scale,   # meters per depth unit
                    "model": str(intr.model),
                    "coeffs": list(intr.coeffs),
                })
            else:
                self.aligners.append(None)

        # Warmup: wait for streams to start and auto-exposure to stabilize
        print(f"Initialized {self.num_cameras} camera(s) at {camera_fps} fps"
              f"{' + depth' if record_depth else ''}, warming up...")
        time.sleep(0.5)  # Give cameras time to start streaming
        for _ in range(10):
            self.get_frames(timeout_ms=2000)
        print(f"Cameras ready")

    def get_frames(self, timeout_ms=1000):
        """Capture frames from all cameras.

        Returns:
            If record_depth is False: list of color arrays (or None on failure).
            If record_depth is True:  list of (color, depth_uint16) tuples;
              either element may be None on failure.
        """
        results = []
        for pipeline, aligner in zip(self.pipelines, self.aligners):
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if aligner is not None:
                    frameset = aligner.process(frameset)
                    color_frame = frameset.get_color_frame()
                    depth_frame = frameset.get_depth_frame()
                    color = np.asanyarray(color_frame.get_data()) if color_frame else None
                    depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                    results.append((color, depth))
                else:
                    color_frame = frameset.get_color_frame()
                    results.append(np.asanyarray(color_frame.get_data()) if color_frame else None)
            except Exception:
                results.append((None, None) if aligner is not None else None)
        return results

    def get_intrinsics(self):
        """Return list of intrinsics dicts (one per camera). Empty if depth off."""
        return list(self.intrinsics)

    def close(self):
        """Stop all camera pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()


class DataRecorder:
    """Records robot state and camera images in parallel threads."""

    def __init__(self, rtde_r, gripper, camera_serials, hz=20, camera_resolution=(640, 480),
                 record_depth=False, context=None):
        """
        Args:
            rtde_r: RTDE receive interface
            gripper: RobotiqGripper instance
            camera_serials: List of RealSense camera serial numbers
            hz: Recording frequency in Hz (also used for camera fps)
            camera_resolution: (width, height) tuple
            record_depth: If True, also capture aligned depth (uint16 PNGs in
                camera{i}_depth/, plus intrinsics.json for point-cloud recon).
            context: Optional mutable dict stamped into each state.csv row.
                Recognized keys: chunk_index, phase, active_target_id,
                active_destination_id. Missing keys write as "". Update this
                dict in the collect loop at every phase transition.
        """
        self.rtde_r = rtde_r
        self.gripper = gripper
        self.hz = hz
        self.num_cameras = len(camera_serials)
        self.record_depth = record_depth
        self.context = context
        self.recording = False
        self.record_thread = None

        # Setup cameras (use hz for camera fps)
        self.cameras = RealsenseCapture(camera_serials, resolution=camera_resolution,
                                         fps=hz, record_depth=record_depth)

        # Data storage
        self.output_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self.last_gripper_pos = 0
        self.last_timestamp = 0.0

    def start(self, output_dir):
        """Start recording to the specified directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create folder for each camera
        for i in range(self.num_cameras):
            os.makedirs(os.path.join(output_dir, f"camera{i+1}"), exist_ok=True)
            if self.record_depth:
                os.makedirs(os.path.join(output_dir, f"camera{i+1}_depth"), exist_ok=True)

        # Snapshot intrinsics once per episode (needed for point-cloud recon).
        if self.record_depth:
            import json
            intrinsics = self.cameras.get_intrinsics()
            with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
                json.dump({f"camera{i+1}": intr for i, intr in enumerate(intrinsics)},
                          f, indent=2)

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
        # Context columns (blank when self.context is None)
        if self.context is not None:
            header += ["chunk_index", "phase", "active_target_id", "active_destination_id"]

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

            # Save images for each camera. When record_depth is on, each
            # frames entry is (color, depth_uint16); write depth alongside
            # color with the same filename in camera{i}_depth/.
            camera_files = []
            images = results.get("images", [])
            for i in range(self.num_cameras):
                if i < len(images) and images[i] is not None:
                    if self.record_depth:
                        color, depth = images[i]
                    else:
                        color, depth = images[i], None
                    if color is not None:
                        cv2.imwrite(
                            os.path.join(self.output_dir, f"camera{i+1}", filename),
                            cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                        )
                    if self.record_depth and depth is not None:
                        # Save raw uint16 (depth units). Multiply by depth_scale
                        # (in intrinsics.json) to get meters.
                        cv2.imwrite(
                            os.path.join(self.output_dir, f"camera{i+1}_depth", filename),
                            depth
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
            if self.context is not None:
                ctx = self.context
                row += [
                    ctx.get("chunk_index", ""),
                    ctx.get("phase", ""),
                    ctx.get("active_target_id", ""),
                    ctx.get("active_destination_id", ""),
                ]
            self.csv_writer.writerow(row)

            self.last_timestamp = timestamp
            self.frame_idx += 1

            # Maintain recording rate
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _capture_robot_state(self, results):
        """Capture robot joint positions, TCP pose (as quaternion), and gripper position."""
        try:
            results["joints"] = list(self.rtde_r.getActualQ())

            # Get TCP pose [x, y, z, rx, ry, rz] and convert orientation to quaternion
            tcp_pose = self.rtde_r.getActualTCPPose()
            results["tcp_pos"] = list(tcp_pose[:3])

            # Convert axis-angle [rx, ry, rz] to quaternion [qx, qy, qz, qw]
            axis_angle = np.array(tcp_pose[3:6])
            quat = axis_angle_to_quaternion(axis_angle)
            results["tcp_quat"] = list(quat)
        except Exception as e:
            print(f"Warning: failed to capture robot state: {e}")

        try:
            self.last_gripper_pos = self.gripper.get_current_position()
        except Exception as e:
            print(f"Warning: failed to capture gripper position, using last value {self.last_gripper_pos}: {e}")
        results["gripper_pos"] = self.last_gripper_pos

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
            frames = self.cameras.get_frames()
            for i, item in enumerate(frames):
                if self.record_depth:
                    color, depth = item
                    if color is None:
                        errors.append(f"Camera {i+1}: No color frame received")
                    elif depth is None:
                        errors.append(f"Camera {i+1}: No depth frame received")
                    else:
                        print(f"  Camera {i+1}: OK (color {color.shape}, depth {depth.shape} {depth.dtype})")
                else:
                    if item is None:
                        errors.append(f"Camera {i+1}: No frame received")
                    else:
                        print(f"  Camera {i+1}: OK ({item.shape})")
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

    def mark(self):
        """Return the current frame boundary and latest written timestamp.

        Use to stamp phase-transition events into a sidecar file. `frame_idx`
        is the number of rows written so far (and therefore the next row's
        index); call immediately after the physical event has occurred.
        """
        return self.frame_idx, self.last_timestamp
