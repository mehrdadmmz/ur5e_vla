"""
ArUco Marker Monitor for Block World Planning.

Tracks ArUco markers on blocks and provides their poses in robot base frame.
Supports both base-mounted (fixed) and wrist-mounted (eye-in-hand) cameras.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ArucoMonitor:
    """
    Continuous ArUco marker detection and tracking.

    Provides block poses in robot base frame for planning.
    """

    # Block dimensions by marker ID (customize for your blocks)
    # Format: {marker_id: (width, length, height)}
    DEFAULT_BLOCK_DIMS = {
        # Cubes (class 2)
        0: (0.0375, 0.0375, 0.06),
        1: (0.045, 0.045, 0.03),
        2: (0.045, 0.045, 0.03),
        3: (0.0375, 0.0375, 0.06),
        5: (0.0375, 0.0375, 0.06),
        7: (0.0375, 0.0375, 0.06),
        # Planks (class 3)
        4: (0.15, 0.04, 0.02),
        6: (0.15, 0.05, 0.03),
    }

    # Block class by marker ID
    DEFAULT_BLOCK_CLASS = {
        0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 7: 2,  # Cubes
        4: 3, 6: 3,  # Planks
    }

    def __init__(
        self,
        camera_serial: str,
        camera_type: str = "base",  # "base" or "wrist"
        T_cam_base: Optional[np.ndarray] = None,
        X_ee_cam: Optional[np.ndarray] = None,
        rtde_r=None,
        marker_length: float = 0.032,
        frequency: float = 10.0,
        resolution: Tuple[int, int] = (640, 480),
        visual: bool = False,
        block_dims: Optional[Dict] = None,
        block_class: Optional[Dict] = None,
        position_offset: Optional[List[float]] = None,
    ):
        """
        Initialize ArUco monitor.

        Args:
            camera_serial: RealSense camera serial number
            camera_type: "base" (fixed camera) or "wrist" (eye-in-hand)
            T_cam_base: 4x4 transform from camera to base frame (for base camera)
            X_ee_cam: 4x4 transform from end-effector to camera (for wrist camera)
            rtde_r: RTDE receive interface (required for wrist camera)
            marker_length: ArUco marker size in meters
            frequency: Detection frequency in Hz
            resolution: Camera resolution (width, height)
            visual: Show detection visualization
            block_dims: Custom block dimensions {marker_id: (w, l, h)}
            block_class: Custom block classes {marker_id: class}
            position_offset: [x, y, z] offset to apply to detected positions in base frame
        """
        self.camera_serial = camera_serial
        self.camera_type = camera_type
        self.marker_length = marker_length
        self.frequency = frequency
        self.visual = visual
        self.position_offset = np.array(position_offset) if position_offset else np.zeros(3)

        # Block properties
        self.block_dims = block_dims or self.DEFAULT_BLOCK_DIMS
        self.block_class = block_class or self.DEFAULT_BLOCK_CLASS

        # Transforms
        if camera_type == "base":
            if T_cam_base is None:
                raise ValueError("T_cam_base required for base camera")
            self.T_cam_base = T_cam_base
        elif camera_type == "wrist":
            if X_ee_cam is None or rtde_r is None:
                raise ValueError("X_ee_cam and rtde_r required for wrist camera")
            self.X_ee_cam = X_ee_cam
            self.rtde_r = rtde_r
        else:
            raise ValueError(f"Unknown camera_type: {camera_type}")

        # Threading
        self.latest_poses: Dict[int, np.ndarray] = {}
        self.last_seen: Dict[int, float] = {}  # marker_id -> timestamp when last detected
        self.currently_visible: set = set()     # markers visible in most recent frame
        self.confidence: Dict[int, Dict] = {}   # marker_id -> {reproj_error, sharpness, confident}
        self.lock = threading.Lock()
        self._running = False
        self._thread = None

        # Confidence thresholds
        self.max_reproj_error = 2.0      # pixels - reject if higher
        self.min_sharpness = 30.0        # gradient magnitude - reject if lower
        self.confident_reproj = 1.0      # pixels - confident if below this
        self.confident_sharpness = 50.0  # gradient magnitude - confident if above this

        # Initialize
        self._init_aruco()
        self._init_camera(camera_serial, resolution)

    def _init_aruco(self):
        """Initialize ArUco detector."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

        # Handle both old and new OpenCV API
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_api = True

    def _init_camera(self, serial: str, resolution: Tuple[int, int]):
        """Initialize RealSense camera."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)

        profile = self.pipeline.start(config)

        # Get intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intr.coeffs)

        # Warmup
        for _ in range(30):
            self.pipeline.wait_for_frames()

        print(f"ArUco monitor initialized: camera={serial}, type={self.camera_type}")

    def _get_robot_pose(self) -> np.ndarray:
        """Get current robot TCP pose as 4x4 matrix (for wrist camera)."""
        tcp = self.rtde_r.getActualTCPPose()
        pos = np.array(tcp[:3])
        axis_angle = np.array(tcp[3:6])

        angle = np.linalg.norm(axis_angle)
        if angle < 1e-8:
            R = np.eye(3)
        else:
            axis = axis_angle / angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def _compute_corner_sharpness(self, gray: np.ndarray, corners: np.ndarray) -> float:
        """
        Compute sharpness metric for marker corners using gradient magnitude.

        Higher values indicate sharper/cleaner corner detection.
        """
        sharpness_values = []
        h, w = gray.shape

        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            # Sample a small region around corner (5x5)
            x1, x2 = max(0, x - 2), min(w, x + 3)
            y1, y2 = max(0, y - 2), min(h, y + 3)

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            region = gray[y1:y2, x1:x2].astype(np.float32)

            # Compute gradient magnitude using Sobel
            gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            sharpness_values.append(np.mean(magnitude))

        return np.mean(sharpness_values) if sharpness_values else 0.0

    def _detect_markers(self) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """
        Detect ArUco markers and return poses in camera frame with confidence.

        Returns:
            Tuple of (poses dict, confidence dict)
            - poses: {marker_id: 4x4 transform}
            - confidence: {marker_id: {reproj_error, sharpness, confident}}
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return {}, {}

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

        result = {}
        confidence_result = {}
        rvecs = []
        tvecs = []
        valid_ids = []

        if ids is not None:
            # Define 3D marker corners (marker coordinate system)
            half_size = self.marker_length / 2
            obj_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            # Estimate pose for each marker using solvePnP
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i].reshape(-1, 2)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corner, self.camera_matrix, self.dist_coeffs
                )
                if not success:
                    continue

                # Compute reprojection error
                projected, _ = cv2.projectPoints(
                    obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                projected = projected.reshape(-1, 2)
                reproj_error = np.mean(np.linalg.norm(corner - projected, axis=1))

                # Compute corner sharpness
                sharpness = self._compute_corner_sharpness(gray, corner)

                # Check if detection passes minimum quality
                if reproj_error > self.max_reproj_error:
                    continue  # Reject poor pose estimate
                if sharpness < self.min_sharpness:
                    continue  # Reject blurry detection

                # Determine if detection is confident (high quality)
                confident = (reproj_error < self.confident_reproj and
                            sharpness > self.confident_sharpness)

                # Store results
                rvecs.append(rvec)
                tvecs.append(tvec)
                valid_ids.append(marker_id)

                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                result[int(marker_id)] = T

                confidence_result[int(marker_id)] = {
                    'reproj_error': reproj_error,
                    'sharpness': sharpness,
                    'confident': confident
                }

        # Visualization
        if self.visual:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                for i in range(len(rvecs)):
                    cv2.drawFrameAxes(
                        img, self.camera_matrix, self.dist_coeffs,
                        rvecs[i], tvecs[i], self.marker_length * 0.5
                    )
                # Show confidence info
                for i, mid in enumerate(valid_ids):
                    conf = confidence_result[int(mid)]
                    color = (0, 255, 0) if conf['confident'] else (0, 165, 255)  # Green or orange
                    text = f"ID{mid} E:{conf['reproj_error']:.2f} S:{conf['sharpness']:.0f}"
                    cv2.putText(img, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("ArUco Monitor", img)
            cv2.waitKey(1)

        return result, confidence_result

    def _transform_to_base(self, T_cam_marker: np.ndarray) -> np.ndarray:
        """Transform marker pose from camera frame to base frame."""
        if self.camera_type == "base":
            # Fixed camera: T_base_marker = T_base_cam @ T_cam_marker
            T_base = self.T_cam_base @ T_cam_marker
        else:
            # Wrist camera: T_base_marker = T_base_ee @ X_ee_cam @ T_cam_marker
            T_base_ee = self._get_robot_pose()
            T_base = T_base_ee @ self.X_ee_cam @ T_cam_marker

        # Apply calibration offset
        T_base[:3, 3] += self.position_offset
        return T_base

    def _monitor_loop(self):
        """Background thread for continuous detection.

        If a marker is not detected, keeps the previous position.
        Only updates positions for markers that are currently visible.
        Tracks which markers are currently visible vs using stale data.
        """
        while self._running:
            poses_cam, conf_cam = self._detect_markers()
            current_time = time.time()

            with self.lock:
                # Track which markers are visible in this frame
                self.currently_visible = set(poses_cam.keys())

                # Update only the markers that were detected
                # Keep previous positions for markers not currently visible
                for marker_id, T_cam in poses_cam.items():
                    T_base = self._transform_to_base(T_cam)
                    self.latest_poses[marker_id] = T_base
                    self.last_seen[marker_id] = current_time
                    # Store confidence info
                    if marker_id in conf_cam:
                        self.confidence[marker_id] = conf_cam[marker_id]

            time.sleep(1.0 / self.frequency)

    def start(self):
        """Start background monitoring thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            print("ArUco monitor started")

    def stop(self):
        """Stop monitoring and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.pipeline.stop()
        if self.visual:
            cv2.destroyAllWindows()
        print("ArUco monitor stopped")

    def get_poses(self) -> Dict[int, np.ndarray]:
        """Get latest marker poses in base frame."""
        with self.lock:
            return dict(self.latest_poses)

    def get_visible_markers(self) -> set:
        """Get set of marker IDs currently visible."""
        with self.lock:
            return set(self.currently_visible)

    def get_stale_markers(self, max_age: float = 2.0) -> set:
        """Get set of marker IDs with stale (old) data.

        Args:
            max_age: Maximum age in seconds before data is considered stale

        Returns:
            Set of marker IDs with data older than max_age
        """
        current_time = time.time()
        with self.lock:
            stale = set()
            for marker_id, last_time in self.last_seen.items():
                if current_time - last_time > max_age:
                    stale.add(marker_id)
            return stale

    def clear_marker(self, marker_id: int):
        """Remove a marker from tracking (e.g., if picked up by robot)."""
        with self.lock:
            self.latest_poses.pop(marker_id, None)
            self.last_seen.pop(marker_id, None)
            self.currently_visible.discard(marker_id)
            self.confidence.pop(marker_id, None)

    def get_uncertain_markers(self) -> List[int]:
        """Get list of marker IDs with uncertain (low confidence) detections.

        Returns markers that are visible but not confident (high reproj error or low sharpness).
        """
        with self.lock:
            uncertain = []
            for marker_id in self.currently_visible:
                if marker_id in self.confidence:
                    if not self.confidence[marker_id].get('confident', True):
                        uncertain.append(marker_id)
            return uncertain

    def get_confidence(self, marker_id: int) -> Optional[Dict]:
        """Get confidence info for a marker."""
        with self.lock:
            return self.confidence.get(marker_id, None)

    def get_marker_position(self, marker_id: int) -> Optional[np.ndarray]:
        """Get position of a specific marker in base frame."""
        with self.lock:
            if marker_id in self.latest_poses:
                return self.latest_poses[marker_id][:3, 3].copy()
            return None

    def get_block_observations(
        self,
        table_id: int = 9,
        table_z: float = 0.0,
        robot_id: int = 8,
        gripper_pos: Optional[np.ndarray] = None,
        gripper_open: bool = True,
        print_status: bool = True,
    ) -> Tuple[List[List], List[int], List[int]]:
        """
        Get observations in format expected by predicate_util.

        Returns list of [id, class, x, y, z, w, l, h] for each block,
        plus table and robot entries.

        If a marker is not currently visible, uses the last known position.

        Args:
            table_id: ID for table object
            table_z: Z coordinate of table surface
            robot_id: ID for robot/gripper object
            gripper_pos: Current gripper position [x, y, z]
            gripper_open: Whether gripper is open (1) or closed (0)
            print_status: Print which markers are visible/stale

        Returns:
            Tuple of (observations list, stale marker IDs, uncertain marker IDs)
        """
        observations = []
        stale_ids = []
        uncertain_ids = []

        # Get current visibility and uncertainty
        visible = self.get_visible_markers()
        uncertain = self.get_uncertain_markers()

        # Get block poses
        poses = self.get_poses()
        for marker_id, T_base in poses.items():
            if marker_id not in self.block_dims:
                continue

            pos = T_base[:3, 3]
            w, l, h = self.block_dims[marker_id]
            cls = self.block_class.get(marker_id, 2)

            observations.append([
                marker_id,  # id
                float(cls),  # class
                pos[0],  # x
                pos[1],  # y
                pos[2],  # z (marker is on top of block, so this is ~= z_center + h/2)
                w, l, h
            ])

            # Track if using stale data
            if marker_id not in visible:
                stale_ids.append(marker_id)
            # Track if detection is uncertain
            elif marker_id in uncertain:
                uncertain_ids.append(marker_id)

        if print_status:
            if visible:
                print(f"  [ArUco] Currently visible: {sorted(visible)}")
            if stale_ids:
                print(f"  [ArUco] Using stale positions for markers: {stale_ids}")
            if uncertain_ids:
                # Print confidence details for uncertain markers
                for mid in uncertain_ids:
                    conf = self.get_confidence(mid)
                    if conf:
                        print(f"  [ArUco] Uncertain marker {mid}: reproj_err={conf['reproj_error']:.2f}px, sharpness={conf['sharpness']:.1f}")

        # Add table
        observations.append([
            table_id,
            0.0,  # class 0 = table
            0.0, 0.0, table_z,
            2.0, 2.0, 0.1  # Large table
        ])

        # Add robot
        if gripper_pos is not None:
            observations.append([
                robot_id,
                1.0,  # class 1 = robot
                gripper_pos[0], gripper_pos[1], gripper_pos[2],
                1 if gripper_open else 0
            ])

        return observations, stale_ids, uncertain_ids

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except:
            pass


# =============================================================================
# Factory functions for common setups
# =============================================================================

def create_base_camera_monitor(
    camera_serial: str,
    T_cam_base: np.ndarray,
    **kwargs
) -> ArucoMonitor:
    """Create monitor for base-mounted (fixed) camera."""
    return ArucoMonitor(
        camera_serial=camera_serial,
        camera_type="base",
        T_cam_base=T_cam_base,
        **kwargs
    )


def create_wrist_camera_monitor(
    camera_serial: str,
    X_ee_cam: np.ndarray,
    rtde_r,
    **kwargs
) -> ArucoMonitor:
    """Create monitor for wrist-mounted camera."""
    return ArucoMonitor(
        camera_serial=camera_serial,
        camera_type="wrist",
        X_ee_cam=X_ee_cam,
        rtde_r=rtde_r,
        **kwargs
    )
