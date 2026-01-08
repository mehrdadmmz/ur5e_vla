"""
Block Manipulation Primitives for UR5e.

Low-level robot actions for picking and placing blocks.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation


class BlockPrimitives:
    """
    Robot action primitives for block manipulation.

    Provides pick, place, align, and cover actions for PDDL plan execution.
    """

    def __init__(
        self,
        rtde_c,
        rtde_r,
        gripper,
        speed: float = 0.3,
        acceleration: float = 0.3,
        approach_height: float = 0.10,  # Height above target for approach
        grasp_depth: float = 0.02,      # How far below block top to grasp
        place_clearance: float = 0.005,  # Clearance when placing
        gripper_z_offset: float = 0.0,  # Distance from flange to gripper fingertips
        action_delay: float = 0.5,      # Delay between motion steps
        beside_gap: float = 0.06,       # Gap between blocks when placing beside
        linear_speed_factor: float = 0.5,  # Speed multiplier for linear moves
        linear_acceleration: float = 0.1,  # Acceleration for linear moves (m/s^2)
        gripper_open_pos: int = 0,
        gripper_close_pos: int = 255,
        gripper_speed: int = 100,
        gripper_force: int = 50,
        gripper_open_wait: float = 0.5,  # Wait after opening gripper
        gripper_close_wait: float = 1.5,  # Fallback wait if no blocking move
        gripper_open_margin: int = 10,   # Margin for "is open" check
        release_min_dist: float = 0.08,  # Min distance from other blocks for release
        table_bounds: Optional[Dict] = None,  # {x_min, x_max, y_min, y_max}
        contact_speed: float = 0.02,    # Speed for contact detection (m/s)
        contact_acceleration: float = 0.5,  # Acceleration for contact detection (m/s^2)
        contact_retract: float = 0.002,  # Retract distance after contact (meters)
        safe_height: float = 0.45,      # Minimum safe Z height for traversal moves
    ):
        """
        Initialize block primitives.

        Args:
            rtde_c: RTDE control interface
            rtde_r: RTDE receive interface
            gripper: Robotiq gripper instance
            speed: Joint speed (rad/s)
            acceleration: Joint acceleration (rad/s^2)
            approach_height: Height above block for approach moves
            grasp_depth: How deep to grasp into block from top
            place_clearance: Extra clearance when placing
            gripper_z_offset: Distance from robot flange to gripper fingertips
            action_delay: Delay between motion steps in seconds
            beside_gap: Gap between blocks when placing beside (for bridge bases)
            linear_speed_factor: Speed multiplier for linear moves
            gripper_open_pos: Gripper position for open (0-255)
            gripper_close_pos: Gripper position for close (0-255)
            gripper_speed: Gripper speed (0-255)
            gripper_force: Gripper force (0-255)
            gripper_open_wait: Wait time after opening gripper
            gripper_close_wait: Fallback wait time if no blocking move available
            gripper_open_margin: Position margin for "is open" check
            release_min_dist: Minimum distance from other blocks for release action
            table_bounds: Workspace bounds for release {x_min, x_max, y_min, y_max}
            contact_speed: Speed for contact detection moves (m/s)
            contact_acceleration: Acceleration for contact detection (m/s^2)
            contact_retract: Distance to retract after contact (meters)
        """
        self.rtde_c = rtde_c
        self.rtde_r = rtde_r
        self.gripper = gripper
        self.speed = speed
        self.acceleration = acceleration
        self.approach_height = approach_height
        self.grasp_depth = grasp_depth
        self.place_clearance = place_clearance
        self.gripper_z_offset = gripper_z_offset
        self.action_delay = action_delay
        self.beside_gap = beside_gap
        self.linear_speed_factor = linear_speed_factor
        self.linear_acceleration = linear_acceleration
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.gripper_open_wait = gripper_open_wait
        self.gripper_close_wait = gripper_close_wait
        self.gripper_open_margin = gripper_open_margin
        self.release_min_dist = release_min_dist
        self.table_bounds = table_bounds or {'x_min': -0.3, 'x_max': 0.3, 'y_min': 0.3, 'y_max': 0.7}
        self.contact_speed = contact_speed
        self.contact_acceleration = contact_acceleration
        self.contact_retract = contact_retract
        self.safe_height = safe_height

        # Default orientation: gripper pointing down
        # This is a 180 degree rotation around X (gripper Z pointing down)
        self.default_orientation = Rotation.from_rotvec([np.pi, 0, 0]).as_rotvec()

        # Grasp orientations per block ID (will be set from config)
        self.grasp_orientations: Dict[int, List[float]] = {}

        # Block dimensions per block ID [width, length, height] (will be set from config)
        self.block_dims: Dict[int, List[float]] = {}

        # Per-block gripper positions {block_id: [open_pos, close_pos]} (will be set from config)
        self.gripper_positions: Dict[int, List[int]] = {}

        # Joint limits per joint index (will be set from config)
        self.joint_limits: Optional[Dict[int, List[float]]] = None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _pose_to_list(self, position: np.ndarray, orientation: Optional[np.ndarray] = None) -> List[float]:
        """Convert position and orientation to RTDE pose format [x,y,z,rx,ry,rz]."""
        if orientation is None:
            orientation = self.default_orientation
        return list(position) + list(orientation)

    def _move_to_pose(self, pose: List[float], speed: Optional[float] = None) -> bool:
        """Move robot to target pose using moveJ."""
        if speed is None:
            speed = self.speed

        current_joints = self.rtde_r.getActualQ()

        # Check IK
        if not self.rtde_c.getInverseKinematicsHasSolution(pose, current_joints):
            print(f"IK failed for pose: {pose[:3]}")
            return False

        # Get target joints
        target_joints = self.rtde_c.getInverseKinematics(pose, current_joints)

        # Normalize joints to fit within limits (IK may return equivalent angles outside range)
        target_joints = self._normalize_joints_to_limits(target_joints)

        # Check joint limits
        if not self._check_joint_limits(target_joints):
            print(f"Target pose violates joint limits, skipping move")
            return False

        # Move
        self.rtde_c.moveJ(target_joints, speed, self.acceleration)
        return True

    def _move_linear(self, pose: List[float], speed: Optional[float] = None) -> bool:
        """Move robot linearly to target pose.

        Uses linear_acceleration (m/s²) instead of joint acceleration.
        """
        if speed is None:
            speed = self.speed * self.linear_speed_factor

        self.rtde_c.moveL(pose, speed, self.linear_acceleration)
        return True

    def _move_to_home(self) -> bool:
        """Move robot to safe height at current XY position."""
        current_pose = self.rtde_r.getActualTCPPose()
        home_pose = current_pose.copy()
        home_pose[2] = self.safe_height
        print(f"Moving to safe height: z={self.safe_height:.3f}")
        return self._move_to_pose(home_pose)

    def _ensure_safe_height(self) -> bool:
        """Ensure robot is at safe height before traversal moves.

        If current Z is below safe_height, moves straight up first.
        This prevents the robot from taking low paths through the workspace.
        """
        current_pose = self.rtde_r.getActualTCPPose()
        current_z = current_pose[2]

        if current_z < self.safe_height:
            # Move straight up to safe height first
            safe_pose = current_pose.copy()
            safe_pose[2] = self.safe_height
            print(f"  Lifting to safe height: {current_z:.3f} -> {self.safe_height:.3f}")
            return self._move_linear(safe_pose)
        return True

    def _move_until_contact(
        self,
        direction: List[float] = None,
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
        retract_on_contact: Optional[float] = None
    ) -> bool:
        """
        Move robot until contact is detected.

        Uses force feedback to detect when the gripper/block makes contact
        with a surface. This is more robust than calculated Z positions.

        Args:
            direction: 6D direction vector [x, y, z, rx, ry, rz].
                      Default is [0, 0, -1, 0, 0, 0] (move down).
            speed: Linear speed in m/s (uses self.contact_speed if None)
            acceleration: Acceleration in m/s² (uses self.contact_acceleration if None)
            retract_on_contact: Distance to retract after contact (meters).
                               Uses self.contact_retract if None.
                               Small retract prevents excessive force on block.

        Returns:
            True if contact was detected, False if movement failed
        """
        if direction is None:
            direction = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]  # Down
        if speed is None:
            speed = self.contact_speed
        if acceleration is None:
            acceleration = self.contact_acceleration
        if retract_on_contact is None:
            retract_on_contact = self.contact_retract

        # Tool velocity in the direction of movement
        xd = [d * speed for d in direction]

        try:
            # moveUntilContact returns True if contact detected
            contact = self.rtde_c.moveUntilContact(xd, direction, acceleration)

            if contact and retract_on_contact > 0:
                # Retract slightly to reduce force on the placed block
                current_pose = self.rtde_r.getActualTCPPose()
                retract_pose = current_pose.copy()
                # Retract in opposite direction
                for i in range(3):
                    retract_pose[i] -= direction[i] * retract_on_contact
                self.rtde_c.moveL(retract_pose, speed, acceleration)

            return contact

        except Exception as e:
            print(f"moveUntilContact failed: {e}")
            return False

    def _open_gripper(self, block_id: Optional[int] = None):
        """Open the gripper, using per-block position if configured."""
        if self.gripper:
            # Use per-block open position if available, otherwise global default
            open_pos = self.gripper_open_pos
            if block_id is not None and block_id in self.gripper_positions:
                open_pos = self.gripper_positions[block_id][0]
            self.gripper.move(open_pos, self.gripper_speed, self.gripper_force)
            time.sleep(self.gripper_open_wait)

    def _close_gripper(self, block_id: Optional[int] = None):
        """Close the gripper and wait until it finishes moving, using per-block position if configured."""
        if self.gripper:
            # Use per-block close position if available, otherwise global default
            close_pos = self.gripper_close_pos
            if block_id is not None and block_id in self.gripper_positions:
                close_pos = self.gripper_positions[block_id][1]
            # Use blocking move_and_wait_for_pos instead of non-blocking move
            if hasattr(self.gripper, 'move_and_wait_for_pos'):
                pos, status = self.gripper.move_and_wait_for_pos(
                    close_pos, self.gripper_speed, self.gripper_force
                )
                print(f"Gripper closed: pos={pos}, status={status}")
            else:
                # Fallback to non-blocking with longer wait
                self.gripper.move(close_pos, self.gripper_speed, self.gripper_force)
                time.sleep(self.gripper_close_wait)
            time.sleep(0.2)  # Extra settling time

    def _get_block_from_obs(self, block_id: int, observations: List[List]) -> Optional[List]:
        """Get block data from observations by ID."""
        for obs in observations:
            if obs[0] == block_id:
                return obs
        return None

    def get_gripper_position(self) -> np.ndarray:
        """Get current gripper TCP position."""
        tcp = self.rtde_r.getActualTCPPose()
        return np.array(tcp[:3])

    def is_gripper_open(self) -> bool:
        """Check if gripper is open (at or below configured open position)."""
        if self.gripper:
            pos = self.gripper.get_current_position()
            # Consider open if at or below the configured open position
            return pos <= self.gripper_open_pos + self.gripper_open_margin
        return True

    def open_gripper(self, block_id: Optional[int] = None):
        """Open the gripper (public interface)."""
        self._open_gripper(block_id)

    def close_gripper(self, block_id: Optional[int] = None):
        """Close the gripper (public interface)."""
        self._close_gripper(block_id)

    def get_default_orientation(self) -> np.ndarray:
        """Get the default gripper-down orientation as axis-angle."""
        return self.default_orientation.copy()

    def set_grasp_orientations(self, grasp_orientations: Dict[int, List[float]]):
        """Set grasp orientations for blocks.

        Args:
            grasp_orientations: Dict mapping block_id to list of valid Z-rotation angles
                               (in radians, relative to block frame)
        """
        self.grasp_orientations = grasp_orientations

    def set_block_dims(self, block_dims: Dict[int, List[float]]):
        """Set block dimensions for blocks.

        Args:
            block_dims: Dict mapping block_id to [width, length, height] in meters
        """
        self.block_dims = block_dims

    def set_gripper_positions(self, gripper_positions: Dict[int, List[int]]):
        """Set per-block gripper positions.

        Args:
            gripper_positions: Dict mapping block_id to [open_pos, close_pos] (0-255)
        """
        self.gripper_positions = gripper_positions

    def _get_long_axis_offset(self, block_id: int) -> float:
        """Get the angular offset of the block's long axis from its X-axis.

        For blocks where Y > X (long axis is Y), returns π/2.
        For blocks where X >= Y (long axis is X), returns 0.

        This is used to compute parallel placement for blocks with different aspect ratios.

        Args:
            block_id: Block marker ID

        Returns:
            0.0 if long axis is X, π/2 if long axis is Y
        """
        dims = self.block_dims.get(block_id, [0.04, 0.04, 0.04])
        if dims[1] > dims[0]:  # Y > X, long axis is Y
            return np.pi / 2
        return 0.0

    def _find_reference_plank_below(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        observations: List[List],
        xy_tolerance: float = 0.10
    ) -> Optional[List]:
        """Find a plank below the target position to use as orientation reference.

        When placing a plank on a cube, we want to align with other planks in the
        stack (e.g., for Chinese characters where planks sandwich cubes).

        Args:
            target_x, target_y, target_z: Position of placement target
            observations: Current world state
            xy_tolerance: Max XY distance to consider "in the same stack"

        Returns:
            Observation of reference plank, or None if not found
        """
        best_plank = None
        best_z = -float('inf')

        for obs in observations:
            if len(obs) < 9:
                continue
            block_class = obs[1]
            if block_class != 3:  # Not a plank
                continue

            bx, by, bz = obs[2], obs[3], obs[4]

            # Check if below target
            if bz >= target_z:
                continue

            # Check XY proximity (in same stack)
            xy_dist = np.sqrt((bx - target_x)**2 + (by - target_y)**2)
            if xy_dist > xy_tolerance:
                continue

            # Prefer highest plank below target
            if bz > best_z:
                best_z = bz
                best_plank = obs

        return best_plank

    def set_joint_limits(self, joint_limits: Dict[int, List[float]]):
        """Set joint limits for each joint.

        Args:
            joint_limits: Dict mapping joint index (0-5) to [min, max] in radians
        """
        self.joint_limits = joint_limits

    def _normalize_joint_to_limits(self, angle: float, min_limit: float, max_limit: float) -> float:
        """Normalize joint angle to fit within limits if possible.

        For revolute joints, angles are periodic (mod 2π). This finds an equivalent
        angle within the given limits by adding/subtracting 2π.

        Args:
            angle: Joint angle in radians
            min_limit: Minimum limit in radians
            max_limit: Maximum limit in radians

        Returns:
            Normalized angle within limits, or original if no valid equivalent exists
        """
        # Try adding/subtracting multiples of 2π to find a valid angle
        two_pi = 2 * np.pi
        for offset in [0, -two_pi, two_pi, -2*two_pi, 2*two_pi]:
            normalized = angle + offset
            if min_limit <= normalized <= max_limit:
                return normalized
        return angle  # Return original if no valid equivalent found

    def _normalize_joints_to_limits(self, joints: List[float]) -> List[float]:
        """Normalize all joint angles to fit within configured limits.

        Args:
            joints: List of 6 joint angles in radians

        Returns:
            List of normalized joint angles
        """
        if self.joint_limits is None:
            return list(joints)

        normalized = []
        for i, j in enumerate(joints):
            if i in self.joint_limits:
                min_j, max_j = self.joint_limits[i]
                normalized.append(self._normalize_joint_to_limits(j, min_j, max_j))
            else:
                normalized.append(j)
        return normalized

    def _check_joint_limits(self, joints: List[float]) -> bool:
        """Check if joint configuration is within configured limits.

        Args:
            joints: List of 6 joint angles in radians

        Returns:
            True if all joints are within limits, False otherwise
        """
        if self.joint_limits is None:
            return True

        for i, j in enumerate(joints):
            if i in self.joint_limits:
                min_j, max_j = self.joint_limits[i]
                if j < min_j or j > max_j:
                    print(f"Joint {i} out of limits: {np.degrees(j):.1f}° "
                          f"(limits: {np.degrees(min_j):.1f}° to {np.degrees(max_j):.1f}°)")
                    return False
        return True

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _compute_grasp_orientation(
        self,
        block_id: int,
        T_base_marker: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> np.ndarray:
        """Compute gripper orientation for grasping a block.

        Combines block's detected Z-rotation with configured grasp offset.
        Gripper always points down (Z-down), but rotates around Z to align
        with block sides.

        Selects the grasp angle that results in total Z-rotation closest to 0,
        minimizing wrist joint movement. If target_orientation is provided
        (e.g., for cover action), selects an angle compatible with the target.

        Args:
            block_id: Block marker ID
            T_base_marker: 4x4 transform of marker in base frame (optional)
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            Rotation vector [rx, ry, rz] for gripper pose
        """
        # If no transform provided, use default orientation
        if T_base_marker is None:
            return self.default_orientation.copy()

        # Extract block's Z-rotation from marker transform
        R_marker = T_base_marker[:3, :3]
        block_z_angle = np.arctan2(R_marker[1, 0], R_marker[0, 0])

        # Get configured grasp angles for this block (default: [0])
        grasp_angles = self.grasp_orientations.get(block_id, [0.0])
        if isinstance(grasp_angles, (int, float)):
            grasp_angles = [grasp_angles]  # Handle single value

        # Compute total rotation for each candidate angle
        candidates = [
            (a, self._normalize_angle(block_z_angle + a))
            for a in grasp_angles
        ]

        if target_orientation is not None:
            # For cover action: only allow 0° or 180° from target (plank is symmetric)
            # Filter candidates to those within ~45° of target or target+180°
            valid_candidates = []
            for angle, total_rot in candidates:
                diff = abs(self._normalize_angle(total_rot - target_orientation))
                # Accept if within ~45° of 0° or 180° from target
                if diff < np.pi / 4 or abs(diff - np.pi) < np.pi / 4:
                    valid_candidates.append((angle, total_rot, diff))

            if valid_candidates:
                # Pick the one closest to target (prefer 0° over 180°)
                best = min(valid_candidates, key=lambda x: min(x[2], abs(x[2] - np.pi)))
                best_angle, total_z_rotation = best[0], best[1]
                print(f"  [Grasp] Selected orientation {np.degrees(total_z_rotation):.1f}° "
                      f"for target {np.degrees(target_orientation):.1f}°")
            else:
                # Fall back to original selection if no valid candidates
                best_angle, total_z_rotation = min(candidates, key=lambda x: abs(x[1]))
                print(f"  [Grasp] Warning: No valid orientation for target, using {np.degrees(total_z_rotation):.1f}°")
        else:
            # Original behavior: pick angle closest to 0 (minimize wrist rotation)
            best_angle, total_z_rotation = min(candidates, key=lambda x: abs(x[1]))

        # Build gripper orientation: 180° around X (pointing down) + Z rotation
        R_gripper = Rotation.from_euler('xz', [np.pi, total_z_rotation])
        return R_gripper.as_rotvec()

    def _compute_place_orientation(self, target_yaw: float) -> np.ndarray:
        """
        Compute gripper orientation for placing a block at target yaw.

        Args:
            target_yaw: Target yaw angle in radians (Z-axis rotation)

        Returns:
            Rotation vector [rx, ry, rz] for gripper pose
        """
        # Gripper points down (180° around X) with Z rotation for target yaw
        R_gripper = Rotation.from_euler('xz', [np.pi, target_yaw])
        return R_gripper.as_rotvec()

    def move_to_cartesian(
        self,
        position: List[float],
        orientation: Optional[np.ndarray] = None,
        linear: bool = False,
        speed: Optional[float] = None
    ) -> bool:
        """
        Move robot to a Cartesian position.

        Args:
            position: [x, y, z] target position in meters
            orientation: Optional axis-angle orientation [rx, ry, rz].
                        If None, uses default gripper-down orientation.
            linear: If True, use linear motion (moveL). If False, use joint motion.
            speed: Optional speed override

        Returns:
            True if successful
        """
        pose = self._pose_to_list(np.array(position), orientation)
        if linear:
            return self._move_linear(pose, speed)
        else:
            return self._move_to_pose(pose, speed)

    # =========================================================================
    # High-Level Actions (matching PDDL actions)
    # =========================================================================

    def _pick_block(
        self,
        block_id: int,
        observations: List[List],
        action_desc: str,
        block_transform: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> bool:
        """
        Common pick logic for pick_up, unstack, and remove_beside.

        Args:
            block_id: ID of block to pick
            observations: Current world state
            action_desc: Description for logging (e.g., "Picking", "Unstacking")
            block_transform: Optional 4x4 transform of block in base frame (for orientation)
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            True if successful
        """
        block = self._get_block_from_obs(block_id, observations)
        if block is None:
            print(f"Block {block_id} not found in observations")
            return False

        x, y, z = block[2], block[3], block[4]

        # Compute grasp orientation based on block pose (with optional target for cover)
        grasp_orientation = self._compute_grasp_orientation(block_id, block_transform, target_orientation)

        # Log orientation info
        if block_transform is not None:
            R_marker = block_transform[:3, :3]
            block_z_angle = np.arctan2(R_marker[1, 0], R_marker[0, 0])
            print(f"{action_desc} at [{x:.3f}, {y:.3f}, {z:.3f}], block_rot={np.degrees(block_z_angle):.1f}°")
        else:
            print(f"{action_desc} at [{x:.3f}, {y:.3f}, {z:.3f}]")

        # Calculate positions
        grasp_z = z - self.grasp_depth + self.gripper_z_offset
        approach_pos = np.array([x, y, grasp_z + self.approach_height])
        grasp_pos = np.array([x, y, grasp_z])

        # Execute pick sequence
        # First ensure we're at safe height to avoid low traversal paths
        if not self._ensure_safe_height():
            return False

        self._open_gripper(block_id)

        # Move to approach (with computed orientation)
        if not self._move_to_pose(self._pose_to_list(approach_pos, grasp_orientation)):
            return False

        # Move down to grasp (maintain orientation)
        if not self._move_linear(self._pose_to_list(grasp_pos, grasp_orientation)):
            return False

        # Grasp
        self._close_gripper(block_id)

        # Lift (maintain orientation)
        if not self._move_linear(self._pose_to_list(approach_pos, grasp_orientation)):
            return False

        return True

    def pick_up(
        self,
        block_id: int,
        table_id: int,
        robot_id: int,
        observations: List[List],
        block_transform: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> bool:
        """
        Pick up a block from the table.

        PDDL action: (pick-up ?b1 ?t1 ?r1)

        Args:
            block_id: ID of block to pick
            table_id: ID of table (unused, for PDDL compatibility)
            robot_id: ID of robot (unused, for PDDL compatibility)
            observations: Current world state
            block_transform: Optional 4x4 transform for grasp orientation
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            True if successful
        """
        success = self._pick_block(block_id, observations, f"Picking block {block_id}", block_transform, target_orientation)
        if success:
            print(f"Picked block {block_id}")
        return success

    def unstack(
        self,
        block_id: int,
        under_block_id: int,
        robot_id: int,
        observations: List[List],
        block_transform: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> bool:
        """
        Unstack a block from on top of another block.

        PDDL action: (unstack ?b1 ?b2 ?r1)

        Args:
            block_id: ID of block to pick (on top)
            under_block_id: ID of block underneath (unused, position from observations)
            robot_id: ID of robot (unused, for PDDL compatibility)
            observations: Current world state
            block_transform: Optional 4x4 transform for grasp orientation
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            True if successful
        """
        success = self._pick_block(
            block_id, observations,
            f"Unstacking block {block_id} from block {under_block_id}",
            block_transform, target_orientation
        )
        if success:
            print(f"Unstacked block {block_id}")
        return success

    def remove_beside(
        self,
        block_id: int,
        beside_block_id: int,
        table_id: int,
        robot_id: int,
        observations: List[List],
        block_transform: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> bool:
        """
        Pick up a block that is beside another block.

        PDDL action: (remove-beside ?b1 ?b2 ?t1 ?r1)

        Args:
            block_id: ID of block to pick (beside another)
            beside_block_id: ID of block it's beside (unused, position from observations)
            table_id: ID of table (unused)
            robot_id: ID of robot (unused)
            observations: Current world state
            block_transform: Optional 4x4 transform for grasp orientation
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            True if successful
        """
        success = self._pick_block(
            block_id, observations,
            f"Removing block {block_id} from beside block {beside_block_id}",
            block_transform, target_orientation
        )
        if success:
            print(f"Removed block {block_id} from beside")
        return success

    def pick_floating(
        self,
        block_id: int,
        robot_id: int,
        observations: List[List],
        block_transform: Optional[np.ndarray] = None,
        target_orientation: Optional[float] = None
    ) -> bool:
        """
        Pick up a floating block (perception error recovery).

        PDDL action: (pick_floating ?b1 ?r1)

        Floating blocks are blocks that are neither on-table nor above anything,
        typically due to perception errors. This action picks them up so they
        can be released to a valid position on the table.

        Args:
            block_id: ID of block to pick
            robot_id: ID of robot (unused, for PDDL compatibility)
            observations: Current world state
            block_transform: Optional 4x4 transform for grasp orientation
            target_orientation: Target yaw for placement (optional, for cover action)

        Returns:
            True if successful
        """
        success = self._pick_block(
            block_id, observations,
            f"Picking floating block {block_id}",
            block_transform, target_orientation
        )
        if success:
            print(f"Picked floating block {block_id}")
        return success

    def put_down(
        self,
        block_id: int,
        target_id: int,
        robot_id: int,
        observations: List[List]
    ) -> bool:
        """
        Put down a block on top of another block (stack).

        PDDL action: (put-down ?b1 ?b2 ?r1)

        Args:
            block_id: ID of block being held
            target_id: ID of block to place on
            robot_id: ID of robot (unused)
            observations: Current world state

        Returns:
            True if successful
        """
        block = self._get_block_from_obs(block_id, observations)
        target = self._get_block_from_obs(target_id, observations)

        if block is None or target is None:
            print(f"Block {block_id} or target {target_id} not found")
            return False

        # Target position (tz is already the top of target block)
        tx, ty, tz = target[2], target[3], target[4]

        # Determine orientation reference block
        # For planks placed on non-planks, look for a reference plank below in the stack
        block_class = block[1]
        target_class = target[1]
        reference_block = target
        reference_id = target_id

        if block_class == 3 and target_class != 3:
            # Placing plank on non-plank - look for reference plank below
            ref_plank = self._find_reference_plank_below(tx, ty, tz, observations)
            if ref_plank is not None:
                reference_block = ref_plank
                reference_id = ref_plank[0]
                print(f"  [Reference] Using plank {reference_id} below for orientation")

        # Get reference yaw for alignment - place block with long axis parallel to reference
        reference_yaw = reference_block[8] if len(reference_block) > 8 else 0.0

        # Correct for different aspect ratios: if blocks have different long axis directions,
        # we need to offset the yaw so their long axes end up parallel
        # Long axis offset: 0 if X is long, π/2 if Y is long
        placed_long_axis = self._get_long_axis_offset(block_id)
        reference_long_axis = self._get_long_axis_offset(reference_id)
        axis_correction = placed_long_axis - reference_long_axis
        corrected_yaw = reference_yaw - axis_correction

        # Compensate for grasp offset: block's final yaw = gripper_yaw - grasp_offset
        # So gripper_yaw = corrected_yaw + grasp_offset to achieve final_yaw = corrected_yaw
        grasp_angles = self.grasp_orientations.get(block_id, [0.0])
        grasp_offset = grasp_angles[0] if isinstance(grasp_angles, list) else grasp_angles
        gripper_yaw = corrected_yaw + grasp_offset
        place_orientation = self._compute_place_orientation(gripper_yaw)

        # Held block dimensions
        bh = block[7]

        # Approach position: above target at approach_height
        # Use target Z + block height as reference for approach
        approach_z = tz + bh + self.approach_height + self.gripper_z_offset
        approach_pos = np.array([tx, ty, approach_z])

        print(f"Placing block {block_id} on block {target_id} "
              f"(ref={reference_id}, ref_yaw={np.degrees(reference_yaw):.1f}°, axis_corr={np.degrees(axis_correction):.1f}°, "
              f"grasp_offset={np.degrees(grasp_offset):.1f}°, gripper_yaw={np.degrees(gripper_yaw):.1f}°)")

        # Move to approach with target orientation
        if not self._move_to_pose(self._pose_to_list(approach_pos, place_orientation)):
            return False

        # Move down until contact detected
        print(f"  Moving down until contact...")
        if not self._move_until_contact():
            print(f"  Warning: No contact detected, continuing anyway")

        # Release
        self._open_gripper(block_id)

        # Retract (maintain orientation)
        if not self._move_linear(self._pose_to_list(approach_pos, place_orientation)):
            return False

        print(f"Placed block {block_id} on block {target_id}")
        return True

    def align(
        self,
        block_id: int,
        target_id: int,
        table_id: int,
        robot_id: int,
        observations: List[List]
    ) -> bool:
        """
        Place block beside another block on the table (for bridge foundation).

        PDDL action: (align ?b1 ?b2 ?t1 ?r1)

        Args:
            block_id: ID of block being held
            target_id: ID of block to place beside
            table_id: ID of table
            robot_id: ID of robot (unused)
            observations: Current world state

        Returns:
            True if successful
        """
        block = self._get_block_from_obs(block_id, observations)
        target = self._get_block_from_obs(target_id, observations)
        table = self._get_block_from_obs(table_id, observations)

        if block is None or target is None:
            print(f"Block {block_id} or target {target_id} not found")
            return False

        # Target position
        tx, ty, tz = target[2], target[3], target[4]
        tl = target[6]  # target length (for offset)

        # Get target yaw for alignment - place block with long axis parallel to target's long axis
        target_yaw = target[8] if len(target) > 8 else 0.0

        # Correct for different aspect ratios: if blocks have different long axis directions,
        # we need to offset the yaw so their long axes end up parallel
        # Long axis offset: 0 if X is long, π/2 if Y is long
        placed_long_axis = self._get_long_axis_offset(block_id)
        target_long_axis = self._get_long_axis_offset(target_id)
        axis_correction = placed_long_axis - target_long_axis
        corrected_yaw = target_yaw - axis_correction

        # Compensate for grasp offset: block's final yaw = gripper_yaw - grasp_offset
        # So gripper_yaw = corrected_yaw + grasp_offset to achieve final_yaw = corrected_yaw
        grasp_angles = self.grasp_orientations.get(block_id, [0.0])
        grasp_offset = grasp_angles[0] if isinstance(grasp_angles, list) else grasp_angles
        gripper_yaw = corrected_yaw + grasp_offset
        place_orientation = self._compute_place_orientation(gripper_yaw)

        # Held block dimensions
        bl = block[6]  # block length
        bh = block[7]  # block height

        # Place beside: offset perpendicular to target's orientation
        # offset_dist = half of target length + half of block length + gap
        offset_dist = tl / 2 + bl / 2 + self.beside_gap

        # Offset in direction perpendicular to target yaw (use original target_yaw for position)
        # yaw=0 means facing +X, so perpendicular is +Y
        # For a yaw of θ, perpendicular direction is (sin(θ), -cos(θ)) rotated 90°
        # Actually: perpendicular to yaw direction is (-sin(yaw), cos(yaw))
        place_x = tx - offset_dist * np.sin(target_yaw)
        place_y = ty + offset_dist * np.cos(target_yaw)

        table_z = table[4] if table else 0.0

        # Approach position: above table at approach_height
        approach_z = table_z + bh + self.approach_height + self.gripper_z_offset
        approach_pos = np.array([place_x, place_y, approach_z])

        print(f"Aligning block {block_id} beside block {target_id} "
              f"(target_yaw={np.degrees(target_yaw):.1f}°, axis_corr={np.degrees(axis_correction):.1f}°, "
              f"grasp_offset={np.degrees(grasp_offset):.1f}°, gripper_yaw={np.degrees(gripper_yaw):.1f}°)")

        # Move to approach with target orientation
        if not self._move_to_pose(self._pose_to_list(approach_pos, place_orientation)):
            return False

        # Move down until contact detected
        print(f"  Moving down until contact...")
        if not self._move_until_contact():
            print(f"  Warning: No contact detected, continuing anyway")

        # Release
        self._open_gripper(block_id)

        # Retract (maintain orientation)
        if not self._move_linear(self._pose_to_list(approach_pos, place_orientation)):
            return False

        print(f"Aligned block {block_id} beside block {target_id}")
        return True

    def cover(
        self,
        plank_id: int,
        left_id: int,
        right_id: int,
        robot_id: int,
        observations: List[List]
    ) -> bool:
        """
        Place a plank across two pillars (bridge top).

        PDDL action: (cover ?b1 ?b2 ?b3 ?r1)

        Args:
            plank_id: ID of plank being held
            left_id: ID of left pillar
            right_id: ID of right pillar
            robot_id: ID of robot (unused)
            observations: Current world state

        Returns:
            True if successful
        """
        plank = self._get_block_from_obs(plank_id, observations)
        left = self._get_block_from_obs(left_id, observations)
        right = self._get_block_from_obs(right_id, observations)

        if plank is None or left is None or right is None:
            print(f"Plank {plank_id} or pillars not found")
            return False

        # Get pillar positions (lz, rz are already pillar tops)
        lx, ly, lz = left[2], left[3], left[4]
        rx, ry, rz = right[2], right[3], right[4]

        # Compute plank orientation from pillar positions
        # Plank's long axis should be parallel to the line connecting pillars
        # Gripper grasps along short side, so gripper yaw = pillar_angle + 90°
        dx = rx - lx
        dy = ry - ly
        pillar_angle = np.arctan2(dy, dx)
        gripper_yaw = pillar_angle + np.pi / 2  # Perpendicular to pillar line
        place_orientation = self._compute_place_orientation(gripper_yaw)

        # Plank dimensions
        ph = plank[7]

        # Place position: centered between pillars, on top
        place_x = (lx + rx) / 2
        place_y = (ly + ry) / 2
        # lz, rz are already pillar tops (marker positions)
        pillar_top = max(lz, rz)

        # Approach position: above pillars at approach_height
        approach_z = pillar_top + ph + self.approach_height + self.gripper_z_offset
        approach_pos = np.array([place_x, place_y, approach_z])

        print(f"Covering pillars {left_id}, {right_id} with plank {plank_id} "
              f"(pillar_angle={np.degrees(pillar_angle):.1f}°, gripper_yaw={np.degrees(gripper_yaw):.1f}°)")

        # Move to approach with plank orientation
        if not self._move_to_pose(self._pose_to_list(approach_pos, place_orientation)):
            return False

        # Move down until contact detected
        print(f"  Moving down until contact...")
        if not self._move_until_contact():
            print(f"  Warning: No contact detected, continuing anyway")

        # Release
        self._open_gripper(plank_id)

        # Retract (maintain orientation)
        if not self._move_linear(self._pose_to_list(approach_pos, place_orientation)):
            return False

        print(f"Covered with plank {plank_id}")
        return True

    def _point_in_rounded_rect(
        self,
        px: float, py: float,
        rect_x: float, rect_y: float,
        rect_w: float, rect_l: float,
        rect_yaw: float,
        radius: float
    ) -> bool:
        """
        Check if point (px, py) is inside a rounded rectangle.

        The rounded rectangle is centered at (rect_x, rect_y), has dimensions
        rect_w × rect_l, is rotated by rect_yaw, and has corner radius = radius.

        This represents the Minkowski sum of the rectangle with a circle of given radius.

        Args:
            px, py: Point to test
            rect_x, rect_y: Center of rectangle
            rect_w, rect_l: Width and length of rectangle
            rect_yaw: Rotation angle of rectangle (radians)
            radius: Inflation radius (corner radius)

        Returns:
            True if point is inside the rounded rectangle
        """
        # Transform point into rectangle's local coordinate frame
        dx = px - rect_x
        dy = py - rect_y
        cos_yaw = np.cos(-rect_yaw)
        sin_yaw = np.sin(-rect_yaw)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        # Half-dimensions of the inner rectangle (before inflation)
        half_w = rect_w / 2
        half_l = rect_l / 2

        # Check if inside the inflated rounded rectangle
        # This is equivalent to checking distance to the original rectangle

        # Clamp to rectangle bounds to find nearest point on rectangle edge
        nearest_x = max(-half_w, min(half_w, local_x))
        nearest_y = max(-half_l, min(half_l, local_y))

        # Distance from point to nearest point on rectangle
        dist_sq = (local_x - nearest_x) ** 2 + (local_y - nearest_y) ** 2

        # Inside if distance < radius (or inside the rectangle itself)
        return dist_sq < radius ** 2

    def release(
        self,
        block_id: int,
        table_id: int,
        robot_id: int,
        observations: List[List]
    ) -> bool:
        """
        Release block onto an empty spot on the table.

        PDDL action: (release ?b1 ?t1 ?r1)

        Finds an empty spot on the table away from other blocks and places there.

        Args:
            block_id: ID of block being held
            table_id: ID of table
            robot_id: ID of robot (unused)
            observations: Current world state

        Returns:
            True if successful
        """
        table = self._get_block_from_obs(table_id, observations)
        block = self._get_block_from_obs(block_id, observations)

        if block is None:
            print(f"Block {block_id} not found in observations")
            return False

        # Block dimensions
        bw, bl, bh = block[5], block[6], block[7]

        # Get table Z (surface height)
        table_z = table[4] if table else 0.0

        # Find empty spot: collect positions, sizes, and yaw of all other blocks on table
        other_blocks = []
        for obs in observations:
            if obs[0] != block_id and obs[1] in [2, 3]:  # Other cubes/planks
                # Store position, size, and yaw: (x, y, width, length, yaw)
                yaw = float(obs[8]) if len(obs) > 8 else 0.0
                other_blocks.append((obs[2], obs[3], obs[5], obs[6], yaw))

        # Calculate center of existing blocks as base
        tb = self.table_bounds
        if other_blocks:
            avg_x = sum(x for x, y, w, l, yaw in other_blocks) / len(other_blocks)
            avg_y = sum(y for x, y, w, l, yaw in other_blocks) / len(other_blocks)
        else:
            avg_x = (tb['x_min'] + tb['x_max']) / 2
            avg_y = (tb['y_min'] + tb['y_max']) / 2

        # Search for empty spot near the cluster, with offset to avoid collision
        # Try positions in a spiral pattern around the cluster center
        place_x, place_y = avg_x, avg_y
        found = False

        # Inflation radius for collision detection (1/4 of original to prevent beside after release)
        inflation = (self.release_min_dist + max(bw, bl) / 2) / 4

        # Search offsets: start far, move inward (places blocks far from cluster)
        offsets = []
        for r in [self.release_min_dist * 5, self.release_min_dist * 4,
                  self.release_min_dist * 3, self.release_min_dist * 2.5,
                  self.release_min_dist * 2, self.release_min_dist * 1.5,
                  self.release_min_dist, self.release_min_dist * 0.5]:
            for angle_idx in range(12):  # 12 directions (30 degree increments)
                angle = angle_idx * (np.pi / 6)
                dx = r * np.cos(angle)
                dy = r * np.sin(angle)
                offsets.append((dx, dy))

        for dx, dy in offsets:
            candidate_x = avg_x + dx
            candidate_y = avg_y + dy

            # Check if within table bounds (with margin for block size)
            margin_x = bw / 2 + 0.01
            margin_y = bl / 2 + 0.01
            if not (tb['x_min'] + margin_x < candidate_x < tb['x_max'] - margin_x and
                    tb['y_min'] + margin_y < candidate_y < tb['y_max'] - margin_y):
                continue

            # Check if far enough from all other blocks using rounded rectangle collision
            is_empty = True
            for (ox, oy, ow, ol, oyaw) in other_blocks:
                if self._point_in_rounded_rect(
                    candidate_x, candidate_y,
                    ox, oy, ow, ol, oyaw,
                    inflation
                ):
                    is_empty = False
                    break

            if is_empty:
                place_x, place_y = candidate_x, candidate_y
                found = True
                break

        if not found:
            # No valid spot found - return to home position while keeping block held
            print(f"  [Release] WARNING: No empty spot found for block {block_id}")
            print(f"    Table bounds: x=[{tb['x_min']:.2f}, {tb['x_max']:.2f}], y=[{tb['y_min']:.2f}, {tb['y_max']:.2f}]")
            print(f"    Other blocks: {len(other_blocks)}, cluster center: ({avg_x:.3f}, {avg_y:.3f})")
            print(f"  [Release] Returning to home position while keeping block held...")
            self._move_to_home()
            return False

        # Approach position: above table at approach_height
        approach_z = table_z + bh + self.approach_height + self.gripper_z_offset
        approach_pos = np.array([place_x, place_y, approach_z])

        print(f"Releasing block {block_id} to empty spot at [{place_x:.3f}, {place_y:.3f}]")

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down until contact detected
        print(f"  Moving down until contact...")
        if not self._move_until_contact():
            print(f"  Warning: No contact detected, continuing anyway")

        # Release
        self._open_gripper(block_id)

        # Retract
        if not self._move_linear(self._pose_to_list(approach_pos)):
            return False

        print(f"Released block {block_id} on table")
        return True

    def rotate(
        self,
        block_id: int,
        target_id: int,
        robot_id: int,
        observations: List[List],
        block_transforms: Optional[Dict[int, np.ndarray]] = None
    ) -> bool:
        """
        Rotate held block to align with target block orientation.

        PDDL action: (rotate ?b1 ?b2 ?r1)

        This action rotates the gripper (with held block) in place to match
        the target block's orientation. Used before place actions when
        alignment is required.

        Args:
            block_id: ID of held block
            target_id: ID of target block to align with
            robot_id: ID of robot (unused)
            observations: Current world state
            block_transforms: Optional dict mapping block_id to 4x4 transform

        Returns:
            True if successful
        """
        target = self._get_block_from_obs(target_id, observations)
        if target is None:
            print(f"Target block {target_id} not found")
            return False

        # Get target yaw from observations
        target_yaw = target[8] if len(target) > 8 else 0.0

        # Compute target gripper orientation
        target_orientation = self._compute_place_orientation(target_yaw)

        # Get current gripper position
        current_pose = self.rtde_r.getActualTCPPose()
        current_pos = np.array(current_pose[:3])

        # Rotate in place (move to same position with new orientation)
        pose = self._pose_to_list(current_pos, target_orientation)

        print(f"Rotating held block {block_id} to align with block {target_id} "
              f"(target_yaw={np.degrees(target_yaw):.1f}°)")

        if not self._move_to_pose(pose):
            return False

        print(f"Rotated block {block_id} to match orientation of block {target_id}")
        return True

    # =========================================================================
    # Action Dispatcher
    # =========================================================================

    def execute_action(
        self,
        action_name: str,
        args: Tuple,
        observations: List[List],
        block_transforms: Optional[Dict[int, np.ndarray]] = None,
        next_action: Optional[Tuple[str, Tuple]] = None
    ) -> bool:
        """
        Execute a PDDL action.

        Args:
            action_name: Name of action (e.g., 'pick-up', 'put-down')
            args: Action arguments as tuple of string IDs
            observations: Current world state
            block_transforms: Optional dict mapping block_id to 4x4 transform (for grasp orientation)
            next_action: Optional tuple (action_name, args) for lookahead (to compute target orientation)

        Returns:
            True if successful
        """
        # Convert string IDs to integers
        int_args = [int(a) for a in args]

        # Get block transform for pick actions (first arg is block_id)
        block_transform = None
        if block_transforms is not None and len(int_args) > 0:
            block_transform = block_transforms.get(int_args[0])

        # Compute target_orientation for pick actions based on next action lookahead
        target_orientation = None
        if next_action:
            next_name = next_action[0]
            next_args = [int(a) for a in next_action[1]]

            if next_name == 'cover':
                # cover args: (plank_id, left_id, right_id, robot_id)
                if len(next_args) >= 3:
                    left_id, right_id = next_args[1], next_args[2]
                    left = self._get_block_from_obs(left_id, observations)
                    right = self._get_block_from_obs(right_id, observations)
                    if left is not None and right is not None:
                        dx = right[2] - left[2]
                        dy = right[3] - left[3]
                        pillar_angle = np.arctan2(dy, dx)
                        # Add 90° because gripper grasps plank along short side,
                        # so gripper yaw should be perpendicular to pillar line
                        # for plank's long axis to be parallel to pillar line
                        target_orientation = pillar_angle + np.pi / 2
                        print(f"  [Cover lookahead] Pillars at ({left[2]:.3f},{left[3]:.3f}) and ({right[2]:.3f},{right[3]:.3f})")
                        print(f"  [Cover lookahead] Pillar angle: {np.degrees(pillar_angle):.1f}°, gripper target: {np.degrees(target_orientation):.1f}°")

            elif next_name in ['align', 'put-down']:
                # align args: (moving_id, target_id, robot_id)
                # put-down args: (block_id, target_id, robot_id)
                # Placement uses gripper_yaw = (ref_yaw - axis_correction) + grasp_offset
                # So lookahead should match this expected gripper_yaw
                if len(next_args) >= 2:
                    target_block_id = next_args[1]  # args[1] is the target block
                    target_block = self._get_block_from_obs(target_block_id, observations)
                    if target_block is not None and len(target_block) > 8:
                        pick_block_id = int_args[0]
                        pick_block = self._get_block_from_obs(pick_block_id, observations)

                        # Determine reference block for orientation
                        reference_block = target_block
                        reference_id = target_block_id

                        # For put-down: if placing plank on non-plank, look for reference plank below
                        if (next_name == 'put-down' and pick_block is not None and
                            pick_block[1] == 3 and target_block[1] != 3):
                            tx, ty, tz = target_block[2], target_block[3], target_block[4]
                            ref_plank = self._find_reference_plank_below(tx, ty, tz, observations)
                            if ref_plank is not None:
                                reference_block = ref_plank
                                reference_id = ref_plank[0]
                                print(f"  [Lookahead] Using plank {reference_id} below for orientation")

                        reference_yaw = float(reference_block[8])

                        # Axis correction for different aspect ratios (parallel long axes)
                        placed_long_axis = self._get_long_axis_offset(pick_block_id)
                        reference_long_axis = self._get_long_axis_offset(reference_id)
                        axis_correction = placed_long_axis - reference_long_axis
                        corrected_yaw = reference_yaw - axis_correction

                        # Add grasp offset of block being picked
                        grasp_angles = self.grasp_orientations.get(pick_block_id, [0.0])
                        grasp_offset = grasp_angles[0] if isinstance(grasp_angles, list) else grasp_angles
                        target_orientation = corrected_yaw + grasp_offset
                        print(f"  [Align/Put-down lookahead] ref={reference_id}, ref_yaw: {np.degrees(reference_yaw):.1f}°, "
                              f"axis_corr: {np.degrees(axis_correction):.1f}°, "
                              f"grasp_offset: {np.degrees(grasp_offset):.1f}°, gripper target: {np.degrees(target_orientation):.1f}°")

        action_map = {
            'pick-up': lambda: self.pick_up(*int_args, observations, block_transform, target_orientation),
            'unstack': lambda: self.unstack(*int_args, observations, block_transform, target_orientation),
            'remove-beside': lambda: self.remove_beside(*int_args, observations, block_transform, target_orientation),
            'pick_floating': lambda: self.pick_floating(*int_args, observations, block_transform, target_orientation),
            'put-down': lambda: self.put_down(*int_args, observations),
            'align': lambda: self.align(*int_args, observations),
            'cover': lambda: self.cover(*int_args, observations),
            'release': lambda: self.release(*int_args, observations),
            'rotate': lambda: self.rotate(*int_args, observations, block_transforms),
        }

        if action_name not in action_map:
            print(f"Unknown action: {action_name}")
            return False

        return action_map[action_name]()
