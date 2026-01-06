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

        # Default orientation: gripper pointing down
        # This is a 180 degree rotation around X (gripper Z pointing down)
        self.default_orientation = Rotation.from_rotvec([np.pi, 0, 0]).as_rotvec()

        # Grasp orientations per block ID (will be set from config)
        self.grasp_orientations: Dict[int, List[float]] = {}

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

    def _open_gripper(self):
        """Open the gripper."""
        if self.gripper:
            self.gripper.move(self.gripper_open_pos, self.gripper_speed, self.gripper_force)
            time.sleep(self.gripper_open_wait)

    def _close_gripper(self):
        """Close the gripper and wait until it finishes moving."""
        if self.gripper:
            # Use blocking move_and_wait_for_pos instead of non-blocking move
            if hasattr(self.gripper, 'move_and_wait_for_pos'):
                pos, status = self.gripper.move_and_wait_for_pos(
                    self.gripper_close_pos, self.gripper_speed, self.gripper_force
                )
                print(f"Gripper closed: pos={pos}, status={status}")
            else:
                # Fallback to non-blocking with longer wait
                self.gripper.move(self.gripper_close_pos, self.gripper_speed, self.gripper_force)
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

    def open_gripper(self):
        """Open the gripper (public interface)."""
        self._open_gripper()

    def close_gripper(self):
        """Close the gripper (public interface)."""
        self._close_gripper()

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
        T_base_marker: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gripper orientation for grasping a block.

        Combines block's detected Z-rotation with configured grasp offset.
        Gripper always points down (Z-down), but rotates around Z to align
        with block sides.

        Selects the grasp angle that results in total Z-rotation closest to 0,
        minimizing wrist joint movement.

        Args:
            block_id: Block marker ID
            T_base_marker: 4x4 transform of marker in base frame (optional)

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
        # Pick the one that results in normalized total_z_rotation closest to 0
        # This minimizes wrist joint rotation from default pose
        candidates = [
            (a, self._normalize_angle(block_z_angle + a))
            for a in grasp_angles
        ]
        best_angle, total_z_rotation = min(candidates, key=lambda x: abs(x[1]))

        # Build gripper orientation: 180° around X (pointing down) + Z rotation
        R_gripper = Rotation.from_euler('xz', [np.pi, total_z_rotation])
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
        block_transform: Optional[np.ndarray] = None
    ) -> bool:
        """
        Common pick logic for pick_up, unstack, and remove_beside.

        Args:
            block_id: ID of block to pick
            observations: Current world state
            action_desc: Description for logging (e.g., "Picking", "Unstacking")
            block_transform: Optional 4x4 transform of block in base frame (for orientation)

        Returns:
            True if successful
        """
        block = self._get_block_from_obs(block_id, observations)
        if block is None:
            print(f"Block {block_id} not found in observations")
            return False

        x, y, z = block[2], block[3], block[4]

        # Compute grasp orientation based on block pose
        grasp_orientation = self._compute_grasp_orientation(block_id, block_transform)

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
        self._open_gripper()

        # Move to approach (with computed orientation)
        if not self._move_to_pose(self._pose_to_list(approach_pos, grasp_orientation)):
            return False

        # Move down to grasp (maintain orientation)
        if not self._move_linear(self._pose_to_list(grasp_pos, grasp_orientation)):
            return False

        # Grasp
        self._close_gripper()

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
        block_transform: Optional[np.ndarray] = None
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

        Returns:
            True if successful
        """
        success = self._pick_block(block_id, observations, f"Picking block {block_id}", block_transform)
        if success:
            print(f"Picked block {block_id}")
        return success

    def unstack(
        self,
        block_id: int,
        under_block_id: int,
        robot_id: int,
        observations: List[List],
        block_transform: Optional[np.ndarray] = None
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

        Returns:
            True if successful
        """
        success = self._pick_block(
            block_id, observations,
            f"Unstacking block {block_id} from block {under_block_id}",
            block_transform
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
        block_transform: Optional[np.ndarray] = None
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

        Returns:
            True if successful
        """
        success = self._pick_block(
            block_id, observations,
            f"Removing block {block_id} from beside block {beside_block_id}",
            block_transform
        )
        if success:
            print(f"Removed block {block_id} from beside")
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

        # Held block dimensions
        bh = block[7]

        # Place position: on top of target
        # tz is already the top of target block (marker position)
        # Block is held with grasp at grasp_depth below its top, so:
        #   block_bottom = fingertips_z + grasp_depth - bh
        # We want block_bottom = tz + clearance, so:
        #   fingertips_z = tz + clearance + bh - grasp_depth
        place_z = tz + bh - self.grasp_depth + self.place_clearance + self.gripper_z_offset

        approach_pos = np.array([tx, ty, place_z + self.approach_height])
        place_pos = np.array([tx, ty, place_z])

        print(f"Placing block {block_id} on block {target_id} at [{tx:.3f}, {ty:.3f}, {place_z - self.gripper_z_offset:.3f}]")

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down to place
        if not self._move_linear(self._pose_to_list(place_pos)):
            return False

        # Release
        self._open_gripper()

        # Retract
        if not self._move_linear(self._pose_to_list(approach_pos)):
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
        tl = target[6]  # target length (for Y offset)

        # Held block dimensions
        bl = block[6]  # block length
        bh = block[7]  # block height

        # Place beside: offset in Y direction
        place_x = tx
        place_y = ty + tl / 2 + bl / 2 + self.beside_gap
        table_z = table[4] if table else 0.0
        # Block is held with grasp at grasp_depth below its top
        # We want block_bottom = table_z + clearance
        place_z = table_z + bh - self.grasp_depth + self.place_clearance + self.gripper_z_offset

        approach_pos = np.array([place_x, place_y, place_z + self.approach_height])
        place_pos = np.array([place_x, place_y, place_z])

        print(f"Aligning block {block_id} beside block {target_id}")

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down to place
        if not self._move_linear(self._pose_to_list(place_pos)):
            return False

        # Release
        self._open_gripper()

        # Retract
        if not self._move_linear(self._pose_to_list(approach_pos)):
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

        # Plank dimensions
        ph = plank[7]

        # Place position: centered between pillars, on top
        place_x = (lx + rx) / 2
        place_y = (ly + ry) / 2
        # lz, rz are already pillar tops (marker positions)
        pillar_top = max(lz, rz)
        # Plank is held with grasp at grasp_depth below its top
        # We want plank_bottom = pillar_top + clearance
        place_z = pillar_top + ph - self.grasp_depth + self.place_clearance + self.gripper_z_offset

        approach_pos = np.array([place_x, place_y, place_z + self.approach_height])
        place_pos = np.array([place_x, place_y, place_z])

        print(f"Covering pillars {left_id}, {right_id} with plank {plank_id}")

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down to place
        if not self._move_linear(self._pose_to_list(place_pos)):
            return False

        # Release
        self._open_gripper()

        # Retract
        if not self._move_linear(self._pose_to_list(approach_pos)):
            return False

        print(f"Covered with plank {plank_id}")
        return True

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

        # Find empty spot: collect positions and sizes of all other blocks on table
        other_blocks = []
        for obs in observations:
            if obs[0] != block_id and obs[1] in [2, 3]:  # Other cubes/planks
                # Store position and size: (x, y, width, length)
                other_blocks.append((obs[2], obs[3], obs[5], obs[6]))

        # Calculate center of existing blocks as base
        tb = self.table_bounds
        if other_blocks:
            avg_x = sum(x for x, y, w, l in other_blocks) / len(other_blocks)
            avg_y = sum(y for x, y, w, l in other_blocks) / len(other_blocks)
        else:
            avg_x = (tb['x_min'] + tb['x_max']) / 2
            avg_y = (tb['y_min'] + tb['y_max']) / 2

        # Search for empty spot near the cluster, with offset to avoid collision
        # Try positions in a spiral pattern around the cluster center
        place_x, place_y = avg_x, avg_y
        found = False

        # Search offsets: start close, move outward
        offsets = []
        for r in [self.release_min_dist, self.release_min_dist * 1.5,
                  self.release_min_dist * 2, self.release_min_dist * 2.5,
                  self.release_min_dist * 3, self.release_min_dist * 4]:
            for angle_idx in range(8):  # 8 directions
                angle = angle_idx * (np.pi / 4)  # 45 degree increments
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

            # Check if far enough from all other blocks using actual block sizes
            is_empty = True
            for (ox, oy, ow, ol) in other_blocks:
                # Minimum distance = half of each block's size + safety margin
                min_dist_x = (bw + ow) / 2 + 0.05  # 5cm safety margin (larger for safe release)
                min_dist_y = (bl + ol) / 2 + 0.05
                # Use axis-aligned bounding box collision check
                if abs(candidate_x - ox) < min_dist_x and abs(candidate_y - oy) < min_dist_y:
                    is_empty = False
                    break

            if is_empty:
                place_x, place_y = candidate_x, candidate_y
                found = True
                break

        if not found:
            # Fallback: place at offset from center (further out)
            place_x = avg_x + self.release_min_dist * 3
            place_y = avg_y
            print(f"  [Release] Warning: No ideal spot found, using fallback position")

        # Place position on table
        # Block is held with grasp at grasp_depth below its top
        # We want block_bottom = table_z + clearance
        place_z = table_z + bh - self.grasp_depth + self.place_clearance + self.gripper_z_offset

        approach_pos = np.array([place_x, place_y, place_z + self.approach_height])
        place_pos = np.array([place_x, place_y, place_z])

        print(f"Releasing block {block_id} to empty spot at [{place_x:.3f}, {place_y:.3f}]")

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down to place
        if not self._move_linear(self._pose_to_list(place_pos)):
            return False

        # Release
        self._open_gripper()

        # Retract
        if not self._move_linear(self._pose_to_list(approach_pos)):
            return False

        print(f"Released block {block_id} on table")
        return True

    # =========================================================================
    # Action Dispatcher
    # =========================================================================

    def execute_action(
        self,
        action_name: str,
        args: Tuple,
        observations: List[List],
        block_transforms: Optional[Dict[int, np.ndarray]] = None
    ) -> bool:
        """
        Execute a PDDL action.

        Args:
            action_name: Name of action (e.g., 'pick-up', 'put-down')
            args: Action arguments as tuple of string IDs
            observations: Current world state
            block_transforms: Optional dict mapping block_id to 4x4 transform (for grasp orientation)

        Returns:
            True if successful
        """
        # Convert string IDs to integers
        int_args = [int(a) for a in args]

        # Get block transform for pick actions (first arg is block_id)
        block_transform = None
        if block_transforms is not None and len(int_args) > 0:
            block_transform = block_transforms.get(int_args[0])

        action_map = {
            'pick-up': lambda: self.pick_up(*int_args, observations, block_transform),
            'unstack': lambda: self.unstack(*int_args, observations, block_transform),
            'remove-beside': lambda: self.remove_beside(*int_args, observations, block_transform),
            'put-down': lambda: self.put_down(*int_args, observations),
            'align': lambda: self.align(*int_args, observations),
            'cover': lambda: self.cover(*int_args, observations),
            'release': lambda: self.release(*int_args, observations),
        }

        if action_name not in action_map:
            print(f"Unknown action: {action_name}")
            return False

        return action_map[action_name]()
