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
        gripper_open_pos: int = 0,
        gripper_close_pos: int = 255,
        gripper_speed: int = 100,
        gripper_force: int = 50,
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
            gripper_open_pos: Gripper position for open (0-255)
            gripper_close_pos: Gripper position for close (0-255)
            gripper_speed: Gripper speed (0-255)
            gripper_force: Gripper force (0-255)
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
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force

        # Default orientation: gripper pointing down
        # This is a 180 degree rotation around X (gripper Z pointing down)
        self.default_orientation = Rotation.from_rotvec([np.pi, 0, 0]).as_rotvec()

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

        # Move
        target_joints = self.rtde_c.getInverseKinematics(pose, current_joints)
        self.rtde_c.moveJ(target_joints, speed, self.acceleration)
        return True

    def _move_linear(self, pose: List[float], speed: Optional[float] = None) -> bool:
        """Move robot linearly to target pose."""
        if speed is None:
            speed = self.speed * 0.5  # Slower for linear moves

        self.rtde_c.moveL(pose, speed, self.acceleration)
        return True

    def _open_gripper(self):
        """Open the gripper."""
        if self.gripper:
            self.gripper.move(self.gripper_open_pos, self.gripper_speed, self.gripper_force)
            time.sleep(0.5)

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
                time.sleep(1.5)
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
            return pos <= self.gripper_open_pos + 10  # Small margin
        return True

    # =========================================================================
    # High-Level Actions (matching PDDL actions)
    # =========================================================================

    def pick_up(
        self,
        block_id: int,
        table_id: int,
        robot_id: int,
        observations: List[List]
    ) -> bool:
        """
        Pick up a block from the table.

        PDDL action: (pick-up ?b1 ?t1 ?r1)

        Args:
            block_id: ID of block to pick
            table_id: ID of table (unused, for PDDL compatibility)
            robot_id: ID of robot (unused, for PDDL compatibility)
            observations: Current world state

        Returns:
            True if successful
        """
        block = self._get_block_from_obs(block_id, observations)
        if block is None:
            print(f"Block {block_id} not found in observations")
            return False

        # Block position and dimensions
        # Note: z is the marker position = TOP of block (not center)
        x, y, z = block[2], block[3], block[4]
        w, l, h = block[5], block[6], block[7]

        # Grasp point: slightly below block top surface
        # z is already the top of the block, so we just subtract grasp_depth
        # Add gripper_z_offset because TCP is at flange, not fingertips
        grasp_z = z - self.grasp_depth + self.gripper_z_offset

        # Approach point: above the block
        approach_pos = np.array([x, y, grasp_z + self.approach_height])
        grasp_pos = np.array([x, y, grasp_z])

        print(f"Picking block {block_id} at [{x:.3f}, {y:.3f}, {z:.3f}]")

        # Execute pick sequence
        self._open_gripper()

        # Move to approach
        if not self._move_to_pose(self._pose_to_list(approach_pos)):
            return False

        # Move down to grasp
        if not self._move_linear(self._pose_to_list(grasp_pos)):
            return False

        # Grasp
        self._close_gripper()

        # Lift
        if not self._move_linear(self._pose_to_list(approach_pos)):
            return False

        print(f"Picked block {block_id}")
        return True

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
        tw, tl, th = target[5], target[6], target[7]

        # Held block dimensions
        bl = block[6]
        bh = block[7]

        # Place beside: offset in Y direction
        place_x = tx
        place_y = ty + tl / 2 + bl / 2 + 0.01  # Small gap
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
        Release block onto table (emergency drop).

        PDDL action: (release ?b1 ?t1 ?r1)

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

        # Get current position
        current_pos = self.get_gripper_position()

        # Move to safe height above current position
        safe_pos = current_pos.copy()
        safe_pos[2] = 0.3  # Safe height

        print(f"Releasing block {block_id}")

        # Move to safe height
        self._move_to_pose(self._pose_to_list(safe_pos))

        # Release
        self._open_gripper()

        print(f"Released block {block_id}")
        return True

    # =========================================================================
    # Action Dispatcher
    # =========================================================================

    def execute_action(
        self,
        action_name: str,
        args: Tuple,
        observations: List[List]
    ) -> bool:
        """
        Execute a PDDL action.

        Args:
            action_name: Name of action (e.g., 'pick-up', 'put-down')
            args: Action arguments as tuple of string IDs
            observations: Current world state

        Returns:
            True if successful
        """
        # Convert string IDs to integers
        int_args = [int(a) for a in args]

        action_map = {
            'pick-up': lambda: self.pick_up(*int_args, observations),
            'put-down': lambda: self.put_down(*int_args, observations),
            'align': lambda: self.align(*int_args, observations),
            'cover': lambda: self.cover(*int_args, observations),
            'release': lambda: self.release(*int_args, observations),
        }

        if action_name not in action_map:
            print(f"Unknown action: {action_name}")
            return False

        return action_map[action_name]()
