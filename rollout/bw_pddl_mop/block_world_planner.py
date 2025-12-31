#!/usr/bin/env python3
"""
Block World Planner for UR5e.

Integrated perception-planning-execution system for building block structures.

Usage:
    python block_world_planner.py --goal bridge
    python block_world_planner.py --goal tower
"""

import os
import sys
import time
import yaml
import numpy as np
import threading
import select
import termios
import tty
from datetime import datetime
from typing import List, Tuple, Dict, Optional


# =============================================================================
# Logging to file and console
# =============================================================================

class TeeLogger:
    """Duplicates output to both console and log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w')
        self.log_path = log_path

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# =============================================================================
# Keyboard Monitor for Pause/Continue
# =============================================================================

class KeyboardMonitor:
    """Non-blocking keyboard monitor for pause/continue control."""

    def __init__(self):
        self._pause_requested = False
        self._continue_requested = False
        self._running = False
        self._thread = None
        self._old_settings = None

    def start(self):
        """Start keyboard monitoring thread."""
        if self._running:
            return
        self._running = True
        self._pause_requested = False
        self._continue_requested = False
        # Save terminal settings
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
        except:
            self._old_settings = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Press 'p' to pause, 'c' to continue, 'q' to quit")

    def stop(self):
        """Stop keyboard monitoring."""
        self._running = False
        # Restore terminal settings
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background thread to monitor keyboard input."""
        while self._running:
            try:
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    # Set terminal to raw mode temporarily
                    if self._old_settings is not None:
                        tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    # Restore terminal settings immediately
                    if self._old_settings is not None:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

                    if ch.lower() == 'p':
                        self._pause_requested = True
                        print("\n[Keyboard] PAUSE requested - will pause after current action")
                    elif ch.lower() == 'c':
                        self._continue_requested = True
                        print("\n[Keyboard] CONTINUE requested")
                    elif ch.lower() == 'q':
                        print("\n[Keyboard] QUIT requested")
                        self._pause_requested = True  # Use pause to trigger quit
                        self._running = False
            except Exception:
                pass
            time.sleep(0.05)

    def is_pause_requested(self) -> bool:
        """Check if pause was requested."""
        return self._pause_requested

    def is_continue_requested(self) -> bool:
        """Check if continue was requested."""
        return self._continue_requested

    def clear_pause(self):
        """Clear pause request."""
        self._pause_requested = False

    def clear_continue(self):
        """Clear continue request."""
        self._continue_requested = False

    def wait_for_continue(self) -> bool:
        """Wait until continue is pressed. Returns False if quit requested."""
        print("\n" + "="*60)
        print("PAUSED - Press 'c' to continue, 'q' to quit")
        print("="*60)
        self._continue_requested = False
        while self._running and not self._continue_requested:
            time.sleep(0.1)
        return self._running


# Robot interfaces
import rtde_control
import rtde_receive

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aruco_monitor import ArucoMonitor, create_base_camera_monitor, create_wrist_camera_monitor
from predicate_util import get_logical_state, is_goal_satisfied, get_unsatisfied_goals, set_thresholds, apply_action_effects, print_block_positions
from pddl_solver import solve_pddl, plan_to_string
from block_primitives import BlockPrimitives

# Try to import gripper
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'collect'))
    import robotiq_gripper
    HAS_GRIPPER = True
except ImportError:
    HAS_GRIPPER = False
    print("Warning: robotiq_gripper not available")


# =============================================================================
# Goal Definitions
# =============================================================================

GOAL_LIBRARY = {
    # Bridge: two pillars with plank on top
    'bridge': [
        ('on-table', '1', '9'), ('on-table', '2', '9'),
        ('beside', '2', '1'),
        ('above', '5', '1'), ('above', '7', '2'),
        ('above_both', '4', '5', '7'),
    ],

    # Tall bridge: double-height pillars
    'tall_bridge': [
        ('on-table', '1', '9'), ('on-table', '2', '9'),
        ('beside', '2', '1'),
        ('above', '0', '1'), ('above', '3', '2'),
        ('above', '7', '0'), ('above', '5', '3'),
        ('above_both', '4', '7', '5'),
    ],

    # Simple tower: stack of blocks
    'tower': [
        ('on-table', '0', '9'),
        ('above', '1', '0'),
        ('above', '2', '1'),
        ('above', '3', '2'),
    ],

    # CN Tower: tall stack
    'cn_tower': [
        ('on-table', '7', '9'),
        ('above', '1', '7'), ('above', '3', '1'),
        ('above', '5', '3'), ('above', '6', '5'),
        ('above', '0', '6'),
    ],

    # House shape
    'house': [
        ('on-table', '3', '9'), ('on-table', '7', '9'),
        ('beside', '7', '3'),
        ('above', '2', '3'), ('above', '1', '7'),
        ('above', '5', '1'), ('above', '0', '2'),
        ('above_both', '6', '0', '5'),
    ],
}


# =============================================================================
# Main Planner Class
# =============================================================================

class BlockWorldPlanner:
    """
    Integrated block world planning and execution.

    Combines perception (ArUco), planning (PDDL), and execution (UR5e).
    """

    def __init__(self, config: dict):
        """
        Initialize the planner.

        Args:
            config: Configuration dictionary with robot, camera, and block settings
        """
        self.config = config

        # Robot connection
        robot_ip = config['robot']['ip']
        print(f"Connecting to robot at {robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Robot connected.")

        # Gripper
        if HAS_GRIPPER:
            gripper_port = config.get('gripper', {}).get('port', 63352)
            self.gripper = robotiq_gripper.RobotiqGripper()
            self.gripper.connect(robot_ip, gripper_port)
            self.gripper.activate()
            print("Gripper connected.")
        else:
            self.gripper = None

        # ArUco monitor
        self._init_perception(config)

        # Block primitives
        motion_cfg = config.get('motion', {})
        gripper_cfg = config.get('gripper', {})
        self.primitives = BlockPrimitives(
            rtde_c=self.rtde_c,
            rtde_r=self.rtde_r,
            gripper=self.gripper,
            speed=config['robot'].get('speed', 0.3),
            acceleration=config['robot'].get('acceleration', 0.3),
            approach_height=motion_cfg.get('approach_height', 0.10),
            grasp_depth=motion_cfg.get('grasp_depth', 0.02),
            place_clearance=motion_cfg.get('place_clearance', 0.005),
            gripper_z_offset=gripper_cfg.get('z_offset', 0.0),
            action_delay=motion_cfg.get('action_delay', 0.5),
            beside_gap=motion_cfg.get('beside_gap', 0.06),
            linear_speed_factor=motion_cfg.get('linear_speed_factor', 0.5),
            gripper_open_pos=gripper_cfg.get('open_pos', 0),
            gripper_close_pos=gripper_cfg.get('close_pos', 255),
            gripper_speed=gripper_cfg.get('speed', 100),
            gripper_force=gripper_cfg.get('force', 50),
            gripper_open_wait=gripper_cfg.get('open_wait', 0.5),
            gripper_close_wait=gripper_cfg.get('close_wait', 1.5),
            gripper_open_margin=gripper_cfg.get('open_margin', 10),
            release_min_dist=motion_cfg.get('release_min_dist', 0.08),
            table_bounds=config.get('table_bounds', None),
        )

        # Table and robot IDs
        self.table_id = config.get('table_id', 9)
        self.robot_id = config.get('robot_id', 8)
        self.table_z = config.get('table_z', 0.0)

        # Set predicate thresholds from config
        if 'predicates' in config:
            set_thresholds(config['predicates'])

        # Initial pose
        self.init_pose = config['robot'].get('init_pose', None)

    def move_to_init_pose(self, open_gripper: bool = True):
        """Move robot to initial Cartesian pose if configured.

        Pose format: [x, y, z, qx, qy, qz, qw] (meters and quaternion)
        Converts to RTDE format: [x, y, z, rx, ry, rz] (axis-angle)

        Args:
            open_gripper: If True, open gripper after reaching pose.
                          Set to False when holding a block for re-observation.
        """
        if self.init_pose is None:
            print("No init_pose configured, skipping.")
            return

        from scipy.spatial.transform import Rotation

        pose = self.init_pose
        x, y, z = pose[0], pose[1], pose[2]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]

        # Convert quaternion to axis-angle (rotvec)
        rot = Rotation.from_quat([qx, qy, qz, qw])
        rx, ry, rz = rot.as_rotvec()

        rtde_pose = [x, y, z, rx, ry, rz]
        print(f"Moving to init pose: xyz=[{x:.3f}, {y:.3f}, {z:.3f}], quat=[{qx}, {qy}, {qz}, {qw}]")

        speed = self.config['robot'].get('speed', 0.3)
        accel = self.config['robot'].get('acceleration', 0.3)
        self.rtde_c.moveL(rtde_pose, speed, accel)

        # Only open gripper if requested (not when holding a block)
        if open_gripper:
            self.primitives._open_gripper()
        print("Reached init pose.")

    def _init_perception(self, config: dict):
        """Initialize ArUco monitor based on config."""
        camera_cfg = config['camera']
        aruco_cfg = config.get('aruco', {})
        camera_type = camera_cfg.get('type', 'base')
        position_offset = camera_cfg.get('offset', [0.0, 0.0, 0.0])

        # Common ArUco parameters
        aruco_params = dict(
            marker_length=aruco_cfg.get('marker_size', 0.032),
            visual=config.get('visual', False),
            position_offset=position_offset,
            max_reproj_error=aruco_cfg.get('max_reproj_error', 2.0),
            min_sharpness=aruco_cfg.get('min_sharpness', 30.0),
            confident_reproj=aruco_cfg.get('confident_reproj', 1.0),
            confident_sharpness=aruco_cfg.get('confident_sharpness', 50.0),
            stale_age=aruco_cfg.get('stale_age', 2.0),
        )

        if camera_type == 'base':
            # Base-mounted camera with fixed transform
            T_cam_base = np.array(camera_cfg['T_cam_base'])
            self.aruco_monitor = create_base_camera_monitor(
                camera_serial=camera_cfg['serial'],
                T_cam_base=T_cam_base,
                **aruco_params,
            )
        elif camera_type == 'wrist':
            # Wrist-mounted camera with hand-eye calibration
            X_ee_cam = np.load(camera_cfg['calibration_file'])
            self.aruco_monitor = create_wrist_camera_monitor(
                camera_serial=camera_cfg['serial'],
                X_ee_cam=X_ee_cam,
                rtde_r=self.rtde_r,
                **aruco_params,
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        self.aruco_monitor.start()
        print(f"Perception initialized ({camera_type} camera)")

    def get_observations(self, print_status: bool = True) -> Tuple[List[List], List[int], List[int]]:
        """Get current world state from perception.

        Returns:
            Tuple of (observations list, stale marker IDs, uncertain marker IDs)
        """
        gripper_pos = self.primitives.get_gripper_position()
        gripper_open = self.primitives.is_gripper_open()

        return self.aruco_monitor.get_block_observations(
            table_id=self.table_id,
            table_z=self.table_z,
            robot_id=self.robot_id,
            gripper_pos=gripper_pos,
            gripper_open=gripper_open,
            print_status=print_status,
        )

    def get_logical_state(self, print_status: bool = True, debug: bool = False) -> List[Tuple]:
        """Get current logical state from perception.

        Args:
            print_status: Print ArUco detection status
            debug: Print detailed debug info for above detection
        """
        observations, stale_ids, uncertain_ids = self.get_observations(print_status=print_status)
        return get_logical_state(observations, debug=debug)

    def reobserve_uncertain_blocks(self, uncertain_ids: List[int], hover_height: Optional[float] = None) -> bool:
        """
        Re-observe uncertain blocks by moving robot above each one.

        For wrist camera, moving directly above a block gives better view.

        Args:
            uncertain_ids: List of marker IDs with uncertain detection
            hover_height: Height above block to hover (meters). Defaults to config.

        Returns:
            True if any blocks were re-observed successfully
        """
        if not uncertain_ids:
            return False

        # Get config values
        motion_cfg = self.config.get('motion', {})
        if hover_height is None:
            hover_height = motion_cfg.get('hover_height', 0.15)
        observation_wait = motion_cfg.get('observation_wait', 0.5)

        print(f"\n[Re-observation] Moving to observe uncertain blocks: {uncertain_ids}")
        reobserved = False

        for marker_id in uncertain_ids:
            # Get current (uncertain) position
            pos = self.aruco_monitor.get_marker_position(marker_id)
            if pos is None:
                print(f"  [Re-observation] Cannot find position for marker {marker_id}")
                continue

            # Move above the block
            hover_pos = [pos[0], pos[1], pos[2] + hover_height]
            print(f"  [Re-observation] Moving above block {marker_id} at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            # Create pose with default orientation (gripper pointing down)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_rotvec([np.pi, 0, 0]).as_rotvec()
            pose = hover_pos + list(rot)

            # Move to hover position
            speed = self.config['robot'].get('speed', 0.3)
            accel = self.config['robot'].get('acceleration', 0.3)
            self.rtde_c.moveL(pose, speed, accel)

            # Wait for detection to update
            time.sleep(observation_wait)

            # Check if detection improved
            conf = self.aruco_monitor.get_confidence(marker_id)
            if conf:
                print(f"  [Re-observation] Block {marker_id} after hover: reproj_err={conf['reproj_error']:.2f}px, sharpness={conf['sharpness']:.1f}, confident={conf['confident']}")
                if conf['confident']:
                    reobserved = True

        # Return to init pose after re-observation
        # Keep gripper state (don't open if holding a block)
        holding_block = not self.primitives.is_gripper_open()
        print(f"  [Re-observation] Returning to init pose... (holding={holding_block})")
        self.move_to_init_pose(open_gripper=not holding_block)
        time.sleep(observation_wait)

        return reobserved

    def emergency_release(self) -> bool:
        """
        Emergency release: if robot is holding a block, release it to an empty spot on table.

        Used when pausing to ensure the robot can safely return to init pose.

        Returns:
            True if block was released or gripper was already open
        """
        if self.primitives.is_gripper_open():
            print("[Emergency Release] Gripper already open, nothing to release")
            return True

        print("\n[Emergency Release] Robot holding a block, releasing to table...")

        # Move to init pose first to get fresh observations
        self.move_to_init_pose(open_gripper=False)
        time.sleep(1.0)

        # Get current observations
        observations, _, _ = self.get_observations(print_status=True)

        # Find which block we're holding
        # The held block should be close to gripper in 3D space (it's in the gripper)
        # Table blocks will be much lower in Z
        gripper_pos = self.primitives.get_gripper_position()
        held_block_id = None
        min_dist = float('inf')

        for obs in observations:
            if obs[1] in [2, 3]:  # Cube or plank
                bx, by, bz = obs[2], obs[3], obs[4]
                # Use 3D distance to distinguish held block from table blocks
                dist = np.sqrt((bx - gripper_pos[0])**2 +
                              (by - gripper_pos[1])**2 +
                              (bz - gripper_pos[2])**2)
                if dist < min_dist:
                    min_dist = dist
                    held_block_id = obs[0]

        # If the closest block is too far (> 15cm), it's probably not the held block
        # (held block might not be detected - it's in the gripper)
        if held_block_id is None or min_dist > 0.15:
            print(f"[Emergency Release] Warning: Cannot identify held block (min_dist={min_dist:.3f})")
            print("[Emergency Release] Releasing at safe position on table...")
            # Release at a safe position - use simple put-down at init position offset
            self._emergency_drop_at_safe_position()
            return True

        print(f"[Emergency Release] Identified held block {held_block_id} (dist={min_dist:.3f}m)")

        # Create a modified observation with the held block at gripper position
        # This ensures release() can find the block
        modified_obs = []
        for obs in observations:
            if obs[0] == held_block_id:
                # Update held block position to gripper position
                modified = list(obs)
                modified[2] = gripper_pos[0]
                modified[3] = gripper_pos[1]
                modified[4] = gripper_pos[2]
                modified_obs.append(modified)
            else:
                modified_obs.append(obs)

        # Use the release action with modified observations
        success = self.primitives.release(
            block_id=held_block_id,
            table_id=self.table_id,
            robot_id=self.robot_id,
            observations=modified_obs
        )

        if success:
            print("[Emergency Release] Block released successfully")
            self.move_to_init_pose()
            time.sleep(1.0)
        else:
            print("[Emergency Release] Release failed, dropping at safe position")
            self._emergency_drop_at_safe_position()

        return True

    def _emergency_drop_at_safe_position(self):
        """Drop held block at a safe position on the table."""
        # Get table bounds from config
        tb = self.config.get('table_bounds', {'x_min': -0.3, 'x_max': 0.3, 'y_min': 0.3, 'y_max': 0.7})
        table_z = self.config.get('table_z', 0.0)

        # Calculate safe drop position - center of table
        drop_x = (tb['x_min'] + tb['x_max']) / 2
        drop_y = (tb['y_min'] + tb['y_max']) / 2
        drop_z = table_z + 0.15  # 15cm above table

        print(f"[Emergency Drop] Moving to [{drop_x:.3f}, {drop_y:.3f}, {drop_z:.3f}]")

        # Move above drop position
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_rotvec([np.pi, 0, 0]).as_rotvec()
        pose = [drop_x, drop_y, drop_z] + list(rot)

        speed = self.config['robot'].get('speed', 0.3)
        accel = self.config['robot'].get('acceleration', 0.3)
        self.rtde_c.moveL(pose, speed, accel)

        # Lower to just above table
        pose[2] = table_z + 0.05
        self.rtde_c.moveL(pose, speed * 0.5, accel)

        # Open gripper
        self.primitives._open_gripper()

        # Move back up
        pose[2] = drop_z
        self.rtde_c.moveL(pose, speed, accel)

        # Return to init
        self.move_to_init_pose()

    def plan(self, goal: List[Tuple]) -> Optional[List[Tuple]]:
        """
        Generate a plan to achieve the goal.

        Args:
            goal: List of goal predicates

        Returns:
            List of actions or None if no plan found
        """
        logical_state = self.get_logical_state()

        print("\nCurrent state:")
        for pred in sorted(logical_state, key=lambda x: x[0]):
            print(f"  {pred}")

        print("\nGoal:")
        for pred in goal:
            print(f"  {pred}")

        # Check if already satisfied
        if is_goal_satisfied(goal, logical_state):
            print("\nGoal already satisfied!")
            return []

        print("\nPlanning...")
        plan = solve_pddl(logical_state, goal, debug=False)

        if plan:
            print("\nPlan found:")
            print(plan_to_string(plan))
        else:
            print("\nNo plan found!")

        return plan

    def execute_plan(self, plan: List[Tuple]) -> bool:
        """
        Execute a plan step by step.

        Args:
            plan: List of actions from planner

        Returns:
            True if all actions succeeded
        """
        if not plan:
            return True

        for i, (action_name, args) in enumerate(plan):
            print(f"\n{'='*50}")
            print(f"Step {i+1}/{len(plan)}: {action_name}({', '.join(args)})")
            print('='*50)

            # Get fresh observations and logical state
            logical_state = self.get_logical_state(print_status=True)
            observations, _, _ = self.get_observations(print_status=False)

            print(f"\nCurrent logical state ({len(logical_state)} predicates):")
            for pred in sorted(logical_state, key=lambda x: x[0]):
                print(f"  {pred}")

            # Execute action
            success = self.primitives.execute_action(action_name, args, observations)

            if not success:
                print(f"Action failed: {action_name}")
                return False

            # Only return to init pose after PLACE actions (when hand is empty)
            # After pick-up, robot is holding a block - don't open gripper or move away!
            if action_name != 'pick-up':
                print("Returning to init pose for perception update...")
                self.move_to_init_pose()
                time.sleep(1.0)  # Wait for perception to update

                # Get updated observations and logical state
                logical_state = self.get_logical_state(print_status=True)
                print(f"\nUpdated logical state after action ({len(logical_state)} predicates):")
                for pred in sorted(logical_state, key=lambda x: x[0]):
                    print(f"  {pred}")

        return True

    def run(self, goal: List[Tuple], max_replans: int = 100) -> bool:
        """
        Run the full perception-planning-execution loop.

        Strategy:
        - After pick-up/unstack: Apply PDDL effects symbolically, then replan
          (since robot arm blocks wrist camera when holding a block)
        - After place actions (align, put-down, cover, release): Re-observe
          from perception, update state, then replan

        Pause/Continue:
        - Press 'p' to pause after current action
        - When paused: if holding a block, release it first
        - Press 'c' to continue (replans from current state)
        - Press 'q' to quit

        Args:
            goal: List of goal predicates
            max_replans: Maximum number of replanning attempts

        Returns:
            True if goal achieved
        """
        print("\n" + "="*60)
        print("BLOCK WORLD PLANNER")
        print("="*60)

        # Start keyboard monitor for pause/continue
        keyboard = KeyboardMonitor()
        keyboard.start()

        try:
            return self._run_loop(goal, max_replans, keyboard)
        finally:
            keyboard.stop()

    def _run_loop(self, goal: List[Tuple], max_replans: int, keyboard: KeyboardMonitor) -> bool:
        """Internal run loop with pause/continue support."""
        # Move to initial pose first
        self.move_to_init_pose()

        # Wait for camera to stabilize and detect markers
        print("Waiting for perception to stabilize...")
        time.sleep(2.0)

        # Track logical state (can be from perception or symbolic updates)
        logical_state = None
        use_symbolic_state = False  # True after pick actions

        for attempt in range(max_replans):
            # Check for pause request at start of each cycle
            if keyboard.is_pause_requested():
                keyboard.clear_pause()
                print("\n[PAUSE] Pause requested at start of planning cycle")

                # If holding a block, release it first
                self.emergency_release()

                # Wait for continue
                if not keyboard.wait_for_continue():
                    print("\n[QUIT] User requested quit")
                    return False

                keyboard.clear_continue()
                print("\n[CONTINUE] Resuming - will replan from current state")
                use_symbolic_state = False  # Force fresh observation
                # Don't fall through - restart loop to get fresh observations

            print(f"\n{'='*60}")
            print(f"PLANNING CYCLE {attempt + 1}/{max_replans}")
            print('='*60)

            # Get state: either from perception or use symbolically updated state
            if use_symbolic_state and logical_state is not None:
                print("[Using symbolically updated state after pick action]")
                use_symbolic_state = False  # Reset for next cycle
            else:
                # Get fresh state from perception
                # Enable debug on first few cycles to help diagnose issues
                debug_mode = (attempt < 3) and self.config.get('debug_predicates', False)
                logical_state = self.get_logical_state(print_status=True, debug=debug_mode)

            num_blocks = sum(1 for p in logical_state if p[0] == 'box')

            if num_blocks == 0:
                print(f"No blocks detected! Waiting 5s for markers to appear...")
                time.sleep(5.0)
                continue

            print(f"\nDetected {num_blocks} blocks")
            print(f"Current logical state ({len(logical_state)} predicates):")
            for pred in sorted(logical_state, key=lambda x: x[0]):
                print(f"  {pred}")

            # Check if goal already satisfied
            if is_goal_satisfied(goal, logical_state):
                print("\n" + "="*60)
                print("GOAL ACHIEVED!")
                print("="*60)
                return True

            # Show unsatisfied goals
            unsatisfied = get_unsatisfied_goals(goal, logical_state)
            print(f"\nUnsatisfied goals ({len(unsatisfied)}):")
            for g in unsatisfied:
                print(f"  {g}")

            # Generate plan for current state
            print("\nPlanning...")
            plan = solve_pddl(logical_state, goal, debug=False)

            if plan is None:
                print("Planning failed! Waiting 5s for perception update...")
                use_symbolic_state = False  # Force re-observation
                time.sleep(5.0)
                continue

            if len(plan) == 0:
                print("\n" + "="*60)
                print("GOAL ACHIEVED!")
                print("="*60)
                return True

            print(f"\nPlan ({len(plan)} steps):")
            print(plan_to_string(plan))

            # Execute only the FIRST action of the plan
            action_name, args = plan[0]
            print(f"\n{'='*50}")
            print(f"EXECUTING: {action_name}({', '.join(args)})")
            print('='*50)

            # For PLACE actions: move to init pose first to get fresh target observations
            # This is especially important after a pick action when robot is holding a block
            place_actions = ['align', 'put-down', 'cover', 'release']
            if action_name in place_actions:
                # Check if robot is holding a block (gripper closed)
                if not self.primitives.is_gripper_open():
                    print("\n[Pre-place observation] Robot holding block, moving to init for fresh observations...")
                    self.move_to_init_pose(open_gripper=False)  # Keep gripper closed!
                    time.sleep(1.0)

            # Get observations for action execution
            observations, stale_ids, uncertain_ids = self.get_observations(print_status=False)

            # Re-observe uncertain blocks before executing action
            if uncertain_ids:
                print(f"\nUncertain detections for blocks: {uncertain_ids}")
                self.reobserve_uncertain_blocks(uncertain_ids)
                # Get fresh observations after re-observation
                observations, stale_ids, uncertain_ids = self.get_observations(print_status=False)

            # Log all object locations before action
            print_block_positions(observations)
            if stale_ids:
                print(f"WARNING: Stale markers (not seen recently): {stale_ids}")
            if uncertain_ids:
                print(f"WARNING: Still uncertain after re-observation: {uncertain_ids}")

            # Log robot TCP position
            tcp = self.rtde_r.getActualTCPPose()
            print(f"Robot TCP: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]")
            gripper_open = self.primitives.is_gripper_open()
            print(f"Gripper: {'OPEN' if gripper_open else 'CLOSED'}")

            # Execute action
            success = self.primitives.execute_action(action_name, args, observations)

            if not success:
                print(f"Action failed: {action_name}")
                print("Will replan on next cycle...")
                use_symbolic_state = False  # Force re-observation
                time.sleep(1.0)
                continue

            # Check for pause request after action execution
            if keyboard.is_pause_requested():
                keyboard.clear_pause()
                print("\n[PAUSE] Pause requested after action execution")

                # If holding a block, release it first
                self.emergency_release()

                # Wait for continue
                if not keyboard.wait_for_continue():
                    print("\n[QUIT] User requested quit")
                    return False

                keyboard.clear_continue()
                print("\n[CONTINUE] Resuming - will replan from current state")
                use_symbolic_state = False  # Force fresh observation
                continue  # Go to next planning cycle

            # Handle state update based on action type
            if action_name in ['pick-up', 'unstack', 'remove-beside']:
                # Apply PDDL effects symbolically (camera blocked by robot arm)
                print(f"\n[Applying {action_name} effects symbolically]")
                logical_state = apply_action_effects(logical_state, action_name, args)
                use_symbolic_state = True

                print(f"Updated logical state after {action_name} ({len(logical_state)} predicates):")
                for pred in sorted(logical_state, key=lambda x: x[0]):
                    print(f"  {pred}")

                # Continue immediately to next planning cycle (will use symbolic state)
            else:
                # Place actions: return to init pose and re-observe
                print("\nReturning to init pose for perception update...")
                self.move_to_init_pose()
                time.sleep(1.0)

                # Get fresh observations and state from perception
                observations, stale_ids, uncertain_ids = self.get_observations(print_status=True)

                # Re-observe uncertain blocks after place action
                if uncertain_ids:
                    print(f"\nUncertain detections after place: {uncertain_ids}")
                    self.reobserve_uncertain_blocks(uncertain_ids)
                    observations, stale_ids, uncertain_ids = self.get_observations(print_status=True)

                print_block_positions(observations)
                if stale_ids:
                    print(f"WARNING: Stale markers (not seen recently): {stale_ids}")
                if uncertain_ids:
                    print(f"WARNING: Still uncertain after re-observation: {uncertain_ids}")

                logical_state = get_logical_state(observations)
                use_symbolic_state = False

                print(f"\nUpdated logical state from perception ({len(logical_state)} predicates):")
                for pred in sorted(logical_state, key=lambda x: x[0]):
                    print(f"  {pred}")

        print("\nMax replanning attempts exceeded!")
        return False

    def close(self):
        """Clean up resources."""
        self.aruco_monitor.stop()
        self.rtde_c.stopScript()
        print("Planner closed.")


# =============================================================================
# Configuration
# =============================================================================

def create_default_config(
    robot_ip: str = "192.168.56.101",
    camera_serial: str = "337322072205",  # Base camera
    camera_type: str = "base",
) -> dict:
    """Create default configuration."""
    config = {
        'robot': {
            'ip': robot_ip,
            'speed': 0.3,
            'acceleration': 0.3,
        },
        'gripper': {
            'port': 63352,
        },
        'camera': {
            'serial': camera_serial,
            'type': camera_type,
        },
        'aruco': {
            'marker_size': 0.032,
        },
        'motion': {
            'approach_height': 0.10,
            'grasp_depth': 0.02,
        },
        'table_id': 9,
        'robot_id': 8,
        'table_z': 0.0,
        'visual': False,
    }

    # Add camera-specific config
    if camera_type == "base":
        # Default base camera transform (update with your calibration!)
        config['camera']['T_cam_base'] = [
            [0.0761, 0.8458, -0.5280, 1.038],
            [0.9950, -0.0305, 0.0946, 0.009],
            [0.0640, -0.5326, -0.8439, 0.416],
            [0.0, 0.0, 0.0, 1.0]
        ]
    elif camera_type == "wrist":
        config['camera']['calibration_file'] = "handeye_data/calibration_latest.npy"

    return config


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Block World Planner for UR5e")
    parser.add_argument('--config', type=str, default=None, help='Config YAML file')
    parser.add_argument('--goal', type=str, default='bridge',
                        choices=list(GOAL_LIBRARY.keys()),
                        help='Goal structure to build')
    parser.add_argument('--robot-ip', type=str, default='192.168.56.101',
                        help='Robot IP address')
    parser.add_argument('--camera-serial', type=str, default='337322072205',
                        help='RealSense camera serial')
    parser.add_argument('--camera-type', type=str, default='base',
                        choices=['base', 'wrist'],
                        help='Camera mounting type')
    parser.add_argument('--visual', action='store_true',
                        help='Show ArUco detection visualization')
    parser.add_argument('--max-replans', type=int, default=100,
                        help='Maximum replanning attempts')

    args = parser.parse_args()

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = create_default_config(
            robot_ip=args.robot_ip,
            camera_serial=args.camera_serial,
            camera_type=args.camera_type,
        )

    config['visual'] = args.visual

    # Set up logging to file
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_{args.goal}_{timestamp}.log')
    logger = TeeLogger(log_path)
    sys.stdout = logger

    # Get goal
    goal = GOAL_LIBRARY[args.goal]

    print(f"Logging to: {log_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Goal: {args.goal}")
    print(f"Config: {args.config or 'default'}")

    # Run planner
    planner = BlockWorldPlanner(config)

    try:
        success = planner.run(goal, max_replans=args.max_replans)
        if success:
            print("\nTask completed successfully!")
        else:
            print("\nTask failed.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        planner.close()
        print(f"\nLog saved to: {log_path}")
        sys.stdout = logger.terminal  # Restore stdout
        logger.close()


if __name__ == '__main__':
    main()
