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
from datetime import datetime
from typing import List, Tuple, Optional

# Terminal handling for keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


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
# Pause Controller - Terminal keyboard input
# =============================================================================

class PauseController:
    """
    Pause/continue controller using terminal keyboard input.

    Press 'p' to pause (will pause after current action)
    Press 'c' to continue
    Press 'q' to quit
    """

    def __init__(self):
        self._pause_requested = False
        self._quit_requested = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._old_settings = None
        self._is_tty = False

    def start(self):
        """Start keyboard monitoring."""
        self._running = True
        self._pause_requested = False
        self._quit_requested = False

        # Check if stdin is a TTY
        try:
            self._is_tty = sys.stdin.isatty() and HAS_TERMIOS
        except Exception:
            self._is_tty = False

        if self._is_tty:
            try:
                # Save current terminal settings
                self._old_settings = termios.tcgetattr(sys.stdin)
                # Start keyboard monitoring thread
                self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
                self._thread.start()
                print("\n" + "="*60)
                print("KEYBOARD CONTROLS (press in this terminal):")
                print("  'p' = PAUSE after current action")
                print("  'c' = CONTINUE after pause")
                print("  'q' = QUIT")
                print("="*60 + "\n")
            except Exception as e:
                print(f"[Warning] Could not initialize keyboard input: {e}")
                self._is_tty = False

        if not self._is_tty:
            print("\n" + "="*60)
            print("KEYBOARD INPUT NOT AVAILABLE")
            print("Use file-based controls instead:")
            print("  To PAUSE:    touch /tmp/robot_pause")
            print("  To CONTINUE: rm /tmp/robot_pause")
            print("  To QUIT:     touch /tmp/robot_quit")
            print("="*60 + "\n")

    def stop(self):
        """Stop keyboard monitoring."""
        self._running = False

        # Restore terminal settings
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _keyboard_loop(self):
        """Background thread for keyboard monitoring."""
        fd = sys.stdin.fileno()

        while self._running:
            try:
                # Use select to check for input with timeout (non-blocking)
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)

                if readable:
                    # Temporarily set terminal to raw mode to read single char
                    tty.setcbreak(fd)  # Use cbreak instead of raw - less invasive
                    try:
                        ch = sys.stdin.read(1)
                    finally:
                        # Restore terminal settings immediately
                        if self._old_settings:
                            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

                    with self._lock:
                        if ch.lower() == 'p':
                            self._pause_requested = True
                            sys.stdout.write("\n>>> PAUSE REQUESTED - will pause after current action\n")
                            sys.stdout.flush()
                        elif ch.lower() == 'c':
                            self._pause_requested = False  # Clear pause = continue
                            # Also remove pause file if it exists
                            if os.path.exists("/tmp/robot_pause"):
                                try:
                                    os.remove("/tmp/robot_pause")
                                except Exception:
                                    pass
                            sys.stdout.write("\n>>> CONTINUE\n")
                            sys.stdout.flush()
                        elif ch.lower() == 'q':
                            self._quit_requested = True
                            sys.stdout.write("\n>>> QUIT REQUESTED\n")
                            sys.stdout.flush()

            except Exception:
                # Silently continue on errors
                pass

            time.sleep(0.05)

    def is_pause_requested(self) -> bool:
        """Check if pause was requested."""
        with self._lock:
            # Also check file-based pause as fallback
            return self._pause_requested or os.path.exists("/tmp/robot_pause")

    def is_quit_requested(self) -> bool:
        """Check if quit was requested."""
        with self._lock:
            return self._quit_requested or os.path.exists("/tmp/robot_quit")

    def clear_pause(self):
        """Clear pause request."""
        with self._lock:
            self._pause_requested = False
        # Also remove file-based pause
        if os.path.exists("/tmp/robot_pause"):
            try:
                os.remove("/tmp/robot_pause")
            except Exception:
                pass

    def wait_for_continue(self) -> bool:
        """Wait until continue or quit. Returns False if quit."""
        print("\n" + "="*60)
        print("PAUSED")
        if self._is_tty:
            print("  Press 'c' to CONTINUE (or rm /tmp/robot_pause)")
            print("  Press 'q' to QUIT (or touch /tmp/robot_quit)")
        else:
            print("  To CONTINUE: rm /tmp/robot_pause")
            print("  To QUIT:     touch /tmp/robot_quit")
        print("="*60)

        # Snapshot what caused the pause
        with self._lock:
            was_keyboard_pause = self._pause_requested
        was_file_pause = os.path.exists("/tmp/robot_pause")

        while self._running:
            if self.is_quit_requested():
                return False

            # Check if the pause source(s) have been cleared
            # - keyboard_cleared: 'c' was pressed (clears _pause_requested)
            # - file_cleared: file was removed (by 'c' press or manually)
            with self._lock:
                keyboard_cleared = was_keyboard_pause and not self._pause_requested
            file_cleared = was_file_pause and not os.path.exists("/tmp/robot_pause")

            # Continue if ANY pause source that was active is now cleared
            # Note: When 'c' is pressed, _keyboard_loop also removes the pause file,
            # so file_cleared handles the case of pressing 'c' for file-only pause
            if keyboard_cleared or file_cleared:
                # Clear all remaining pause state
                self.clear_pause()
                return True

            time.sleep(0.2)

        return False


# Robot interfaces
import rtde_control
import rtde_receive

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aruco_monitor import create_base_camera_monitor, create_wrist_camera_monitor
from predicate_util import get_logical_state, is_goal_satisfied, get_unsatisfied_goals, set_thresholds, print_block_positions
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
        ('above_both', '6', '5', '7'),
    ],

    # Tall bridge: double-height pillars
    'tall_bridge': [
        ('on-table', '1', '9'), ('on-table', '2', '9'),
        ('beside', '2', '1'),
        ('above', '0', '1'), ('above', '3', '2'),
        ('above', '7', '0'), ('above', '5', '3'),
        ('above_both', '6', '7', '5'),
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
            linear_acceleration=motion_cfg.get('linear_acceleration', 0.1),
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

        # Set grasp orientations from config
        if 'grasp_orientations' in config:
            # Convert keys from strings to ints if needed (YAML may load as strings)
            grasp_orientations = {}
            for k, v in config['grasp_orientations'].items():
                grasp_orientations[int(k)] = v
            self.primitives.set_grasp_orientations(grasp_orientations)
            print(f"Loaded grasp orientations for {len(grasp_orientations)} blocks")

        # Set joint limits from config
        if 'joint_limits' in config.get('robot', {}):
            joint_limits = {}
            for k, v in config['robot']['joint_limits'].items():
                joint_limits[int(k)] = v
            self.primitives.set_joint_limits(joint_limits)
            print(f"Loaded joint limits for {len(joint_limits)} joints")

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
            self.primitives.open_gripper()
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

        Note: This method discards stale/uncertain info. Use observe_world()
        for synchronized observations with quality info.

        Args:
            print_status: Print ArUco detection status
            debug: Print detailed debug info for above detection
        """
        observations, _, _ = self.get_observations(print_status=print_status)
        return get_logical_state(observations, debug=debug)

    def observe_world(
        self,
        extra_viewpoints: bool = False,
        stability_wait: Optional[float] = None,
        print_status: bool = True,
    ) -> Tuple[List[List], List[Tuple], List[int], List[int]]:
        """
        Move robot to observation positions and get synchronized state.

        This ensures observations and logical state are always captured together
        when the robot is stationary, eliminating timing mismatches.

        Args:
            extra_viewpoints: If True, visit additional positions around init
                             for better coverage (±10cm offsets)
            stability_wait: Time to wait for camera stabilization (seconds).
                           Defaults to config value or 0.5s.
            print_status: Print ArUco detection status

        Returns:
            Tuple of (observations, logical_state, stale_ids, uncertain_ids)
        """
        motion_cfg = self.config.get('motion', {})
        if stability_wait is None:
            stability_wait = motion_cfg.get('stability_wait', 0.5)

        # Determine if we're holding a block (keep gripper closed if so)
        holding_block = not self.primitives.is_gripper_open()

        # Move to init pose first
        self.move_to_init_pose(open_gripper=not holding_block)

        # Optionally visit extra viewpoints for better coverage
        if extra_viewpoints:
            viewpoints = motion_cfg.get('observe_viewpoints', [
                [0.10, 0.0, 0.0],   # Right (+10cm)
                [-0.10, 0.0, 0.0],  # Left (-10cm)
                [0.0, 0.10, 0.0],   # Forward (+10cm)
                [0.0, -0.10, 0.0],  # Back (-10cm)
            ])

            if self.init_pose is not None:
                base_pos = self.init_pose[:3]  # [x, y, z]
                for offset in viewpoints:
                    viewpoint_pos = [
                        base_pos[0] + offset[0],
                        base_pos[1] + offset[1],
                        base_pos[2] + offset[2],
                    ]
                    self.primitives.move_to_cartesian(viewpoint_pos, linear=True)
                    time.sleep(stability_wait * 0.5)  # Brief pause at each viewpoint

                # Return to init pose
                self.move_to_init_pose(open_gripper=not holding_block)

        # Wait for camera to stabilize
        time.sleep(stability_wait)

        # Get synchronized observations and logical state
        observations, stale_ids, uncertain_ids = self.get_observations(print_status=print_status)
        logical_state = get_logical_state(observations)

        return observations, logical_state, stale_ids, uncertain_ids

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

            # Move to hover position using primitives
            self.primitives.move_to_cartesian(hover_pos, linear=True)

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
        self.primitives.move_to_cartesian([drop_x, drop_y, drop_z], linear=True)

        # Lower to just above table
        self.primitives.move_to_cartesian([drop_x, drop_y, table_z + 0.05], linear=True)

        # Open gripper
        self.primitives.open_gripper()

        # Move back up
        self.primitives.move_to_cartesian([drop_x, drop_y, drop_z], linear=True)

        # Return to init
        self.move_to_init_pose()

    def _execute_pick_and_observe(
        self,
        action_name: str,
        args: Tuple,
        observations: List[List],
    ) -> Tuple[bool, List[List], List[Tuple]]:
        """
        Execute a pick action and observe world afterwards.

        After the pick motion, checks if the gripper actually grasped something:
        - If holding: observe world and ensure holding predicate is set
        - If not holding: pick failed, just observe world for fresh state

        Args:
            action_name: Name of pick action ('pick-up', 'unstack', 'remove-beside')
            args: Action arguments as tuple of string IDs
            observations: Current observations for action execution

        Returns:
            Tuple of (success, new_observations, new_logical_state)
        """
        # Get block transforms for grasp orientation
        block_transforms = self.aruco_monitor.get_block_transforms()

        # Execute the pick action with transforms
        success = self.primitives.execute_action(action_name, args, observations, block_transforms)

        if not success:
            print(f"[Pick] Action {action_name} failed during execution")
            # Observe world to get fresh state after failed attempt
            new_obs, new_state, _, _ = self.observe_world(extra_viewpoints=True)
            return False, new_obs, new_state

        # Check if we actually grasped something
        time.sleep(0.3)  # Brief wait for gripper to settle
        holding = not self.primitives.is_gripper_open()

        if holding:
            print(f"[Pick] Successfully grasped block (gripper closed)")

            # Return to init pose (keeping block) and observe world
            new_obs, fresh_state, _, _ = self.observe_world(
                extra_viewpoints=True,
                print_status=True
            )

            # Merge: keep (holding, block_id, robot_id) from symbolic state,
            # but use fresh observations for other predicates
            # The held block won't be visible (occluded by gripper), so we preserve the holding predicate
            held_block_id = args[0]
            holding_pred = ('holding', held_block_id, str(self.robot_id))

            # Start with fresh state, ensure holding predicate is present
            if holding_pred not in fresh_state:
                fresh_state = list(fresh_state)
                fresh_state.append(holding_pred)
                # Also remove hand_free if present
                hand_free_pred = ('hand_free', str(self.robot_id))
                if hand_free_pred in fresh_state:
                    fresh_state.remove(hand_free_pred)

            return True, new_obs, fresh_state
        else:
            print(f"[Pick] Gripper is open - pick failed (nothing grasped)")
            # Pick failed - get fresh state
            new_obs, new_state, _, _ = self.observe_world(extra_viewpoints=True)
            return False, new_obs, new_state

    def find_free_space(
        self,
        observations: List[List],
        block_width: float,
        block_length: float,
        safety_margin: float = 0.03,
    ) -> Optional[Tuple[float, float]]:
        """
        Find a free spot on the table for placing a block.

        Uses spiral search pattern from the center of existing blocks.

        Args:
            observations: Current world state
            block_width: Width of block to place
            block_length: Length of block to place
            safety_margin: Extra clearance between blocks (meters)

        Returns:
            (x, y) coordinates for placement, or None if no space found
        """
        # Get table bounds from config
        tb = self.config.get('table_bounds', {
            'x_min': -0.3, 'x_max': 0.3,
            'y_min': 0.3, 'y_max': 0.7
        })

        # Collect positions and sizes of all blocks on table
        other_blocks = []
        for obs in observations:
            if obs[1] in [2, 3]:  # Cubes and planks
                other_blocks.append({
                    'x': obs[2], 'y': obs[3],
                    'width': obs[5], 'length': obs[6]
                })

        # Calculate cluster center
        if other_blocks:
            avg_x = sum(b['x'] for b in other_blocks) / len(other_blocks)
            avg_y = sum(b['y'] for b in other_blocks) / len(other_blocks)
        else:
            avg_x = (tb['x_min'] + tb['x_max']) / 2
            avg_y = (tb['y_min'] + tb['y_max']) / 2

        # Spiral search outward from cluster center
        # Search at increasing radii, 8 directions each
        min_radius = 0.08
        max_radius = 0.32
        step = 0.04
        num_directions = 8

        for r in np.arange(min_radius, max_radius + step, step):
            for dir_idx in range(num_directions):
                angle = dir_idx * (2 * np.pi / num_directions)
                dx = r * np.cos(angle)
                dy = r * np.sin(angle)
                cx, cy = avg_x + dx, avg_y + dy

                # Check within table bounds (with margin for block size)
                margin_x = block_width / 2 + 0.01
                margin_y = block_length / 2 + 0.01
                if not (tb['x_min'] + margin_x < cx < tb['x_max'] - margin_x and
                        tb['y_min'] + margin_y < cy < tb['y_max'] - margin_y):
                    continue

                # Check AABB collision with all other blocks
                collision = False
                for other in other_blocks:
                    min_dist_x = (block_width + other['width']) / 2 + safety_margin
                    min_dist_y = (block_length + other['length']) / 2 + safety_margin
                    if abs(cx - other['x']) < min_dist_x and abs(cy - other['y']) < min_dist_y:
                        collision = True
                        break

                if not collision:
                    return (cx, cy)

        # No free space found
        print("[find_free_space] Warning: No free space found on table")
        return None

    def run(self, goal: List[Tuple], max_replans: int = 100) -> bool:
        """
        Run the full perception-planning-execution loop.

        New Architecture:
        - Uses observe_world() to ensure observations and logical state are
          always synchronized (captured together when robot is stationary)
        - After pick actions: checks gripper to verify grasp, then observe_world()
        - After place actions: observe_world() to get fresh state
        - No more timing mismatches between planning and execution

        Pause/Continue:
        - Press 'p' to pause after current action
        - When paused: if holding a block, release it first
        - Press 'c' to continue (observe_world then replan)
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

        # Start pause controller
        pause_ctrl = PauseController()
        pause_ctrl.start()

        try:
            return self._run_loop(goal, max_replans, pause_ctrl)
        finally:
            pause_ctrl.stop()

    def _run_loop(self, goal: List[Tuple], max_replans: int, pause_ctrl: PauseController) -> bool:
        """Internal run loop with pause/continue support."""
        # Initial observation
        observations, logical_state, stale_ids, uncertain_ids = self.observe_world(
            extra_viewpoints=True, print_status=False
        )

        for attempt in range(max_replans):
            if pause_ctrl.is_quit_requested():
                print("[QUIT]")
                return False

            if pause_ctrl.is_pause_requested():
                self.emergency_release()
                if not pause_ctrl.wait_for_continue():
                    return False
                observations, logical_state, stale_ids, uncertain_ids = self.observe_world(
                    extra_viewpoints=True, print_status=False
                )
                continue

            # Count blocks
            num_blocks = sum(1 for p in logical_state if p[0] == 'box')
            if num_blocks == 0:
                print("No blocks! Waiting 5s...")
                time.sleep(5.0)
                observations, logical_state, stale_ids, uncertain_ids = self.observe_world(extra_viewpoints=True)
                continue

            # Compact status line
            print(f"\n{'='*20} Cycle {attempt+1}/{max_replans} | {num_blocks} blocks {'='*20}")

            # Block positions (compact)
            print_block_positions(observations)

            # Predicates grouped by type (compact)
            self._print_predicates_compact(logical_state)

            # Check goal
            if is_goal_satisfied(goal, logical_state):
                print("\n*** GOAL ACHIEVED! ***")
                return True

            # Unsatisfied goals (single line)
            unsatisfied = get_unsatisfied_goals(goal, logical_state)
            if unsatisfied:
                print(f"Need: {unsatisfied}")

            # Plan
            plan = solve_pddl(logical_state, goal, debug=False)
            if plan is None:
                print("Planning failed! Retrying...")
                time.sleep(5.0)
                observations, logical_state, stale_ids, uncertain_ids = self.observe_world(extra_viewpoints=True)
                continue

            if len(plan) == 0:
                print("\n*** GOAL ACHIEVED! ***")
                return True

            # Compact plan display
            plan_str = " -> ".join([f"{a}({','.join(args)})" for a, args in plan])
            print(f"Plan: {plan_str}")

            # Execute first action
            action_name, args = plan[0]
            print(f">> {action_name}({','.join(args)})")

            # Execute
            pick_actions = ['pick-up', 'unstack', 'remove-beside']
            place_actions = ['align', 'put-down', 'cover', 'release']

            if action_name in pick_actions:
                success, observations, logical_state = self._execute_pick_and_observe(
                    action_name, args, observations
                )
            elif action_name in place_actions:
                success = self.primitives.execute_action(action_name, args, observations)
                if success:
                    observations, logical_state, stale_ids, uncertain_ids = self.observe_world(
                        extra_viewpoints=False, print_status=False
                    )
            else:
                print(f"Unknown action: {action_name}")
                success = False

            if not success:
                print(f"FAILED: {action_name}")
                observations, logical_state, stale_ids, uncertain_ids = self.observe_world(extra_viewpoints=True)
                continue

            if pause_ctrl.is_quit_requested():
                return False
            if pause_ctrl.is_pause_requested():
                self.emergency_release()
                if not pause_ctrl.wait_for_continue():
                    return False
                observations, logical_state, stale_ids, uncertain_ids = self.observe_world(
                    extra_viewpoints=True, print_status=False
                )

        print("Max replans exceeded!")
        return False

    def _print_predicates_compact(self, predicates: List[Tuple]):
        """Print predicates in compact grouped format."""
        # Group by predicate type
        groups = {}
        for p in predicates:
            ptype = p[0]
            if ptype not in groups:
                groups[ptype] = []
            groups[ptype].append(p[1:])

        # Print each group on one line
        for ptype in ['on-table', 'above', 'beside', 'holding', 'hand_free', 'top']:
            if ptype in groups:
                items = [','.join(args) for args in groups[ptype]]
                print(f"  {ptype}: {' | '.join(items)}")

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
