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
from typing import List, Tuple, Dict, Optional

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
        self.primitives = BlockPrimitives(
            rtde_c=self.rtde_c,
            rtde_r=self.rtde_r,
            gripper=self.gripper,
            speed=config['robot'].get('speed', 0.3),
            acceleration=config['robot'].get('acceleration', 0.3),
            approach_height=config.get('motion', {}).get('approach_height', 0.10),
            grasp_depth=config.get('motion', {}).get('grasp_depth', 0.02),
            gripper_z_offset=config.get('gripper', {}).get('z_offset', 0.0),
            gripper_open_pos=config.get('gripper', {}).get('open_pos', 0),
            action_delay=config.get('motion', {}).get('action_delay', 0.5),
            beside_gap=config.get('motion', {}).get('beside_gap', 0.06),
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

    def move_to_init_pose(self):
        """Move robot to initial Cartesian pose if configured.

        Pose format: [x, y, z, qx, qy, qz, qw] (meters and quaternion)
        Converts to RTDE format: [x, y, z, rx, ry, rz] (axis-angle)
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

        # Open gripper to configured position
        self.primitives._open_gripper()
        print("Reached init pose.")

    def _init_perception(self, config: dict):
        """Initialize ArUco monitor based on config."""
        camera_cfg = config['camera']
        camera_type = camera_cfg.get('type', 'base')
        position_offset = camera_cfg.get('offset', [0.0, 0.0, 0.0])

        if camera_type == 'base':
            # Base-mounted camera with fixed transform
            T_cam_base = np.array(camera_cfg['T_cam_base'])
            self.aruco_monitor = create_base_camera_monitor(
                camera_serial=camera_cfg['serial'],
                T_cam_base=T_cam_base,
                marker_length=config.get('aruco', {}).get('marker_size', 0.032),
                visual=config.get('visual', False),
                position_offset=position_offset,
            )
        elif camera_type == 'wrist':
            # Wrist-mounted camera with hand-eye calibration
            X_ee_cam = np.load(camera_cfg['calibration_file'])
            self.aruco_monitor = create_wrist_camera_monitor(
                camera_serial=camera_cfg['serial'],
                X_ee_cam=X_ee_cam,
                rtde_r=self.rtde_r,
                marker_length=config.get('aruco', {}).get('marker_size', 0.032),
                visual=config.get('visual', False),
                position_offset=position_offset,
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        self.aruco_monitor.start()
        print(f"Perception initialized ({camera_type} camera)")

    def get_observations(self, print_status: bool = True) -> Tuple[List[List], List[int]]:
        """Get current world state from perception.

        Returns:
            Tuple of (observations list, list of stale marker IDs)
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
        observations, stale_ids = self.get_observations(print_status=print_status)
        return get_logical_state(observations, debug=debug)

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
            observations, _ = self.get_observations(print_status=False)

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

        Args:
            goal: List of goal predicates
            max_replans: Maximum number of replanning attempts

        Returns:
            True if goal achieved
        """
        print("\n" + "="*60)
        print("BLOCK WORLD PLANNER")
        print("="*60)

        # Move to initial pose first
        self.move_to_init_pose()

        # Wait for camera to stabilize and detect markers
        print("Waiting for perception to stabilize...")
        time.sleep(2.0)

        # Track logical state (can be from perception or symbolic updates)
        logical_state = None
        use_symbolic_state = False  # True after pick actions

        for attempt in range(max_replans):
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

            # Get observations for action execution
            observations, stale_ids = self.get_observations(print_status=False)

            # Log all object locations before action
            print_block_positions(observations)
            if stale_ids:
                print(f"WARNING: Stale markers (not seen recently): {stale_ids}")

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
                observations, stale_ids = self.get_observations(print_status=True)
                print_block_positions(observations)
                if stale_ids:
                    print(f"WARNING: Stale markers (not seen recently): {stale_ids}")

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

    # Get goal
    goal = GOAL_LIBRARY[args.goal]
    print(f"Goal: {args.goal}")

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
    finally:
        planner.close()


if __name__ == '__main__':
    main()
