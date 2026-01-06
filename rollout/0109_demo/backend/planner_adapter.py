"""
Planner Adapter - Orchestrates robot, perception, and planning modules.

This module provides:
- PlannerAdapter: Coordinates all modules for UI-driven execution
- Clean module imports from planner/, robot/, perception/
- State broadcasting to UI via StateManager

Usage:
    state_manager = StateManager()
    command_bridge = CommandBridge(state_manager)

    adapter = PlannerAdapter(config, state_manager, command_bridge)
    adapter.initialize()
    adapter.start_execution('bridge')
"""

import os
import sys
import time
import threading
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Add parent path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle both module and direct execution imports
try:
    from .state_manager import StateManager, ExecutionStatus
    from .command_bridge import CommandBridge, Command, CommandType
    from .goals import GOAL_LIBRARY, get_goal_predicates
except ImportError:
    from state_manager import StateManager, ExecutionStatus
    from command_bridge import CommandBridge, Command, CommandType
    from goals import GOAL_LIBRARY, get_goal_predicates


class PlannerAdapter:
    """
    Orchestrates robot, perception, and planning modules.

    Uses the new modular architecture:
    - robot/: RobotController, Gripper, BlockPrimitives
    - perception/: ArucoMonitor, predicates
    - planner/: pddl_solver
    """

    def __init__(
        self,
        config: dict,
        state_manager: StateManager,
        command_bridge: CommandBridge
    ):
        """
        Initialize the adapter.

        Args:
            config: Robot/planner configuration dictionary
            state_manager: StateManager instance for state coordination
            command_bridge: CommandBridge instance for command handling
        """
        self.config = config
        self.state_manager = state_manager
        self.command_bridge = command_bridge

        # Module instances (created during initialize())
        self._robot = None
        self._gripper = None
        self._perception = None
        self._primitives = None

        # Execution state
        self._execution_thread: Optional[threading.Thread] = None
        self._current_goal_name: Optional[str] = None
        self._current_goal: Optional[List[Tuple]] = None
        self._is_initialized = False

        # Config references
        self.table_id = config.get('table_id', 9)
        self.robot_id = config.get('robot_id', 8)
        self.table_z = config.get('table_z', 0.0)

    def initialize(self) -> bool:
        """
        Initialize all modules (robot, gripper, perception, primitives).

        Returns:
            True if initialization succeeded
        """
        try:
            print("[PlannerAdapter] Importing modules...")

            # Import from sibling modules
            from robot import RobotController, Gripper, BlockPrimitives
            from perception import (
                create_base_camera_monitor,
                create_wrist_camera_monitor,
                set_thresholds
            )

            print("[PlannerAdapter] Modules imported successfully")

            self.state_manager.add_log("info", "system", "Initializing system...")
            self.state_manager.update_state(execution_status=ExecutionStatus.IDLE)

            # Initialize robot controller
            print("[PlannerAdapter] Connecting to robot...")
            self._robot = RobotController(self.config)
            if not self._robot.connect():
                raise RuntimeError("Failed to connect to robot")
            self.state_manager.update_state(robot_connected=True)
            self.state_manager.add_log("info", "system", "Robot connected")

            # Initialize gripper
            print("[PlannerAdapter] Connecting to gripper...")
            self._gripper = Gripper(self.config, self._robot.robot_ip)
            self._gripper.connect()  # Non-fatal if fails
            self.state_manager.add_log("info", "system", "Gripper connected")

            # Initialize perception (ArUco monitor)
            print("[PlannerAdapter] Initializing perception...")
            self._init_perception()
            self.state_manager.update_state(camera_connected=True)
            self.state_manager.add_log("info", "system", "Camera connected")

            # Initialize primitives
            print("[PlannerAdapter] Initializing primitives...")
            motion_cfg = self.config.get('motion', {})
            gripper_cfg = self.config.get('gripper', {})

            self._primitives = BlockPrimitives(
                rtde_c=self._robot.rtde_c,
                rtde_r=self._robot.rtde_r,
                gripper=self._gripper.gripper,
                speed=self.config['robot'].get('speed', 0.3),
                acceleration=self.config['robot'].get('acceleration', 0.3),
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
                table_bounds=self.config.get('table_bounds', None),
            )

            # Set grasp orientations from config
            if 'grasp_orientations' in self.config:
                grasp_orientations = {}
                for k, v in self.config['grasp_orientations'].items():
                    grasp_orientations[int(k)] = v
                self._primitives.set_grasp_orientations(grasp_orientations)
                print(f"[PlannerAdapter] Loaded grasp orientations for {len(grasp_orientations)} blocks")

            # Set joint limits from config
            if 'joint_limits' in self.config.get('robot', {}):
                joint_limits = {}
                for k, v in self.config['robot']['joint_limits'].items():
                    joint_limits[int(k)] = v
                self._primitives.set_joint_limits(joint_limits)
                print(f"[PlannerAdapter] Loaded joint limits for {len(joint_limits)} joints")

            # Set predicate thresholds
            if 'predicates' in self.config:
                set_thresholds(self.config['predicates'])

            self._is_initialized = True
            self.state_manager.add_log("info", "system", "System initialized successfully")
            print("[PlannerAdapter] Initialization complete")
            return True

        except Exception as e:
            import traceback
            print(f"[PlannerAdapter] Initialize failed: {e}")
            traceback.print_exc()

            error_msg = str(e).lower()
            if "robot" in error_msg or "rtde" in error_msg:
                self.state_manager.update_state(robot_connected=False)
                self.state_manager.add_log("error", "system", f"Robot connection failed: {e}")
            elif "camera" in error_msg or "realsense" in error_msg or "aruco" in error_msg:
                self.state_manager.update_state(robot_connected=True, camera_connected=False)
                self.state_manager.add_log("error", "system", f"Camera connection failed: {e}")
            else:
                self.state_manager.update_state(robot_connected=False, camera_connected=False)
                self.state_manager.add_log("error", "system", f"Failed to initialize: {e}")
            return False

    def _init_perception(self):
        """Initialize ArUco perception based on config."""
        from perception import create_base_camera_monitor, create_wrist_camera_monitor

        camera_cfg = self.config['camera']
        aruco_cfg = self.config.get('aruco', {})
        camera_type = camera_cfg.get('type', 'base')
        position_offset = camera_cfg.get('offset', [0.0, 0.0, 0.0])

        aruco_params = dict(
            marker_length=aruco_cfg.get('marker_size', 0.032),
            visual=self.config.get('visual', False),
            position_offset=position_offset,
            max_reproj_error=aruco_cfg.get('max_reproj_error', 2.0),
            min_sharpness=aruco_cfg.get('min_sharpness', 30.0),
            confident_reproj=aruco_cfg.get('confident_reproj', 1.0),
            confident_sharpness=aruco_cfg.get('confident_sharpness', 50.0),
            stale_age=aruco_cfg.get('stale_age', 2.0),
        )

        if camera_type == 'base':
            T_cam_base = np.array(camera_cfg['T_cam_base'])
            self._perception = create_base_camera_monitor(
                camera_serial=camera_cfg['serial'],
                T_cam_base=T_cam_base,
                **aruco_params,
            )
        elif camera_type == 'wrist':
            X_ee_cam = np.load(camera_cfg['calibration_file'])
            self._perception = create_wrist_camera_monitor(
                camera_serial=camera_cfg['serial'],
                X_ee_cam=X_ee_cam,
                rtde_r=self._robot.rtde_r,
                **aruco_params,
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        self._perception.start()
        print(f"[PlannerAdapter] Perception initialized ({camera_type} camera)")

    def observe_world(self, extra_viewpoints: bool = False, print_status: bool = True):
        """
        Get synchronized observations and logical state.

        Returns:
            Tuple of (observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open)
        """
        from perception import get_logical_state

        # Move to observation pose if configured
        if extra_viewpoints and hasattr(self._robot, 'init_pose') and self._robot.init_pose:
            self._robot.move_to_init_pose()
            if self._gripper.is_connected():
                self._gripper.open()
            time.sleep(self.config.get('motion', {}).get('stability_wait', 1.0))

        # Get observations from perception
        gripper_pos = self._primitives.get_gripper_position()
        gripper_open = self._primitives.is_gripper_open()

        observations, stale_ids, uncertain_ids = self._perception.get_block_observations(
            table_id=self.table_id,
            table_z=self.table_z,
            robot_id=self.robot_id,
            gripper_pos=gripper_pos,
            gripper_open=gripper_open,
            print_status=print_status,
        )

        # Convert to logical state
        logical_state = get_logical_state(observations, debug=False)

        return observations, logical_state, stale_ids, uncertain_ids, list(gripper_pos), gripper_open

    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._is_initialized

    def is_running(self) -> bool:
        """Check if execution is in progress."""
        return self._execution_thread is not None and self._execution_thread.is_alive()

    def move_to_home(self) -> bool:
        """
        Move robot to initial/home position.

        Returns:
            True if successful
        """
        if not self._is_initialized:
            self.state_manager.add_log("error", "system", "System not initialized")
            return False

        if self.is_running():
            self.state_manager.add_log("error", "system", "Cannot move to home while execution in progress")
            return False

        try:
            self.state_manager.add_log("info", "system", "Moving to home position...")

            # Open gripper first for safety
            if self._gripper.is_connected():
                self._gripper.open()

            # Move to init pose
            if hasattr(self._robot, 'move_to_init_pose'):
                self._robot.move_to_init_pose()
                self.state_manager.add_log("info", "system", "Moved to home position")
                return True
            else:
                self.state_manager.add_log("error", "system", "No init_pose configured")
                return False

        except Exception as e:
            self.state_manager.add_log("error", "system", f"Failed to move to home: {e}")
            return False

    def get_available_goals(self) -> Dict[str, List[Tuple]]:
        """Get available goal configurations."""
        return GOAL_LIBRARY

    def start_execution(self, goal_name: str, max_replans: int = 100) -> bool:
        """
        Start plan execution in background thread.

        Args:
            goal_name: Name of goal from GOAL_LIBRARY
            max_replans: Maximum number of replanning attempts

        Returns:
            True if execution started
        """
        if not self._is_initialized:
            self.state_manager.add_log("error", "system", "System not initialized")
            return False

        if self.is_running():
            self.state_manager.add_log("error", "system", "Execution already in progress")
            return False

        if goal_name not in GOAL_LIBRARY:
            self.state_manager.add_log("error", "system", f"Unknown goal: {goal_name}")
            return False

        self._current_goal_name = goal_name
        self._current_goal = get_goal_predicates(goal_name)

        self.command_bridge.state_manager.reset_control_state()

        self._execution_thread = threading.Thread(
            target=self._run_execution,
            args=(self._current_goal, max_replans),
            daemon=True
        )
        self._execution_thread.start()

        self.state_manager.add_log("info", "system", f"Started execution with goal: {goal_name}")
        return True

    def stop_execution(self) -> None:
        """Stop execution."""
        self.command_bridge.send_command(Command(CommandType.QUIT))

    def emergency_release(self) -> None:
        """Safe release when pausing - open gripper and move up."""
        try:
            self._gripper.open()
            time.sleep(0.3)
            # Move up slightly
            current_pose = self._robot.get_tcp_pose()
            current_pose[2] += 0.05  # Move up 5cm
            self._robot.move_to_pose(current_pose)
        except Exception as e:
            print(f"[PlannerAdapter] Emergency release error: {e}")

    def shutdown(self) -> None:
        """Shutdown all modules."""
        self.stop_execution()

        if self._execution_thread is not None:
            self._execution_thread.join(timeout=5.0)

        if self._perception is not None:
            try:
                self._perception.stop()
            except Exception:
                pass

        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass

        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass

        self.state_manager.update_state(
            robot_connected=False,
            camera_connected=False
        )
        self._is_initialized = False
        self.state_manager.add_log("info", "system", "System shutdown complete")

    # =========================================================================
    # Execution loop with UI integration
    # =========================================================================

    def _run_execution(self, goal: List[Tuple], max_replans: int):
        """
        Main execution loop that reports state to StateManager.
        """
        from perception import get_logical_state, is_goal_satisfied, get_unsatisfied_goals
        from planner import solve_pddl

        self.state_manager.update_state(
            execution_status=ExecutionStatus.EXECUTING,
            goal_name=self._current_goal_name,
            goal=goal,
            max_cycles=max_replans
        )

        self.command_bridge.start()

        try:
            # Initial observation
            self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
            observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                extra_viewpoints=True, print_status=False
            )

            self.command_bridge.broadcast_state(
                observations, logical_state, stale_ids, uncertain_ids,
                gripper_pos, gripper_open
            )

            for attempt in range(max_replans):
                self.command_bridge.broadcast_cycle(attempt + 1, max_replans)

                # ===== CHECK FOR QUIT =====
                if self.command_bridge.is_quit_requested():
                    self.state_manager.add_log("info", "system", "Quit requested")
                    break

                # ===== CHECK FOR PAUSE =====
                if self.command_bridge.is_pause_requested():
                    self.state_manager.update_state(execution_status=ExecutionStatus.PAUSED)
                    self.emergency_release()

                    if not self.command_bridge.wait_for_continue():
                        break

                    self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                    observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                        extra_viewpoints=True, print_status=False
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids,
                        gripper_pos, gripper_open
                    )
                    continue

                # ===== CHECK FOR GOAL CHANGE =====
                if self.command_bridge.has_pending_goal():
                    result = self.command_bridge.get_pending_goal()
                    if result:
                        new_goal_name, _ = result
                        if new_goal_name in GOAL_LIBRARY:
                            goal = get_goal_predicates(new_goal_name)
                            self._current_goal_name = new_goal_name
                            self._current_goal = goal
                            self.state_manager.add_log("info", "system", f"Goal changed to: {new_goal_name}")
                            self.state_manager.update_state(goal_name=new_goal_name, goal=goal)
                            observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                                extra_viewpoints=True, print_status=False
                            )
                            self.command_bridge.broadcast_state(
                                observations, logical_state, stale_ids, uncertain_ids,
                                gripper_pos, gripper_open
                            )
                            continue

                # ===== COUNT BLOCKS =====
                num_blocks = sum(1 for p in logical_state if p[0] == 'box')
                if num_blocks == 0:
                    self.state_manager.add_log("warn", "perception", "No blocks detected, waiting...")
                    time.sleep(5.0)
                    observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids,
                        gripper_pos, gripper_open
                    )
                    continue

                # ===== CHECK GOAL =====
                unsatisfied = get_unsatisfied_goals(goal, logical_state)
                self.command_bridge.broadcast_goal_status(
                    satisfied=len(unsatisfied) == 0,
                    unsatisfied=unsatisfied
                )

                if is_goal_satisfied(goal, logical_state):
                    self.state_manager.add_log("info", "system", "Goal achieved!")
                    self.state_manager.update_state(execution_status=ExecutionStatus.COMPLETED)
                    return

                # ===== PLANNING =====
                self.state_manager.update_state(execution_status=ExecutionStatus.PLANNING)
                plan = solve_pddl(logical_state, goal, debug=False)

                if plan is None or len(plan) == 0:
                    if plan is None:
                        self.state_manager.add_log("warn", "planner", "Planning failed, retrying...")
                    else:
                        self.state_manager.add_log("info", "system", "Goal achieved!")
                        self.state_manager.update_state(execution_status=ExecutionStatus.COMPLETED)
                        return

                    time.sleep(5.0)
                    observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids,
                        gripper_pos, gripper_open
                    )
                    continue

                self.command_bridge.broadcast_plan(plan, self._current_goal_name, goal)

                # ===== EXECUTE FIRST ACTION =====
                self.state_manager.update_state(execution_status=ExecutionStatus.EXECUTING)
                action_name, args = plan[0]

                self.command_bridge.broadcast_action_start(action_name, args, 0)

                pick_actions = ['pick-up', 'unstack', 'remove-beside']
                place_actions = ['align', 'put-down', 'cover', 'release']

                success = False
                if action_name in pick_actions:
                    success = self._execute_pick_action(action_name, args, observations)
                    if success:
                        # Re-observe after pick
                        self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                        observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                            extra_viewpoints=False, print_status=False
                        )
                elif action_name in place_actions:
                    block_transforms = self._perception.get_block_transforms()
                    success = self._primitives.execute_action(
                        action_name, args, observations, block_transforms
                    )
                    if success:
                        self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                        observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                            extra_viewpoints=False, print_status=False
                        )
                else:
                    self.state_manager.add_log("error", "execution", f"Unknown action: {action_name}")

                self.command_bridge.broadcast_action_result(action_name, args, success, 0)
                self.command_bridge.broadcast_state(
                    observations, logical_state,
                    stale_ids if 'stale_ids' in dir() else [],
                    uncertain_ids if 'uncertain_ids' in dir() else [],
                    gripper_pos if 'gripper_pos' in dir() else None,
                    gripper_open if 'gripper_open' in dir() else None
                )

                if not success:
                    self.state_manager.add_log("warn", "execution", f"Action failed: {action_name}")
                    observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids,
                        gripper_pos, gripper_open
                    )
                    continue

                # ===== POST-ACTION PAUSE CHECK =====
                if self.command_bridge.is_quit_requested():
                    break

                if self.command_bridge.is_pause_requested():
                    self.state_manager.update_state(execution_status=ExecutionStatus.PAUSED)
                    self.emergency_release()

                    if not self.command_bridge.wait_for_continue():
                        break

                    self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                    observations, logical_state, stale_ids, uncertain_ids, gripper_pos, gripper_open = self.observe_world(
                        extra_viewpoints=True, print_status=False
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids,
                        gripper_pos, gripper_open
                    )

            self.state_manager.add_log("info", "system", "Execution loop ended")

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e).lower()
            if "robot" in error_msg or "rtde" in error_msg or "connection" in error_msg:
                self.state_manager.update_state(robot_connected=False)
                self.state_manager.add_log("error", "system", f"Robot connection lost: {e}")
            elif "camera" in error_msg or "realsense" in error_msg:
                self.state_manager.update_state(camera_connected=False)
                self.state_manager.add_log("error", "system", f"Camera connection lost: {e}")
            else:
                self.state_manager.add_log("error", "system", f"Execution error: {e}")
            self.state_manager.update_state(execution_status=ExecutionStatus.ERROR)

        finally:
            self.command_bridge.stop()
            if self.state_manager.get_state().execution_status not in (
                ExecutionStatus.COMPLETED, ExecutionStatus.ERROR
            ):
                self.state_manager.update_state(execution_status=ExecutionStatus.IDLE)

    def _execute_pick_action(self, action_name: str, args: tuple, observations: list) -> bool:
        """Execute pick action and return success."""
        block_transforms = self._perception.get_block_transforms()
        success = self._primitives.execute_action(
            action_name, args, observations, block_transforms
        )
        return success


def create_planner_with_ui(
    config_path: str,
    state_manager: StateManager = None,
    command_bridge: CommandBridge = None
) -> PlannerAdapter:
    """
    Create a PlannerAdapter with UI integration.

    Args:
        config_path: Path to config.yaml
        state_manager: Optional StateManager (creates one if not provided)
        command_bridge: Optional CommandBridge (creates one if not provided)

    Returns:
        Configured PlannerAdapter
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if state_manager is None:
        state_manager = StateManager()

    if command_bridge is None:
        command_bridge = CommandBridge(state_manager)

    adapter = PlannerAdapter(config, state_manager, command_bridge)
    return adapter
