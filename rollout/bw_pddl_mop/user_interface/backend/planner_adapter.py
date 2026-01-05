"""
Planner Adapter - Integrates BlockWorldPlanner with UI components.

This module provides:
- PlannerAdapter: Wraps BlockWorldPlanner with UI broadcasting
- Modified run loop that reports state to StateManager
- Support for goal changes and action injection from UI

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
from typing import List, Tuple, Optional, Dict, Any

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .state_manager import StateManager, ExecutionStatus
from .command_bridge import CommandBridge, Command, CommandType


class PlannerAdapter:
    """
    Adapter for BlockWorldPlanner that integrates with UI StateManager.

    Runs the planner in a separate thread while reporting state changes
    to the StateManager for UI broadcasting.
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

        self._planner = None
        self._execution_thread: Optional[threading.Thread] = None
        self._current_goal_name: Optional[str] = None
        self._current_goal: Optional[List[Tuple]] = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize the BlockWorldPlanner.

        Returns:
            True if initialization succeeded
        """
        try:
            # Import here to avoid circular imports and allow standalone backend testing
            from block_world_planner import BlockWorldPlanner, GOAL_LIBRARY

            self.state_manager.add_log("info", "system", "Initializing planner...")
            self.state_manager.update_state(execution_status=ExecutionStatus.IDLE)

            # Initialize planner (connects to robot and camera)
            self._planner = BlockWorldPlanner(self.config)
            self._goal_library = GOAL_LIBRARY
            self._is_initialized = True

            # Update connection status - robot and camera connected successfully
            self.state_manager.update_state(
                robot_connected=True,
                camera_connected=True
            )
            self.state_manager.add_log("info", "system", "Robot connected")
            self.state_manager.add_log("info", "system", "Camera connected")

            self.state_manager.add_log("info", "system", "Planner initialized successfully")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            # Try to determine which component failed
            if "robot" in error_msg or "rtde" in error_msg:
                self.state_manager.update_state(robot_connected=False)
                self.state_manager.add_log("error", "system", f"Robot connection failed: {e}")
            elif "camera" in error_msg or "realsense" in error_msg or "aruco" in error_msg:
                self.state_manager.update_state(robot_connected=True, camera_connected=False)
                self.state_manager.add_log("error", "system", f"Camera connection failed: {e}")
            else:
                self.state_manager.update_state(robot_connected=False, camera_connected=False)
                self.state_manager.add_log("error", "system", f"Failed to initialize planner: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if planner is initialized."""
        return self._is_initialized

    def is_running(self) -> bool:
        """Check if execution is in progress."""
        return self._execution_thread is not None and self._execution_thread.is_alive()

    def get_available_goals(self) -> Dict[str, List[Tuple]]:
        """Get available goal configurations."""
        if hasattr(self, '_goal_library'):
            return self._goal_library
        return {}

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
            self.state_manager.add_log("error", "system", "Planner not initialized")
            return False

        if self.is_running():
            self.state_manager.add_log("error", "system", "Execution already in progress")
            return False

        # Get goal from library
        if goal_name not in self._goal_library:
            self.state_manager.add_log("error", "system", f"Unknown goal: {goal_name}")
            return False

        self._current_goal_name = goal_name
        self._current_goal = self._goal_library[goal_name]

        # Reset control state
        self.command_bridge.state_manager.reset_control_state()

        # Start execution thread
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

    def shutdown(self) -> None:
        """Shutdown the adapter and planner."""
        self.stop_execution()

        if self._execution_thread is not None:
            self._execution_thread.join(timeout=5.0)

        if self._planner is not None:
            try:
                self._planner.close()
            except Exception:
                pass

        # Update connection status
        self.state_manager.update_state(
            robot_connected=False,
            camera_connected=False
        )
        self._is_initialized = False

        self.state_manager.add_log("info", "system", "Planner shutdown complete")

    # =========================================================================
    # Execution loop with UI integration
    # =========================================================================

    def _run_execution(self, goal: List[Tuple], max_replans: int):
        """
        Modified execution loop that reports state to StateManager.

        This is based on BlockWorldPlanner._run_loop() but adds:
        1. State broadcasting at key points
        2. Command checking for goal changes and action injection
        3. Integration with CommandBridge for pause/continue/quit
        """
        # Import required functions
        from predicate_util import (
            get_logical_state, is_goal_satisfied, get_unsatisfied_goals
        )
        from pddl_solver import solve_pddl

        self.state_manager.update_state(
            execution_status=ExecutionStatus.EXECUTING,
            goal_name=self._current_goal_name,
            goal=goal,
            max_cycles=max_replans
        )

        # Start command bridge
        self.command_bridge.start()

        try:
            # Initial observation
            self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
            observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                extra_viewpoints=True, print_status=False
            )

            # Broadcast initial state
            self.command_bridge.broadcast_state(
                observations, logical_state, stale_ids, uncertain_ids
            )

            for attempt in range(max_replans):
                # Update cycle count
                self.command_bridge.broadcast_cycle(attempt + 1, max_replans)

                # ===== CHECK FOR QUIT =====
                if self.command_bridge.is_quit_requested():
                    self.state_manager.add_log("info", "system", "Quit requested")
                    break

                # ===== CHECK FOR PAUSE =====
                if self.command_bridge.is_pause_requested():
                    self.state_manager.update_state(execution_status=ExecutionStatus.PAUSED)
                    self._planner.emergency_release()

                    if not self.command_bridge.wait_for_continue():
                        break

                    self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                    observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                        extra_viewpoints=True, print_status=False
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids
                    )
                    continue

                # ===== CHECK FOR GOAL CHANGE =====
                if self.command_bridge.has_pending_goal():
                    result = self.command_bridge.get_pending_goal()
                    if result:
                        new_goal_name, new_goal_predicates = result
                        if new_goal_name in self._goal_library:
                            goal = self._goal_library[new_goal_name]
                            self._current_goal_name = new_goal_name
                            self._current_goal = goal
                            self.state_manager.add_log(
                                "info", "system",
                                f"Goal changed to: {new_goal_name}"
                            )
                            self.state_manager.update_state(
                                goal_name=new_goal_name,
                                goal=goal
                            )
                            # Re-observe and replan
                            observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                                extra_viewpoints=True, print_status=False
                            )
                            self.command_bridge.broadcast_state(
                                observations, logical_state, stale_ids, uncertain_ids
                            )
                            continue

                # ===== COUNT BLOCKS =====
                num_blocks = sum(1 for p in logical_state if p[0] == 'box')
                if num_blocks == 0:
                    self.state_manager.add_log("warn", "perception", "No blocks detected, waiting...")
                    time.sleep(5.0)
                    observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids
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
                    observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids
                    )
                    continue

                # Broadcast plan
                self.command_bridge.broadcast_plan(plan, self._current_goal_name, goal)

                # ===== EXECUTE FIRST ACTION =====
                self.state_manager.update_state(execution_status=ExecutionStatus.EXECUTING)
                action_name, args = plan[0]

                # Broadcast action start
                self.command_bridge.broadcast_action_start(action_name, args, 0)

                # Execute action
                pick_actions = ['pick-up', 'unstack', 'remove-beside']
                place_actions = ['align', 'put-down', 'cover', 'release']

                success = False
                if action_name in pick_actions:
                    success, observations, logical_state = self._planner._execute_pick_and_observe(
                        action_name, args, observations
                    )
                elif action_name in place_actions:
                    success = self._planner.primitives.execute_action(action_name, args, observations)
                    if success:
                        self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                        observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                            extra_viewpoints=False, print_status=False
                        )
                else:
                    self.state_manager.add_log("error", "execution", f"Unknown action: {action_name}")

                # Broadcast action result
                self.command_bridge.broadcast_action_result(action_name, args, success, 0)

                # Broadcast updated state
                self.command_bridge.broadcast_state(
                    observations, logical_state,
                    stale_ids if 'stale_ids' in dir() else [],
                    uncertain_ids if 'uncertain_ids' in dir() else []
                )

                if not success:
                    self.state_manager.add_log("warn", "execution", f"Action failed: {action_name}")
                    observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                        extra_viewpoints=True
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids
                    )
                    continue

                # ===== POST-ACTION PAUSE CHECK =====
                if self.command_bridge.is_quit_requested():
                    break

                if self.command_bridge.is_pause_requested():
                    self.state_manager.update_state(execution_status=ExecutionStatus.PAUSED)
                    self._planner.emergency_release()

                    if not self.command_bridge.wait_for_continue():
                        break

                    self.state_manager.update_state(execution_status=ExecutionStatus.OBSERVING)
                    observations, logical_state, stale_ids, uncertain_ids = self._planner.observe_world(
                        extra_viewpoints=True, print_status=False
                    )
                    self.command_bridge.broadcast_state(
                        observations, logical_state, stale_ids, uncertain_ids
                    )

            # Loop ended
            self.state_manager.add_log("info", "system", "Execution loop ended")

        except Exception as e:
            error_msg = str(e).lower()
            # Check for connection-related errors
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

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create state manager if not provided
    if state_manager is None:
        state_manager = StateManager()

    # Create command bridge if not provided
    if command_bridge is None:
        command_bridge = CommandBridge(state_manager)

    # Create and initialize adapter
    adapter = PlannerAdapter(config, state_manager, command_bridge)

    return adapter
