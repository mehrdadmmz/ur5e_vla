"""
Command Bridge - Extended PauseController for UI integration.

This module extends the existing PauseController to support:
- UI-initiated commands via StateManager
- Command queue for goal changes and action injection
- State broadcasting integration
- Backward compatibility with keyboard/file-based controls
"""

import os
import sys
import time
import threading
import queue
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .state_manager import StateManager, ExecutionStatus
    from .goals import GOAL_LIBRARY, get_goal_predicates
except ImportError:
    from state_manager import StateManager, ExecutionStatus
    from goals import GOAL_LIBRARY, get_goal_predicates


class CommandType(str, Enum):
    """Types of commands that can be sent to the robot."""
    PAUSE = "pause"
    CONTINUE = "continue"
    QUIT = "quit"
    CHANGE_GOAL = "change_goal"
    INJECT_ACTION = "inject_action"
    REPLAN = "replan"


@dataclass
class Command:
    """A command to be executed by the robot."""
    type: CommandType
    params: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.params is None:
            self.params = {}


class CommandBridge:
    """
    Extended controller for UI integration.

    Provides:
    - Integration with StateManager for state broadcasting
    - Command queue for injected actions and goal changes
    - Backward compatibility with keyboard/file-based controls

    Usage:
        state_manager = StateManager()
        bridge = CommandBridge(state_manager)

        # In planner main loop:
        if bridge.is_pause_requested():
            bridge.handle_pause()

        # Check for pending commands
        cmd = bridge.get_pending_command()
        if cmd and cmd.type == CommandType.CHANGE_GOAL:
            new_goal = cmd.params['goal']

        # From UI:
        bridge.send_command(Command(CommandType.PAUSE))
    """

    # Priority levels for commands
    PRIORITY_CONTROL = 0
    PRIORITY_GOAL = 1
    PRIORITY_ACTION = 2

    def __init__(self, state_manager: StateManager, enable_keyboard: bool = True):
        """
        Initialize CommandBridge.

        Args:
            state_manager: StateManager instance for state coordination
            enable_keyboard: Whether to enable keyboard monitoring (default True)
        """
        self.state_manager = state_manager
        self.enable_keyboard = enable_keyboard

        # Thread safety
        self._lock = threading.RLock()

        # Command queue (priority queue)
        self._command_queue = queue.PriorityQueue()

        # Pending goal change
        self._pending_goal: Optional[str] = None
        self._pending_goal_predicates: Optional[List[Tuple]] = None

        # Current action tracking
        self._current_action: Optional[Tuple[str, Tuple]] = None

        # Keyboard monitoring (optional, for backward compatibility)
        self._keyboard_thread: Optional[threading.Thread] = None
        self._keyboard_running = False
        self._old_terminal_settings = None

        # File-based control paths
        self._pause_file = "/tmp/robot_pause"
        self._quit_file = "/tmp/robot_quit"

    def start(self):
        """Start the command bridge (including keyboard monitoring if enabled)."""
        self.state_manager.reset_control_state()

        if self.enable_keyboard:
            self._start_keyboard_monitor()

        self.state_manager.add_log("info", "system", "Command bridge started")

    def stop(self):
        """Stop the command bridge."""
        self._stop_keyboard_monitor()
        self.state_manager.add_log("info", "system", "Command bridge stopped")

    # =========================================================================
    # Command handling
    # =========================================================================

    def send_command(self, command: Command) -> None:
        """
        Send a command to the robot (from UI).

        Args:
            command: Command to send
        """
        # Determine priority
        if command.type in (CommandType.PAUSE, CommandType.CONTINUE, CommandType.QUIT):
            priority = self.PRIORITY_CONTROL
        elif command.type == CommandType.CHANGE_GOAL:
            priority = self.PRIORITY_GOAL
        else:
            priority = self.PRIORITY_ACTION

        # Add to queue
        self._command_queue.put((priority, command.timestamp, command))

        # Handle immediate commands
        if command.type == CommandType.PAUSE:
            self.state_manager.request_pause()
        elif command.type == CommandType.CONTINUE:
            self.state_manager.request_continue()
        elif command.type == CommandType.QUIT:
            self.state_manager.request_quit()
        elif command.type == CommandType.CHANGE_GOAL:
            goal_name = command.params.get('goal_name')
            goal_predicates = command.params.get('goal_predicates')

            # Get predicates from goals.py if not provided
            if goal_predicates is None and goal_name in GOAL_LIBRARY:
                goal_predicates = get_goal_predicates(goal_name)

            with self._lock:
                self._pending_goal = goal_name
                self._pending_goal_predicates = goal_predicates

            # Immediately update state so UI shows the goal
            if goal_predicates is not None:
                self.state_manager.update_state(
                    goal_name=goal_name,
                    goal=goal_predicates
                )

                # Recalculate goal satisfaction against current predicates
                self._recalculate_goal_status(goal_predicates)

        self.state_manager.add_log(
            "info", "system",
            f"Command received: {command.type.value}"
        )

    def get_pending_command(self, timeout: float = 0) -> Optional[Command]:
        """
        Get the next pending command (non-blocking by default).

        Args:
            timeout: How long to wait for a command (0 = non-blocking)

        Returns:
            Command if available, None otherwise
        """
        try:
            _, _, cmd = self._command_queue.get(timeout=timeout)
            return cmd
        except queue.Empty:
            return None

    def has_pending_commands(self) -> bool:
        """Check if there are pending commands."""
        return not self._command_queue.empty()

    # =========================================================================
    # Goal handling
    # =========================================================================

    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal change."""
        with self._lock:
            return self._pending_goal is not None

    def get_pending_goal(self) -> Optional[Tuple[str, List[Tuple]]]:
        """
        Get and clear the pending goal.

        Returns:
            Tuple of (goal_name, goal_predicates) or None
        """
        with self._lock:
            if self._pending_goal is None:
                return None
            goal_name = self._pending_goal
            goal_predicates = self._pending_goal_predicates
            self._pending_goal = None
            self._pending_goal_predicates = None
            return (goal_name, goal_predicates)

    def set_goal(self, goal_name: str, goal_predicates: List[Tuple]) -> None:
        """
        Set a new goal (from UI).

        Args:
            goal_name: Name of the goal (e.g., 'bridge', 'tower')
            goal_predicates: List of goal predicates
        """
        self.send_command(Command(
            type=CommandType.CHANGE_GOAL,
            params={'goal_name': goal_name, 'goal_predicates': goal_predicates}
        ))

    def _recalculate_goal_status(self, goal_predicates: List[Tuple]) -> None:
        """
        Recalculate goal satisfaction against current predicates.

        Args:
            goal_predicates: The goal predicates to check against
        """
        try:
            # Import perception module for goal checking
            try:
                from perception import get_unsatisfied_goals, is_goal_satisfied
            except ImportError:
                # If perception not available, skip calculation
                return

            # Get current predicates from state
            current_state = self.state_manager.get_state()
            current_predicates = current_state.predicates

            if not current_predicates:
                # No current state, mark all as unsatisfied
                self.state_manager.update_state(
                    goal_satisfied=False,
                    unsatisfied_goals=list(goal_predicates)
                )
                return

            # Calculate which goals are satisfied
            unsatisfied = get_unsatisfied_goals(goal_predicates, current_predicates)
            satisfied = is_goal_satisfied(goal_predicates, current_predicates)

            self.state_manager.update_state(
                goal_satisfied=satisfied,
                unsatisfied_goals=unsatisfied
            )

        except Exception as e:
            print(f"[CommandBridge] Error recalculating goal status: {e}")

    # =========================================================================
    # Action tracking
    # =========================================================================

    def set_current_action(self, action_name: str, args: Tuple) -> None:
        """Set the currently executing action."""
        with self._lock:
            self._current_action = (action_name, args)
        self.state_manager.add_log(
            "info", "execution",
            f"Executing: {action_name}({', '.join(str(a) for a in args)})"
        )

    def get_current_action(self) -> Optional[Tuple[str, Tuple]]:
        """Get the currently executing action."""
        with self._lock:
            return self._current_action

    def clear_current_action(self) -> None:
        """Clear the current action (action completed)."""
        with self._lock:
            self._current_action = None

    # =========================================================================
    # Pause/Continue/Quit - backward compatible interface
    # =========================================================================

    def is_pause_requested(self) -> bool:
        """
        Check if pause was requested.
        Checks both StateManager and file-based controls.
        """
        return (
            self.state_manager.is_pause_requested() or
            os.path.exists(self._pause_file)
        )

    def is_quit_requested(self) -> bool:
        """
        Check if quit was requested.
        Checks both StateManager and file-based controls.
        """
        return (
            self.state_manager.is_quit_requested() or
            os.path.exists(self._quit_file)
        )

    def clear_pause(self) -> None:
        """Clear pause request."""
        self.state_manager.request_continue()
        # Also remove file-based pause
        if os.path.exists(self._pause_file):
            try:
                os.remove(self._pause_file)
            except Exception:
                pass

    def wait_for_continue(self) -> bool:
        """
        Wait until continue or quit.
        Returns False if quit was requested.
        """
        self.state_manager.update_state(execution_status=ExecutionStatus.PAUSED)
        self.state_manager.add_log("info", "system", "Execution paused - waiting for continue")

        while True:
            if self.is_quit_requested():
                return False

            # Check if pause was cleared
            if not self.is_pause_requested():
                self.state_manager.add_log("info", "system", "Resuming execution")
                return True

            # Check StateManager continue event
            if self.state_manager.wait_for_continue(timeout=0.2):
                if not self.is_pause_requested():
                    self.state_manager.add_log("info", "system", "Resuming execution")
                    return True

            time.sleep(0.1)

    # =========================================================================
    # State broadcasting helpers
    # =========================================================================

    def broadcast_state(self, observations: List[List],
                       logical_state: List[Tuple],
                       stale_ids: List[int] = None,
                       uncertain_ids: List[int] = None,
                       gripper_pos: List[float] = None,
                       gripper_open: bool = None,
                       holding_block: int = None) -> None:
        """
        Broadcast current world state to UI.

        Args:
            observations: Raw observations from perception
            logical_state: PDDL predicates
            stale_ids: IDs of stale markers
            uncertain_ids: IDs of uncertain markers
            gripper_pos: Current gripper position [x, y, z]
            gripper_open: Whether gripper is open
            holding_block: ID of block being held (or None)
        """
        self.state_manager.update_observations(
            observations=observations,
            stale_ids=stale_ids or [],
            uncertain_ids=uncertain_ids or []
        )

        # Build update dict
        update_dict = {'predicates': logical_state}
        if gripper_pos is not None:
            update_dict['gripper_position'] = gripper_pos
        if gripper_open is not None:
            update_dict['gripper_open'] = gripper_open
        if holding_block is not None:
            update_dict['holding_block'] = holding_block

        self.state_manager.update_state(**update_dict)

    def broadcast_plan(self, plan: List[Tuple[str, Tuple]],
                      goal_name: str = "",
                      goal_predicates: List[Tuple] = None) -> None:
        """
        Broadcast current plan to UI.

        Args:
            plan: List of (action_name, args) tuples
            goal_name: Name of the current goal
            goal_predicates: Goal predicates
        """
        self.state_manager.update_plan(plan, goal_name)
        if goal_predicates:
            self.state_manager.update_state(goal=goal_predicates)

    def broadcast_action_start(self, action_name: str, args: Tuple, index: int) -> None:
        """Broadcast that an action is starting."""
        self.set_current_action(action_name, args)
        self.state_manager.update_state(current_action_index=index)
        self.state_manager.update_action_status(index, "executing")

    def broadcast_action_result(self, action_name: str, args: Tuple,
                               success: bool, index: int) -> None:
        """Broadcast action completion."""
        status = "completed" if success else "failed"
        self.state_manager.update_action_status(index, status)
        self.clear_current_action()

        level = "info" if success else "error"
        self.state_manager.add_log(
            level, "execution",
            f"Action {action_name}({', '.join(str(a) for a in args)}): {status}"
        )

    def broadcast_cycle(self, cycle: int, max_cycles: int) -> None:
        """Broadcast execution cycle progress."""
        self.state_manager.update_state(
            cycle=cycle,
            max_cycles=max_cycles,
            execution_status=ExecutionStatus.EXECUTING
        )

    def broadcast_goal_status(self, satisfied: bool,
                             unsatisfied: List[Tuple] = None) -> None:
        """Broadcast goal satisfaction status."""
        self.state_manager.update_state(
            goal_satisfied=satisfied,
            unsatisfied_goals=unsatisfied or []
        )
        if satisfied:
            self.state_manager.update_state(execution_status=ExecutionStatus.COMPLETED)
            self.state_manager.add_log("info", "system", "Goal achieved!")

    # =========================================================================
    # Keyboard monitoring (backward compatibility)
    # =========================================================================

    def _start_keyboard_monitor(self):
        """Start keyboard monitoring thread."""
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                print("[Info] No TTY available, keyboard controls disabled")
                return

            self._old_terminal_settings = termios.tcgetattr(sys.stdin)
            self._keyboard_running = True
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_loop,
                daemon=True
            )
            self._keyboard_thread.start()

            print("\n" + "=" * 60)
            print("CONTROLS:")
            print("  Keyboard: 'p'=pause, 'c'=continue, 'q'=quit")
            print("  Files: touch /tmp/robot_pause, rm to continue")
            print("  UI: Use web interface at http://localhost:8000")
            print("=" * 60 + "\n")

        except ImportError:
            print("[Info] termios not available, keyboard controls disabled")
        except Exception as e:
            print(f"[Warning] Could not initialize keyboard: {e}")

    def _stop_keyboard_monitor(self):
        """Stop keyboard monitoring thread."""
        self._keyboard_running = False

        if self._old_terminal_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
            except Exception:
                pass

        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)

    def _keyboard_loop(self):
        """Background thread for keyboard monitoring."""
        import select
        import tty
        import termios

        fd = sys.stdin.fileno()

        while self._keyboard_running:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)

                if readable:
                    tty.setcbreak(fd)
                    try:
                        ch = sys.stdin.read(1)
                    finally:
                        if self._old_terminal_settings:
                            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_terminal_settings)

                    if ch.lower() == 'p':
                        self.send_command(Command(CommandType.PAUSE))
                        sys.stdout.write("\n>>> PAUSE REQUESTED\n")
                        sys.stdout.flush()
                    elif ch.lower() == 'c':
                        self.send_command(Command(CommandType.CONTINUE))
                        sys.stdout.write("\n>>> CONTINUE\n")
                        sys.stdout.flush()
                    elif ch.lower() == 'q':
                        self.send_command(Command(CommandType.QUIT))
                        sys.stdout.write("\n>>> QUIT REQUESTED\n")
                        sys.stdout.flush()

            except Exception:
                pass

            time.sleep(0.05)
