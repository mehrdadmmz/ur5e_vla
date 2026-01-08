"""
Thread-safe state manager for coordinating between robot execution and UI.

This module provides the central state hub that:
- Stores the complete system state (observations, predicates, plan, etc.)
- Provides thread-safe access via locks
- Notifies WebSocket clients of state changes
- Manages control events (pause/continue/quit)
"""

import time
import threading
import asyncio
import queue
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable
from enum import Enum


class ExecutionStatus(str, Enum):
    """Execution status states."""
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class BlockState:
    """State of a single block."""
    id: int
    block_class: int  # 2=cube, 3=plank
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    yaw: float = 0.0  # rotation around Z-axis in radians
    confident: bool = True
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.block_class,
            "position": list(self.position),
            "dimensions": list(self.dimensions),
            "yaw": self.yaw,
            "confident": self.confident,
            "stale": self.stale
        }


@dataclass
class PlanAction:
    """A single action in the plan."""
    name: str
    args: Tuple[str, ...]
    status: str = "pending"  # pending, executing, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "status": self.status
        }


@dataclass
class SystemState:
    """Complete system state for UI broadcasting."""

    # Execution state
    execution_status: ExecutionStatus = ExecutionStatus.IDLE

    # PDDL State
    predicates: List[Tuple] = field(default_factory=list)
    goal: List[Tuple] = field(default_factory=list)
    goal_name: str = ""
    goal_satisfied: bool = False
    unsatisfied_goals: List[Tuple] = field(default_factory=list)

    # Block positions
    blocks: List[BlockState] = field(default_factory=list)

    # Raw observations
    observations: List[List] = field(default_factory=list)
    stale_ids: List[int] = field(default_factory=list)
    uncertain_ids: List[int] = field(default_factory=list)

    # Plan state
    current_plan: List[PlanAction] = field(default_factory=list)
    current_action_index: int = -1

    # Execution progress
    cycle: int = 0
    max_cycles: int = 100

    # Robot state
    gripper_position: Tuple[float, float, float] = (0.0, 0.5, 0.6)
    gripper_open: bool = True
    holding_block: Optional[int] = None

    # Connection status
    robot_connected: bool = False
    camera_connected: bool = False

    # Timestamps
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for WebSocket."""
        return {
            "execution_status": self.execution_status.value,
            "predicates": [{"name": p[0], "args": list(p[1:])} for p in self.predicates],
            "goal": [{"name": g[0], "args": list(g[1:])} for g in self.goal],
            "goal_name": self.goal_name,
            "goal_satisfied": self.goal_satisfied,
            "unsatisfied_goals": [{"name": g[0], "args": list(g[1:])} for g in self.unsatisfied_goals],
            "blocks": [b.to_dict() for b in self.blocks],
            "observations": self.observations,
            "stale_ids": self.stale_ids,
            "uncertain_ids": self.uncertain_ids,
            "current_plan": [a.to_dict() for a in self.current_plan],
            "current_action_index": self.current_action_index,
            "cycle": self.cycle,
            "max_cycles": self.max_cycles,
            "gripper_position": list(self.gripper_position),
            "gripper_open": self.gripper_open,
            "holding_block": self.holding_block,
            "robot_connected": self.robot_connected,
            "camera_connected": self.camera_connected,
            "last_update": self.last_update
        }


class StateManager:
    """
    Thread-safe state manager for coordinating between:
    - Robot execution thread
    - Perception thread (ArucoMonitor)
    - FastAPI/WebSocket handlers

    Usage:
        state_manager = StateManager()

        # From robot thread:
        state_manager.update_state(
            predicates=logical_state,
            observations=observations
        )

        # From FastAPI:
        state = state_manager.get_state()
        state_manager.request_pause()
    """

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant for nested calls
        self._state = SystemState()

        # Control events
        self._pause_requested = threading.Event()
        self._quit_requested = threading.Event()
        self._continue_event = threading.Event()

        # Async queue for WebSocket notifications
        self._update_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Log buffer for UI
        self._log_buffer: List[Dict] = []
        self._max_log_buffer = 500

        # State change callbacks (for non-async listeners)
        self._callbacks: List[Callable[[Dict], None]] = []

    def set_async_context(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """Set async context from FastAPI startup."""
        self._update_queue = queue
        self._loop = loop

    def register_callback(self, callback: Callable[[Dict], None]):
        """Register a callback for state changes."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict], None]):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_state(self) -> SystemState:
        """Get a copy of current state (thread-safe)."""
        with self._lock:
            # Create a shallow copy
            return SystemState(
                execution_status=self._state.execution_status,
                predicates=list(self._state.predicates),
                goal=list(self._state.goal),
                goal_name=self._state.goal_name,
                goal_satisfied=self._state.goal_satisfied,
                unsatisfied_goals=list(self._state.unsatisfied_goals),
                blocks=list(self._state.blocks),
                observations=list(self._state.observations),
                stale_ids=list(self._state.stale_ids),
                uncertain_ids=list(self._state.uncertain_ids),
                current_plan=list(self._state.current_plan),
                current_action_index=self._state.current_action_index,
                cycle=self._state.cycle,
                max_cycles=self._state.max_cycles,
                gripper_position=self._state.gripper_position,
                gripper_open=self._state.gripper_open,
                holding_block=self._state.holding_block,
                robot_connected=self._state.robot_connected,
                camera_connected=self._state.camera_connected,
                last_update=self._state.last_update
            )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state as dictionary (thread-safe)."""
        with self._lock:
            return self._state.to_dict()

    def update_state(self, **kwargs) -> None:
        """
        Update state fields and notify UI (thread-safe).

        Args:
            **kwargs: State fields to update

        Example:
            state_manager.update_state(
                execution_status=ExecutionStatus.EXECUTING,
                predicates=logical_state,
                current_action_index=0
            )
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.last_update = time.time()
            state_dict = self._state.to_dict()

        # Notify WebSocket in async context (non-blocking)
        self._notify_update({"type": "state_update", "data": state_dict})

        # Call sync callbacks
        for callback in self._callbacks:
            try:
                callback(state_dict)
            except Exception:
                pass  # Don't let callback errors affect state updates

    def update_observations(self, observations: List[List],
                           stale_ids: List[int],
                           uncertain_ids: List[int],
                           block_dims: Dict[int, List[float]] = None):
        """
        Update observations and derive block states.

        Args:
            observations: Raw observations [[id, class, x, y, z, w, l, h], ...]
            stale_ids: IDs of stale markers
            uncertain_ids: IDs of uncertain markers
            block_dims: Optional block dimensions mapping
        """
        blocks = []
        for obs in observations:
            if len(obs) >= 8 and obs[1] in (2, 3):  # Only blocks (cube=2, plank=3)
                yaw = float(obs[8]) if len(obs) > 8 else 0.0
                block = BlockState(
                    id=int(obs[0]),
                    block_class=int(obs[1]),
                    position=(float(obs[2]), float(obs[3]), float(obs[4])),
                    dimensions=(float(obs[5]), float(obs[6]), float(obs[7])),
                    yaw=yaw,
                    confident=int(obs[0]) not in uncertain_ids,
                    stale=int(obs[0]) in stale_ids
                )
                blocks.append(block)

        self.update_state(
            observations=observations,
            blocks=blocks,
            stale_ids=stale_ids,
            uncertain_ids=uncertain_ids
        )

    def update_plan(self, plan: List[Tuple[str, Tuple]], goal_name: str = ""):
        """
        Update the current plan.

        Args:
            plan: List of (action_name, args) tuples
            goal_name: Name of the current goal
        """
        plan_actions = [
            PlanAction(name=action, args=args, status="pending")
            for action, args in plan
        ]
        self.update_state(
            current_plan=plan_actions,
            current_action_index=0 if plan_actions else -1,
            goal_name=goal_name
        )

    def update_action_status(self, index: int, status: str):
        """Update the status of a specific action in the plan."""
        with self._lock:
            if 0 <= index < len(self._state.current_plan):
                self._state.current_plan[index].status = status
                self._state.last_update = time.time()
                state_dict = self._state.to_dict()

        self._notify_update({"type": "state_update", "data": state_dict})

    def _notify_update(self, message: Dict) -> None:
        """Send update to WebSocket queue (non-blocking)."""
        message["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if self._update_queue and self._loop:
            try:
                self._loop.call_soon_threadsafe(
                    lambda: self._update_queue.put_nowait(message)
                )
            except (asyncio.QueueFull, RuntimeError):
                pass  # Drop if queue full or loop closed

    # =========================================================================
    # Control methods
    # =========================================================================

    def request_pause(self) -> None:
        """Request pause (called from UI)."""
        self._pause_requested.set()
        self._continue_event.clear()
        self.update_state(execution_status=ExecutionStatus.PAUSED)
        self.add_log("info", "system", "Pause requested")

    def request_continue(self) -> None:
        """Request continue (called from UI)."""
        self._pause_requested.clear()
        self._continue_event.set()
        self.add_log("info", "system", "Continue requested")

    def request_quit(self) -> None:
        """Request quit (called from UI)."""
        self._quit_requested.set()
        self._continue_event.set()  # Unblock if waiting
        # Clear goal state on quit
        self.update_state(
            goal=[],
            goal_name="",
            goal_satisfied=False,
            unsatisfied_goals=[],
            current_plan=[],
            current_action_index=-1,
            execution_status=ExecutionStatus.IDLE
        )
        self.add_log("info", "system", "Quit requested - goal cleared")

    def is_pause_requested(self) -> bool:
        """Check if pause requested (called from execution thread)."""
        return self._pause_requested.is_set()

    def is_quit_requested(self) -> bool:
        """Check if quit requested (called from execution thread)."""
        return self._quit_requested.is_set()

    def wait_for_continue(self, timeout: float = None) -> bool:
        """
        Wait for continue signal (called from execution thread).
        Returns True if continue, False if quit.
        """
        self._continue_event.wait(timeout)
        return not self._quit_requested.is_set()

    def reset_control_state(self) -> None:
        """Reset all control events for new execution."""
        self._pause_requested.clear()
        self._quit_requested.clear()
        self._continue_event.clear()

    # =========================================================================
    # Logging
    # =========================================================================

    def add_log(self, level: str, source: str, message: str) -> None:
        """
        Add log entry and notify UI.

        Args:
            level: Log level (debug, info, warn, error)
            source: Source component (planner, perception, execution, system)
            message: Log message
        """
        log_entry = {
            "level": level,
            "source": source,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        with self._lock:
            self._log_buffer.append(log_entry)
            if len(self._log_buffer) > self._max_log_buffer:
                self._log_buffer.pop(0)

        self._notify_update({"type": "log", "data": log_entry})

    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent logs."""
        with self._lock:
            return list(self._log_buffer[-limit:])

    def clear_logs(self) -> None:
        """Clear log buffer."""
        with self._lock:
            self._log_buffer.clear()
