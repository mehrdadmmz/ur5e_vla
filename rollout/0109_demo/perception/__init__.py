"""
Perception Module - ArUco detection and state conversion.

Provides:
- ArucoMonitor: ArUco marker detection and tracking
- create_base_camera_monitor: Factory for fixed camera
- create_wrist_camera_monitor: Factory for wrist-mounted camera
- get_logical_state: Convert observations to PDDL predicates
- is_goal_satisfied: Check if goal is satisfied
- set_thresholds: Configure predicate detection thresholds
"""

from .aruco_monitor import (
    ArucoMonitor,
    create_base_camera_monitor,
    create_wrist_camera_monitor,
)
from .predicates import (
    get_logical_state,
    is_goal_satisfied,
    get_unsatisfied_goals,
    set_thresholds,
    print_block_positions,
)

__all__ = [
    'ArucoMonitor',
    'create_base_camera_monitor',
    'create_wrist_camera_monitor',
    'get_logical_state',
    'is_goal_satisfied',
    'get_unsatisfied_goals',
    'set_thresholds',
    'print_block_positions',
]
