"""
Robot Module - Hardware control for UR5e robot and gripper.

Provides:
- RobotController: RTDE interface wrapper
- Gripper: Robotiq gripper control
- BlockPrimitives: Block manipulation actions
"""

from .controller import RobotController
from .gripper import Gripper
from .primitives import BlockPrimitives

__all__ = ['RobotController', 'Gripper', 'BlockPrimitives']
