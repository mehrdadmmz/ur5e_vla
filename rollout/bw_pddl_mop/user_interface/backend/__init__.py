"""
Backend package for Block World Planner UI.

Provides FastAPI server with WebSocket support for real-time
communication with the React frontend.
"""

from .state_manager import StateManager
from .command_bridge import CommandBridge

__all__ = ['StateManager', 'CommandBridge']
