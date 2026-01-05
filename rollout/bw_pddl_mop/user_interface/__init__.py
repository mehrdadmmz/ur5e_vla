"""
User Interface package for Block World Planner.

Provides:
- FastAPI backend with WebSocket for real-time state updates
- React + Three.js frontend for 3D visualization
- Voice command integration via OpenAI Whisper
"""

from .backend.state_manager import StateManager
from .backend.command_bridge import CommandBridge

__all__ = ['StateManager', 'CommandBridge']
