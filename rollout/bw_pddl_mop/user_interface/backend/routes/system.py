"""
System API routes for robot connection management.

Endpoints:
- POST /connect - Initialize planner and connect to robot/camera
- GET /status - Get current connection status
"""

import os
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..state_manager import StateManager, ExecutionStatus
except ImportError:
    from state_manager import StateManager, ExecutionStatus


router = APIRouter()


# Response models
class ConnectResponse(BaseModel):
    success: bool
    message: str
    robot_connected: bool
    camera_connected: bool


class StatusResponse(BaseModel):
    initialized: bool
    robot_connected: bool
    camera_connected: bool
    execution_status: str


# Dependency to get planner adapter
def get_planner_adapter():
    try:
        from ..main import get_planner_adapter as _get
    except ImportError:
        from main import get_planner_adapter as _get
    return _get()


def get_state_manager():
    try:
        from ..main import get_state_manager as _get
    except ImportError:
        from main import get_state_manager as _get
    return _get()


def get_robot_config():
    try:
        from ..main import get_robot_config as _get
    except ImportError:
        from main import get_robot_config as _get
    return _get()


@router.post("/connect", response_model=ConnectResponse)
async def connect_to_robot():
    """
    Initialize the planner and connect to robot/camera.

    Creates PlannerAdapter if not exists and calls initialize().
    """
    try:
        from ..main import initialize_planner_adapter
    except ImportError:
        from main import initialize_planner_adapter

    state_manager = get_state_manager()

    # Check if already connected
    current_state = state_manager.get_state()
    if current_state.robot_connected and current_state.camera_connected:
        return ConnectResponse(
            success=True,
            message="Already connected",
            robot_connected=True,
            camera_connected=True
        )

    # Initialize planner adapter
    state_manager.add_log("info", "system", "Connecting to robot and camera...")

    try:
        success = initialize_planner_adapter()

        # Get updated state
        updated_state = state_manager.get_state()

        if success:
            return ConnectResponse(
                success=True,
                message="Connected successfully",
                robot_connected=updated_state.robot_connected,
                camera_connected=updated_state.camera_connected
            )
        else:
            # Determine what failed
            if not updated_state.robot_connected and not updated_state.camera_connected:
                message = "Failed to connect to robot and camera"
            elif not updated_state.robot_connected:
                message = "Failed to connect to robot"
            elif not updated_state.camera_connected:
                message = "Failed to connect to camera"
            else:
                message = "Connection failed"

            return ConnectResponse(
                success=False,
                message=message,
                robot_connected=updated_state.robot_connected,
                camera_connected=updated_state.camera_connected
            )

    except Exception as e:
        state_manager.add_log("error", "system", f"Connection error: {e}")
        return ConnectResponse(
            success=False,
            message=str(e),
            robot_connected=False,
            camera_connected=False
        )


@router.get("/status", response_model=StatusResponse)
async def get_system_status():
    """
    Get current system connection status.
    """
    state_manager = get_state_manager()
    adapter = get_planner_adapter()

    current_state = state_manager.get_state()

    return StatusResponse(
        initialized=adapter is not None and adapter.is_initialized(),
        robot_connected=current_state.robot_connected,
        camera_connected=current_state.camera_connected,
        execution_status=current_state.execution_status.value
    )
