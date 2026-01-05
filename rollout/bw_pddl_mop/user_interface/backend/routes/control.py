"""
Control API routes for robot execution.

Endpoints:
- POST /pause - Pause execution
- POST /continue - Continue from pause
- POST /quit - Quit execution
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..state_manager import StateManager
    from ..command_bridge import CommandBridge, Command, CommandType
except ImportError:
    from state_manager import StateManager
    from command_bridge import CommandBridge, Command, CommandType


router = APIRouter()


# Dependency injection helpers
def get_state_manager():
    try:
        from ..main import get_state_manager as _get
    except ImportError:
        from main import get_state_manager as _get
    return _get()


def get_command_bridge():
    try:
        from ..main import get_command_bridge as _get
    except ImportError:
        from main import get_command_bridge as _get
    return _get()


def get_planner_adapter():
    try:
        from ..main import get_planner_adapter as _get
    except ImportError:
        from main import get_planner_adapter as _get
    return _get()


# Request/Response models
class ControlResponse(BaseModel):
    success: bool
    message: str


class GoalRequest(BaseModel):
    goal_name: str
    goal_predicates: Optional[List[tuple]] = None


class ActionRequest(BaseModel):
    action: str
    args: List[str]


class StartRequest(BaseModel):
    goal_name: str
    max_replans: int = 100


# Endpoints

@router.post("/pause", response_model=ControlResponse)
async def pause_execution(
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Pause robot execution after current action completes."""
    try:
        command_bridge.send_command(Command(CommandType.PAUSE))
        return ControlResponse(success=True, message="Pause requested")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue", response_model=ControlResponse)
async def continue_execution(
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Continue execution from paused state."""
    try:
        command_bridge.send_command(Command(CommandType.CONTINUE))
        return ControlResponse(success=True, message="Continue requested")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quit", response_model=ControlResponse)
async def quit_execution(
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Quit execution gracefully."""
    try:
        command_bridge.send_command(Command(CommandType.QUIT))
        return ControlResponse(success=True, message="Quit requested")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/goal", response_model=ControlResponse)
async def set_goal(
    request: GoalRequest,
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Change the current goal."""
    try:
        command_bridge.send_command(Command(
            CommandType.CHANGE_GOAL,
            params={
                "goal_name": request.goal_name,
                "goal_predicates": request.goal_predicates
            }
        ))
        return ControlResponse(
            success=True,
            message=f"Goal change to '{request.goal_name}' requested"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/action", response_model=ControlResponse)
async def execute_action(
    request: ActionRequest,
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Execute a specific action (manual intervention)."""
    try:
        command_bridge.send_command(Command(
            CommandType.INJECT_ACTION,
            params={
                "action": request.action,
                "args": tuple(request.args)
            }
        ))
        return ControlResponse(
            success=True,
            message=f"Action '{request.action}' queued for execution"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replan", response_model=ControlResponse)
async def force_replan(
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """Force replanning from current state."""
    try:
        command_bridge.send_command(Command(CommandType.REPLAN))
        return ControlResponse(success=True, message="Replan requested")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=ControlResponse)
async def start_execution(
    request: StartRequest,
    state_manager: StateManager = Depends(get_state_manager)
):
    """Start execution with specified goal."""
    adapter = get_planner_adapter()

    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail="Planner not initialized. Call /api/system/connect first."
        )

    if not adapter.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="Planner not initialized. Call /api/system/connect first."
        )

    if adapter.is_running():
        raise HTTPException(
            status_code=400,
            detail="Execution already in progress"
        )

    try:
        success = adapter.start_execution(request.goal_name, request.max_replans)
        if success:
            return ControlResponse(
                success=True,
                message=f"Started execution with goal: {request.goal_name}"
            )
        else:
            return ControlResponse(
                success=False,
                message=f"Failed to start execution with goal: {request.goal_name}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
