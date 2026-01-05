"""
State API routes for querying system state.

Endpoints:
- GET /current - Get full current state
- GET /predicates - Get current PDDL predicates
- GET /plan - Get current plan
- GET /blocks - Get block positions
- GET /logs - Get recent logs
- GET /goals - Get available goals
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..state_manager import StateManager, ExecutionStatus
except ImportError:
    from state_manager import StateManager, ExecutionStatus


router = APIRouter()


# Dependency injection
def get_state_manager():
    try:
        from ..main import get_state_manager as _get
    except ImportError:
        from main import get_state_manager as _get
    return _get()


# Response models
class PredicateResponse(BaseModel):
    name: str
    args: List[str]


class BlockResponse(BaseModel):
    id: int
    block_class: int
    position: List[float]
    dimensions: List[float]
    confident: bool
    stale: bool


class PlanActionResponse(BaseModel):
    name: str
    args: List[str]
    status: str


class PlanResponse(BaseModel):
    actions: List[PlanActionResponse]
    current_step: int
    goal_name: str


class StateResponse(BaseModel):
    execution_status: str
    predicates: List[PredicateResponse]
    goal: List[PredicateResponse]
    goal_name: str
    goal_satisfied: bool
    unsatisfied_goals: List[PredicateResponse]
    blocks: List[BlockResponse]
    current_plan: List[PlanActionResponse]
    current_action_index: int
    cycle: int
    max_cycles: int
    gripper_position: List[float]
    gripper_open: bool
    holding_block: Optional[int]
    last_update: float


class LogResponse(BaseModel):
    level: str
    source: str
    message: str
    timestamp: str


class GoalInfo(BaseModel):
    name: str
    description: str
    predicates: List[PredicateResponse]


# Available goals (from block_world_planner.py GOAL_LIBRARY)
AVAILABLE_GOALS = {
    "bridge": {
        "description": "Build a bridge with a plank on two pillars",
        "predicates": [
            ("beside", "0", "1"),
            ("above_both", "4", "0", "1"),
        ]
    },
    "tall_bridge": {
        "description": "Build a tall bridge with stacked pillars",
        "predicates": [
            ("above", "2", "0"),
            ("above", "3", "1"),
            ("beside", "2", "3"),
            ("above_both", "4", "2", "3"),
        ]
    },
    "tower": {
        "description": "Build a 4-block tower",
        "predicates": [
            ("above", "1", "0"),
            ("above", "2", "1"),
            ("above", "3", "2"),
        ]
    },
    "cn_tower": {
        "description": "Build a 5-block tower",
        "predicates": [
            ("above", "1", "0"),
            ("above", "2", "1"),
            ("above", "3", "2"),
            ("above", "5", "3"),
        ]
    },
    "house": {
        "description": "Build a house structure",
        "predicates": [
            ("above", "2", "0"),
            ("above", "3", "1"),
            ("beside", "2", "3"),
            ("above_both", "4", "2", "3"),
            ("above", "5", "4"),
        ]
    },
}


# Endpoints

@router.get("/current", response_model=StateResponse)
async def get_current_state(
    state_manager: StateManager = Depends(get_state_manager)
):
    """Get complete current system state."""
    try:
        state_dict = state_manager.get_state_dict()

        return StateResponse(
            execution_status=state_dict["execution_status"],
            predicates=[PredicateResponse(**p) for p in state_dict["predicates"]],
            goal=[PredicateResponse(**g) for g in state_dict["goal"]],
            goal_name=state_dict["goal_name"],
            goal_satisfied=state_dict["goal_satisfied"],
            unsatisfied_goals=[PredicateResponse(**g) for g in state_dict["unsatisfied_goals"]],
            blocks=[BlockResponse(
                id=b["id"],
                block_class=b["class"],
                position=b["position"],
                dimensions=b["dimensions"],
                confident=b["confident"],
                stale=b["stale"]
            ) for b in state_dict["blocks"]],
            current_plan=[PlanActionResponse(**a) for a in state_dict["current_plan"]],
            current_action_index=state_dict["current_action_index"],
            cycle=state_dict["cycle"],
            max_cycles=state_dict["max_cycles"],
            gripper_position=state_dict["gripper_position"],
            gripper_open=state_dict["gripper_open"],
            holding_block=state_dict["holding_block"],
            last_update=state_dict["last_update"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predicates")
async def get_predicates(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get current PDDL predicates grouped by type."""
    try:
        state = state_manager.get_state()

        # Group predicates by type
        grouped = {}
        for pred in state.predicates:
            pred_type = pred[0]
            if pred_type not in grouped:
                grouped[pred_type] = []
            grouped[pred_type].append(list(pred[1:]))

        return {
            "predicates": grouped,
            "count": len(state.predicates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plan", response_model=PlanResponse)
async def get_current_plan(
    state_manager: StateManager = Depends(get_state_manager)
):
    """Get current plan and execution progress."""
    try:
        state = state_manager.get_state()

        actions = [
            PlanActionResponse(
                name=a.name,
                args=list(a.args),
                status=a.status
            )
            for a in state.current_plan
        ]

        return PlanResponse(
            actions=actions,
            current_step=state.current_action_index,
            goal_name=state.goal_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks")
async def get_block_positions(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get current block positions and states."""
    try:
        state = state_manager.get_state()

        blocks = [
            {
                "id": b.id,
                "class": b.block_class,
                "position": list(b.position),
                "dimensions": list(b.dimensions),
                "confident": b.confident,
                "stale": b.stale
            }
            for b in state.blocks
        ]

        return {
            "blocks": blocks,
            "gripper": {
                "position": list(state.gripper_position),
                "open": state.gripper_open,
                "holding": state.holding_block
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    level: Optional[str] = Query(default=None),
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get recent logs with optional filtering."""
    try:
        logs = state_manager.get_logs(limit)

        # Filter by level if specified
        if level:
            logs = [log for log in logs if log["level"] == level]

        return {
            "logs": logs,
            "count": len(logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/goals")
async def get_available_goals() -> Dict[str, Any]:
    """Get list of available goal configurations."""
    goals = []
    for name, info in AVAILABLE_GOALS.items():
        goals.append(GoalInfo(
            name=name,
            description=info["description"],
            predicates=[
                PredicateResponse(name=p[0], args=list(p[1:]))
                for p in info["predicates"]
            ]
        ))

    return {"goals": goals}


@router.get("/status")
async def get_execution_status(
    state_manager: StateManager = Depends(get_state_manager)
) -> Dict[str, Any]:
    """Get current execution status."""
    try:
        state = state_manager.get_state()

        return {
            "status": state.execution_status.value,
            "cycle": state.cycle,
            "max_cycles": state.max_cycles,
            "goal_name": state.goal_name,
            "goal_satisfied": state.goal_satisfied,
            "current_action": (
                {"name": state.current_plan[state.current_action_index].name,
                 "args": list(state.current_plan[state.current_action_index].args)}
                if 0 <= state.current_action_index < len(state.current_plan)
                else None
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
