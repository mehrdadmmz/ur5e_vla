"""
Voice command API routes.

Endpoints:
- POST /command - Process voice command (audio -> text -> action)
- POST /transcribe - Transcribe audio only
"""

import os
import base64
import tempfile
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..state_manager import StateManager, ExecutionStatus
    from ..command_bridge import CommandBridge, Command, CommandType
    from ..config import config
except ImportError:
    from state_manager import StateManager, ExecutionStatus
    from command_bridge import CommandBridge, Command, CommandType
    from config import config


router = APIRouter()


# Dependency injection
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


# Request/Response models
class VoiceCommandRequest(BaseModel):
    audio_base64: str
    format: str = "wav"  # wav, mp3, webm


class TranscriptionResponse(BaseModel):
    text: str
    success: bool


class VoiceCommandResponse(BaseModel):
    transcript: str
    commands: List[Dict[str, Any]] = []  # List of executed commands
    executed: bool = False
    message: str


# LLM prompt for command parsing
COMMAND_PARSE_SYSTEM_PROMPT = """You are a command parser for a robot block manipulation system. Parse the user's voice command into structured actions.

IMPORTANT: You can return MULTIPLE commands if the user requests multiple actions (e.g., "stop and build a bridge").

Available commands:

1. Control commands:
   - "pause" - pause robot execution
   - "continue" - resume robot execution
   - "quit" - stop and exit

2. Goal commands (set_goal):
   - "bridge" - build a bridge structure (two pillars with a plank on top)
   - "tall_bridge" - build a tall bridge (double-height pillars)
   - "tower" - build a vertical stack of 4 blocks
   - "totem_pole" - build a totem pole (5-block tall tower)
   - "house" - build a house structure
   - "boat" - build a boat structure
   - "tu" - build Chinese character 土 (tǔ, earth). User may say: "build tu", "建个土", "搭土字"
   - "gan" - build Chinese character 干 (gān, dry). User may say: "build gan", "建个干", "搭干字"
   - "shi" - build Chinese character 十 (shí, ten). User may say: "build shi", "建个十", "搭十字"
   - "wang" - build Chinese character 王 (wáng, king). User may say: "build wang", "建个王", "搭王字"

3. Action commands:
   - "pick-up" - pick up a block (requires block_id)
   - "put-down" - place block on another (requires block_id)
   - "release" - release/drop held block

Respond with JSON only. Return an array of commands:
{"commands": [{"command": "<type>", "params": {...}}, ...]}

Single command example:
{"commands": [{"command": "pause", "params": {}}]}

Multiple commands example:
{"commands": [{"command": "pause", "params": {}}, {"command": "set_goal", "params": {"goal_name": "bridge"}}]}

If the command is unclear or not related to robot control, return empty array:
{"commands": []}

Examples:
- "stop the robot" -> {"commands": [{"command": "pause", "params": {}}]}
- "build me a tower" -> {"commands": [{"command": "set_goal", "params": {"goal_name": "tower"}}]}
- "stop and build a bridge" -> {"commands": [{"command": "pause", "params": {}}, {"command": "set_goal", "params": {"goal_name": "bridge"}}]}
- "quit everything and make a house" -> {"commands": [{"command": "quit", "params": {}}, {"command": "set_goal", "params": {"goal_name": "house"}}]}
- "grab block 3" -> {"commands": [{"command": "action", "params": {"action": "pick-up", "block_id": "3"}}]}
- "建个土" -> {"commands": [{"command": "set_goal", "params": {"goal_name": "tu"}}]}
- "搭王字" -> {"commands": [{"command": "set_goal", "params": {"goal_name": "wang"}}]}
- "what's the weather" -> {"commands": []}
- "hello" -> {"commands": []}
"""


async def parse_command(transcript: str) -> List[Dict[str, Any]]:
    """
    Parse transcript into commands using LLM.

    Returns:
        List of command dicts, each with 'command' and 'params' keys.
        Returns empty list if no commands recognized.
    """
    if not transcript or not transcript.strip():
        return []

    try:
        import openai
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenAI package not installed. Run: pip install openai"
        )

    api_key = config.openai_api_key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured"
        )

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": COMMAND_PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": transcript}
            ],
            temperature=0,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()

        result = json.loads(result_text)

        # Handle new format: {"commands": [...]}
        commands = result.get("commands", [])

        # Backward compatibility: handle old format {"command": "...", "params": {...}}
        if not commands and result.get("command"):
            commands = [{"command": result["command"], "params": result.get("params", {})}]

        return commands

    except json.JSONDecodeError as e:
        print(f"[Voice] Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        print(f"[Voice] LLM parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Command parsing failed: {e}")


async def transcribe_audio(audio_data: bytes, format: str = "wav") -> str:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_data: Raw audio bytes
        format: Audio format (wav, mp3, webm)

    Returns:
        Transcribed text
    """
    try:
        import openai
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenAI package not installed. Run: pip install openai"
        )

    api_key = config.openai_api_key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured"
        )

    client = openai.OpenAI(api_key=api_key)

    # Write to temp file (Whisper API requires file)
    suffix = f".{format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return transcript.text
    finally:
        os.unlink(temp_path)


# Endpoints

def execute_single_command(cmd: Dict[str, Any], command_bridge: CommandBridge) -> Dict[str, Any]:
    """
    Execute a single command and return result.

    Returns dict with 'command', 'params', 'executed', 'message'.
    """
    cmd_type = cmd.get("command")
    params = cmd.get("params", {})

    result = {
        "command": cmd_type,
        "params": params,
        "executed": True,
        "message": ""
    }

    if cmd_type == "pause":
        command_bridge.send_command(Command(CommandType.PAUSE))
        result["message"] = "Pausing execution"

    elif cmd_type == "continue":
        command_bridge.send_command(Command(CommandType.CONTINUE))
        result["message"] = "Continuing execution"

    elif cmd_type == "quit":
        command_bridge.send_command(Command(CommandType.QUIT))
        result["message"] = "Quitting execution"

    elif cmd_type == "set_goal":
        goal_name = params.get("goal_name")

        # Check execution state to determine action
        try:
            from ..main import get_planner_adapter, get_state_manager as _get_sm
        except ImportError:
            from main import get_planner_adapter, get_state_manager as _get_sm

        adapter = get_planner_adapter()
        state_manager = _get_sm()

        if adapter is not None and adapter.is_initialized() and not adapter.is_running():
            # Execution not running at all - start new execution with this goal
            success = adapter.start_execution(goal_name)
            if success:
                result["message"] = f"Starting execution with goal: {goal_name}"
            else:
                result["executed"] = False
                result["message"] = f"Failed to start execution with goal: {goal_name}"
        else:
            # Execution thread is alive - change goal
            command_bridge.send_command(Command(
                CommandType.CHANGE_GOAL,
                params={"goal_name": goal_name}
            ))

            # Check if execution is paused - if so, also send continue to resume
            current_state = state_manager.get_state()
            if current_state.execution_status == ExecutionStatus.PAUSED:
                command_bridge.send_command(Command(CommandType.CONTINUE))
                result["message"] = f"Changing goal to {goal_name} and resuming"
            else:
                result["message"] = f"Changing goal to {goal_name}"

    elif cmd_type == "action":
        action = params.get("action")
        block_id = params.get("block_id")
        if block_id:
            command_bridge.send_command(Command(
                CommandType.INJECT_ACTION,
                params={"action": action, "args": (block_id,)}
            ))
            result["message"] = f"Executing {action} on block {block_id}"
        else:
            result["executed"] = False
            result["message"] = f"Block ID not specified for {action}"

    else:
        result["executed"] = False
        result["message"] = f"Unknown command type: {cmd_type}"

    return result


@router.post("/command", response_model=VoiceCommandResponse)
async def process_voice_command(
    request: VoiceCommandRequest,
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """
    Process voice command from audio.

    1. Transcribe audio to text using OpenAI Whisper
    2. Parse text to identify command(s)
    3. Execute command(s) if recognized
    """
    try:
        # Decode audio
        audio_data = base64.b64decode(request.audio_base64)

        # Transcribe
        transcript = await transcribe_audio(audio_data, request.format)

        if not transcript:
            return VoiceCommandResponse(
                transcript="",
                commands=[],
                executed=False,
                message="No speech detected"
            )

        # Parse commands using LLM
        commands = await parse_command(transcript)

        if not commands:
            return VoiceCommandResponse(
                transcript=transcript,
                commands=[],
                executed=False,
                message="No command recognized"
            )

        # Execute each command in sequence
        executed_commands = []
        messages = []
        all_executed = True

        for cmd in commands:
            result = execute_single_command(cmd, command_bridge)
            executed_commands.append(result)
            messages.append(result["message"])
            if not result["executed"]:
                all_executed = False

        return VoiceCommandResponse(
            transcript=transcript,
            commands=executed_commands,
            executed=all_executed,
            message=" | ".join(messages)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_only(request: VoiceCommandRequest):
    """
    Transcribe audio without executing commands.
    Useful for testing or manual processing.
    """
    try:
        audio_data = base64.b64decode(request.audio_base64)
        transcript = await transcribe_audio(audio_data, request.format)

        return TranscriptionResponse(
            text=transcript,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def process_voice_upload(
    file: UploadFile = File(...),
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """
    Process voice command from uploaded file.
    Alternative to base64 encoding for larger files.
    """
    try:
        # Read file
        audio_data = await file.read()

        # Determine format from filename
        format = "wav"
        if file.filename:
            ext = file.filename.split(".")[-1].lower()
            if ext in ("mp3", "webm", "m4a", "ogg"):
                format = ext

        # Transcribe
        transcript = await transcribe_audio(audio_data, format)

        if not transcript:
            return VoiceCommandResponse(
                transcript="",
                commands=[],
                executed=False,
                message="No speech detected"
            )

        # Parse commands using LLM
        commands = await parse_command(transcript)

        if not commands:
            return VoiceCommandResponse(
                transcript=transcript,
                commands=[],
                executed=False,
                message="No command recognized"
            )

        # Execute each command in sequence
        executed_commands = []
        messages = []
        all_executed = True

        for cmd in commands:
            result = execute_single_command(cmd, command_bridge)
            executed_commands.append(result)
            messages.append(result["message"])
            if not result["executed"]:
                all_executed = False

        return VoiceCommandResponse(
            transcript=transcript,
            commands=executed_commands,
            executed=all_executed,
            message=" | ".join(messages)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
