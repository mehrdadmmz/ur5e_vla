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
from typing import Optional, Dict, Any
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..state_manager import StateManager
    from ..command_bridge import CommandBridge, Command, CommandType
    from ..config import config
except ImportError:
    from state_manager import StateManager
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
    command: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    executed: bool = False
    message: str


# LLM prompt for command parsing
COMMAND_PARSE_SYSTEM_PROMPT = """You are a command parser for a robot block manipulation system. Parse the user's voice command into a structured action.

Available commands:

1. Control commands:
   - "pause" - pause robot execution
   - "continue" - resume robot execution
   - "quit" - stop and exit

2. Goal commands (set_goal):
   - "bridge" - build a bridge structure (two pillars with a plank on top)
   - "tall_bridge" - build a tall bridge (double-height pillars)
   - "tower" - build a vertical stack of 4 blocks
   - "cn_tower" - build a 5-block tall tower
   - "house" - build a house structure

3. Action commands:
   - "pick-up" - pick up a block (requires block_id)
   - "put-down" - place block on another (requires block_id)
   - "release" - release/drop held block

Respond with JSON only. Format:
{"command": "<command_type>", "params": {<parameters>}}

For control commands: {"command": "pause", "params": {}}
For goals: {"command": "set_goal", "params": {"goal_name": "bridge"}}
For actions: {"command": "action", "params": {"action": "pick-up", "block_id": "5"}}

If the command is unclear or not related to robot control, respond:
{"command": null, "params": {}}

Examples:
- "stop the robot" -> {"command": "pause", "params": {}}
- "build me a tower" -> {"command": "set_goal", "params": {"goal_name": "tower"}}
- "grab block 3" -> {"command": "action", "params": {"action": "pick-up", "block_id": "3"}}
- "make a house please" -> {"command": "set_goal", "params": {"goal_name": "house"}}
- "what's the weather" -> {"command": null, "params": {}}
"""


async def parse_command(transcript: str) -> Optional[tuple]:
    """
    Parse transcript into a command using LLM.

    Returns:
        Tuple of (command_type, params) or None if not recognized
    """
    if not transcript or not transcript.strip():
        return None

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
            max_tokens=150
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

        command = result.get("command")
        params = result.get("params", {})

        if command is None:
            return None

        return (command, params)

    except json.JSONDecodeError as e:
        print(f"[Voice] Failed to parse LLM response as JSON: {e}")
        return None
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

@router.post("/command", response_model=VoiceCommandResponse)
async def process_voice_command(
    request: VoiceCommandRequest,
    command_bridge: CommandBridge = Depends(get_command_bridge)
):
    """
    Process voice command from audio.

    1. Transcribe audio to text using OpenAI Whisper
    2. Parse text to identify command
    3. Execute command if recognized
    """
    try:
        # Decode audio
        audio_data = base64.b64decode(request.audio_base64)

        # Transcribe
        transcript = await transcribe_audio(audio_data, request.format)

        if not transcript:
            return VoiceCommandResponse(
                transcript="",
                executed=False,
                message="No speech detected"
            )

        # Parse command using LLM
        result = await parse_command(transcript)

        if result is None:
            return VoiceCommandResponse(
                transcript=transcript,
                executed=False,
                message="Command not recognized"
            )

        cmd_type, params = result

        # Execute command
        executed = True
        message = ""

        if cmd_type == "pause":
            command_bridge.send_command(Command(CommandType.PAUSE))
            message = "Pausing execution"

        elif cmd_type == "continue":
            command_bridge.send_command(Command(CommandType.CONTINUE))
            message = "Continuing execution"

        elif cmd_type == "quit":
            command_bridge.send_command(Command(CommandType.QUIT))
            message = "Quitting execution"

        elif cmd_type == "set_goal":
            goal_name = params.get("goal_name")

            # Check if execution is running - if so, change goal; if not, start execution
            try:
                from ..main import get_planner_adapter
            except ImportError:
                from main import get_planner_adapter

            adapter = get_planner_adapter()

            if adapter is not None and adapter.is_initialized() and not adapter.is_running():
                # Start execution with this goal
                success = adapter.start_execution(goal_name)
                if success:
                    message = f"Starting execution with goal: {goal_name}"
                else:
                    executed = False
                    message = f"Failed to start execution with goal: {goal_name}"
            else:
                # Execution already running, just change goal
                command_bridge.send_command(Command(
                    CommandType.CHANGE_GOAL,
                    params={"goal_name": goal_name}
                ))
                message = f"Changing goal to {goal_name}"

        elif cmd_type == "action":
            action = params.get("action")
            block_id = params.get("block_id")
            if block_id:
                command_bridge.send_command(Command(
                    CommandType.INJECT_ACTION,
                    params={"action": action, "args": (block_id,)}
                ))
                message = f"Executing {action} on block {block_id}"
            else:
                executed = False
                message = f"Block ID not specified for {action}"

        else:
            executed = False
            message = f"Unknown command type: {cmd_type}"

        return VoiceCommandResponse(
            transcript=transcript,
            command=cmd_type,
            params=params,
            executed=executed,
            message=message
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
                executed=False,
                message="No speech detected"
            )

        # Parse and execute using LLM
        result = await parse_command(transcript)

        if result is None:
            return VoiceCommandResponse(
                transcript=transcript,
                executed=False,
                message="Command not recognized"
            )

        cmd_type, params = result

        # Same execution logic as /command endpoint
        # (simplified for brevity)
        executed = False
        message = f"Recognized: {cmd_type}"

        if cmd_type in ("pause", "continue", "quit"):
            cmd_map = {
                "pause": CommandType.PAUSE,
                "continue": CommandType.CONTINUE,
                "quit": CommandType.QUIT,
            }
            command_bridge.send_command(Command(cmd_map[cmd_type]))
            executed = True
            message = f"Executing: {cmd_type}"

        elif cmd_type == "set_goal":
            goal_name = params.get('goal_name')

            # Check if execution is running - if so, change goal; if not, start execution
            try:
                from ..main import get_planner_adapter
            except ImportError:
                from main import get_planner_adapter

            adapter = get_planner_adapter()

            if adapter is not None and adapter.is_initialized() and not adapter.is_running():
                # Start execution with this goal
                success = adapter.start_execution(goal_name)
                if success:
                    executed = True
                    message = f"Starting execution with goal: {goal_name}"
                else:
                    executed = False
                    message = f"Failed to start execution with goal: {goal_name}"
            else:
                # Execution already running, just change goal
                command_bridge.send_command(Command(
                    CommandType.CHANGE_GOAL,
                    params=params
                ))
                executed = True
                message = f"Changing goal to: {goal_name}"

        return VoiceCommandResponse(
            transcript=transcript,
            command=cmd_type,
            params=params,
            executed=executed,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
