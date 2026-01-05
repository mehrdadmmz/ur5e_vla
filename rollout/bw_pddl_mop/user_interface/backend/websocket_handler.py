"""
WebSocket connection manager for real-time UI communication.

Handles:
- Multiple WebSocket connections
- Broadcasting state updates to all clients
- Processing incoming commands from clients
"""

import asyncio
import json
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

try:
    from .state_manager import StateManager
    from .command_bridge import CommandBridge, Command, CommandType
except ImportError:
    from state_manager import StateManager
    from command_bridge import CommandBridge, Command, CommandType


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.

    Usage:
        manager = ConnectionManager()

        # In WebSocket endpoint:
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                await manager.handle_message(websocket, data)
        except WebSocketDisconnect:
            manager.disconnect(websocket)

        # Broadcasting:
        await manager.broadcast({"type": "state_update", "data": {...}})
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        print(f"[WebSocket] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients.

        Args:
            message: JSON-serializable message dict
        """
        if not self.active_connections:
            return

        # Serialize once
        try:
            json_message = json.dumps(message)
        except (TypeError, ValueError) as e:
            print(f"[WebSocket] Failed to serialize message: {e}")
            return

        # Send to all clients
        disconnected = set()
        async with self._lock:
            for websocket in self.active_connections:
                try:
                    await websocket.send_text(json_message)
                except Exception:
                    disconnected.add(websocket)

            # Clean up disconnected clients
            self.active_connections -= disconnected

    async def send_to(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific client.

        Args:
            websocket: Target WebSocket connection
            message: JSON-serializable message dict

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            self.disconnect(websocket)
            return False

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self.active_connections)

    def has_connections(self) -> bool:
        """Check if there are any active connections."""
        return len(self.active_connections) > 0


class WebSocketHandler:
    """
    Handles WebSocket message processing and command routing.

    Integrates ConnectionManager with StateManager and CommandBridge
    for bidirectional communication.
    """

    def __init__(self, state_manager: StateManager,
                 command_bridge: CommandBridge,
                 connection_manager: ConnectionManager):
        self.state_manager = state_manager
        self.command_bridge = command_bridge
        self.connection_manager = connection_manager

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """
        Process an incoming WebSocket message.

        Expected message format:
        {
            "type": "command",
            "command": "pause" | "continue" | "quit" | "set_goal" | "execute_action",
            "params": {...}  # Optional parameters
        }
        """
        msg_type = data.get("type")

        if msg_type == "command":
            await self._handle_command(websocket, data)
        elif msg_type == "ping":
            await self.connection_manager.send_to(websocket, {"type": "pong"})
        elif msg_type == "get_state":
            state = self.state_manager.get_state_dict()
            await self.connection_manager.send_to(websocket, {
                "type": "state_update",
                "data": state
            })
        elif msg_type == "get_logs":
            limit = data.get("limit", 100)
            logs = self.state_manager.get_logs(limit)
            await self.connection_manager.send_to(websocket, {
                "type": "logs",
                "data": logs
            })
        else:
            await self.connection_manager.send_to(websocket, {
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_command(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Process a command message."""
        command_str = data.get("command", "").lower()
        params = data.get("params", {})

        response = {"type": "command_response", "command": command_str}

        try:
            if command_str == "pause":
                self.command_bridge.send_command(Command(CommandType.PAUSE))
                response["success"] = True
                response["message"] = "Pause requested"

            elif command_str == "continue":
                self.command_bridge.send_command(Command(CommandType.CONTINUE))
                response["success"] = True
                response["message"] = "Continue requested"

            elif command_str == "quit":
                self.command_bridge.send_command(Command(CommandType.QUIT))
                response["success"] = True
                response["message"] = "Quit requested"

            elif command_str == "set_goal":
                goal_name = params.get("goal_name")
                goal_predicates = params.get("goal_predicates")
                if goal_name:
                    self.command_bridge.send_command(Command(
                        CommandType.CHANGE_GOAL,
                        params={"goal_name": goal_name, "goal_predicates": goal_predicates}
                    ))
                    response["success"] = True
                    response["message"] = f"Goal change to '{goal_name}' requested"
                else:
                    response["success"] = False
                    response["message"] = "Missing goal_name parameter"

            elif command_str == "execute_action":
                action_name = params.get("action")
                action_args = params.get("args", [])
                if action_name:
                    self.command_bridge.send_command(Command(
                        CommandType.INJECT_ACTION,
                        params={"action": action_name, "args": tuple(action_args)}
                    ))
                    response["success"] = True
                    response["message"] = f"Action '{action_name}' queued"
                else:
                    response["success"] = False
                    response["message"] = "Missing action parameter"

            elif command_str == "replan":
                self.command_bridge.send_command(Command(CommandType.REPLAN))
                response["success"] = True
                response["message"] = "Replan requested"

            else:
                response["success"] = False
                response["message"] = f"Unknown command: {command_str}"

        except Exception as e:
            response["success"] = False
            response["message"] = f"Error processing command: {str(e)}"

        await self.connection_manager.send_to(websocket, response)


async def broadcast_loop(queue: asyncio.Queue,
                        connection_manager: ConnectionManager) -> None:
    """
    Coroutine that broadcasts state updates from queue to WebSocket clients.

    This runs continuously, pulling messages from the async queue
    (fed by StateManager) and broadcasting to all connected clients.
    """
    while True:
        try:
            message = await queue.get()
            await connection_manager.broadcast(message)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WebSocket] Broadcast error: {e}")
