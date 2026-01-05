"""
FastAPI backend for Block World Planner UI.

Provides:
- WebSocket endpoint for real-time state updates
- REST API for control commands
- Voice command processing endpoint
- Static file serving for React frontend

Usage:
    # Standalone mode (for development):
    python main.py --config ../../config.yaml

    # As module:
    from user_interface.backend.main import create_app, run_server
    app = create_app(state_manager, command_bridge)
    run_server(app)
"""

import os
import sys
import asyncio
import argparse
from contextlib import asynccontextmanager
from typing import Optional

# Fix module import issue: when running as __main__, also register as 'main'
# so that routes can import from either name and get the same globals
if __name__ == "__main__":
    sys.modules['main'] = sys.modules['__main__']

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle both module and direct execution imports
try:
    from .config import BackendConfig, config
    from .state_manager import StateManager, ExecutionStatus
    from .command_bridge import CommandBridge
    from .websocket_handler import ConnectionManager, WebSocketHandler, broadcast_loop
    from .routes import control, state, voice, system
except ImportError:
    from config import BackendConfig, config
    from state_manager import StateManager, ExecutionStatus
    from command_bridge import CommandBridge
    from websocket_handler import ConnectionManager, WebSocketHandler, broadcast_loop
    from routes import control, state, voice, system


# Global instances (set during app creation)
_state_manager: Optional[StateManager] = None
_command_bridge: Optional[CommandBridge] = None
_connection_manager: Optional[ConnectionManager] = None
_ws_handler: Optional[WebSocketHandler] = None
_broadcast_task: Optional[asyncio.Task] = None
_update_queue: Optional[asyncio.Queue] = None
_planner_adapter = None  # Type: Optional[PlannerAdapter] - lazy import
_robot_config: Optional[dict] = None


def get_state_manager() -> StateManager:
    """Dependency: Get the StateManager instance."""
    if _state_manager is None:
        raise RuntimeError("StateManager not initialized")
    return _state_manager


def get_command_bridge() -> CommandBridge:
    """Dependency: Get the CommandBridge instance."""
    if _command_bridge is None:
        raise RuntimeError("CommandBridge not initialized")
    return _command_bridge


def get_connection_manager() -> ConnectionManager:
    """Dependency: Get the ConnectionManager instance."""
    if _connection_manager is None:
        raise RuntimeError("ConnectionManager not initialized")
    return _connection_manager


def get_planner_adapter():
    """Dependency: Get the PlannerAdapter instance (may be None if not initialized)."""
    return _planner_adapter


def get_robot_config() -> Optional[dict]:
    """Get the robot configuration dictionary."""
    return _robot_config


def initialize_planner_adapter() -> bool:
    """
    Initialize the PlannerAdapter and connect to robot/camera.

    Returns:
        True if initialization succeeded
    """
    global _planner_adapter, _robot_config

    if _state_manager is None or _command_bridge is None:
        raise RuntimeError("StateManager and CommandBridge must be initialized first")

    # Check if already initialized
    if _planner_adapter is not None and _planner_adapter.is_initialized():
        return True

    # Load robot config if not already loaded
    if _robot_config is None:
        import yaml
        config_path = config.robot_config_path

        # Resolve relative path from backend directory
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

        if not os.path.exists(config_path):
            _state_manager.add_log("error", "system", f"Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            _robot_config = yaml.safe_load(f)

        _state_manager.add_log("info", "system", f"Loaded config from {config_path}")

    # Create and initialize planner adapter
    try:
        from planner_adapter import PlannerAdapter
    except ImportError:
        try:
            from .planner_adapter import PlannerAdapter
        except ImportError:
            _state_manager.add_log("error", "system", "Failed to import PlannerAdapter")
            return False

    _planner_adapter = PlannerAdapter(_robot_config, _state_manager, _command_bridge)
    return _planner_adapter.initialize()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI startup/shutdown.
    """
    global _broadcast_task, _update_queue

    print("[Backend] Starting up...")

    # Create async queue for state updates
    _update_queue = asyncio.Queue(maxsize=100)

    # Connect StateManager to async context
    if _state_manager:
        _state_manager.set_async_context(_update_queue, asyncio.get_event_loop())

    # Start broadcast task
    if _connection_manager:
        _broadcast_task = asyncio.create_task(
            broadcast_loop(_update_queue, _connection_manager)
        )

    print("[Backend] Ready to accept connections")

    yield

    # Cleanup
    print("[Backend] Shutting down...")
    if _broadcast_task:
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass


def create_app(
    state_manager: StateManager,
    command_bridge: CommandBridge,
    backend_config: BackendConfig = None
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        state_manager: StateManager instance for state coordination
        command_bridge: CommandBridge instance for command handling
        backend_config: Optional backend configuration

    Returns:
        Configured FastAPI application
    """
    global _state_manager, _command_bridge, _connection_manager, _ws_handler

    # Store instances
    _state_manager = state_manager
    _command_bridge = command_bridge
    _connection_manager = ConnectionManager()
    _ws_handler = WebSocketHandler(state_manager, command_bridge, _connection_manager)

    # Use provided config or global
    cfg = backend_config or config

    # Create app
    app = FastAPI(
        title="Block World Robot Control API",
        description="FastAPI backend for UR5e block manipulation UI",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(control.router, prefix="/api/control", tags=["control"])
    app.include_router(state.router, prefix="/api/state", tags=["state"])
    app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
    app.include_router(system.router, prefix="/api/system", tags=["system"])

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await _connection_manager.connect(websocket)

        try:
            # Send initial state
            initial_state = state_manager.get_state_dict()
            await websocket.send_json({
                "type": "state_update",
                "data": initial_state
            })

            # Send recent logs
            logs = state_manager.get_logs(50)
            await websocket.send_json({
                "type": "logs",
                "data": logs
            })

            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await _ws_handler.handle_message(websocket, data)

        except WebSocketDisconnect:
            _connection_manager.disconnect(websocket)
        except Exception as e:
            print(f"[WebSocket] Error: {e}")
            _connection_manager.disconnect(websocket)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "connections": _connection_manager.connection_count if _connection_manager else 0,
            "execution_status": state_manager.get_state().execution_status.value
        }

    # Serve frontend static files if available
    if cfg.serve_frontend and cfg.static_dir and os.path.isdir(cfg.static_dir):
        # Serve index.html for SPA routing
        @app.get("/")
        async def serve_frontend():
            index_path = os.path.join(cfg.static_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Frontend not built. Run 'npm run build' in frontend directory."}

        # Mount static files
        app.mount("/assets", StaticFiles(directory=os.path.join(cfg.static_dir, "assets")), name="assets")

        # Catch-all for SPA routing
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            # Try to serve the file directly
            file_path = os.path.join(cfg.static_dir, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # Otherwise serve index.html for SPA routing
            index_path = os.path.join(cfg.static_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Not found"}

    return app


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.

    Args:
        app: FastAPI application
        host: Host to bind to
        port: Port to listen on
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# =============================================================================
# Standalone mode - for development/testing
# =============================================================================

def main():
    """Main entry point for standalone mode."""
    parser = argparse.ArgumentParser(description="Block World Planner UI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--config", default="../../config.yaml", help="Robot config path")
    args = parser.parse_args()

    print("=" * 60)
    print("Block World Planner UI Backend")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print("=" * 60)

    # Create standalone instances
    state_manager = StateManager()
    command_bridge = CommandBridge(state_manager, enable_keyboard=False)

    # Update config
    config.robot_config_path = args.config

    # Create and run app
    app = create_app(state_manager, command_bridge, config)

    print("\n[Backend] Starting server...")
    print(f"[Backend] API docs: http://{args.host}:{args.port}/docs")
    print(f"[Backend] WebSocket: ws://{args.host}:{args.port}/ws")
    print()

    run_server(app, args.host, args.port)


if __name__ == "__main__":
    main()
