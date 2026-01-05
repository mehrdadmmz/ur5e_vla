"""
Backend configuration for the Block World Planner UI.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendConfig:
    """Configuration for the FastAPI backend."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Robot config path
    robot_config_path: str = "../../config.yaml"

    # OpenAI API key for voice processing
    openai_api_key: Optional[str] = None

    # CORS settings
    cors_origins: list = None

    # WebSocket settings
    ws_heartbeat_interval: float = 30.0  # seconds
    state_broadcast_interval: float = 0.2  # 5 Hz

    # Static files (frontend)
    static_dir: Optional[str] = None
    serve_frontend: bool = True

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Allow all in development

        # Get OpenAI key from environment if not set
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        # Set default static dir
        if self.static_dir is None:
            self.static_dir = os.path.join(
                os.path.dirname(__file__),
                "..", "frontend", "dist"
            )

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("UI_HOST", "0.0.0.0"),
            port=int(os.environ.get("UI_PORT", "8000")),
            robot_config_path=os.environ.get("ROBOT_CONFIG", "../../config.yaml"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )


# Global config instance
config = BackendConfig()
