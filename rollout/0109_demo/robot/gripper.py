"""
Gripper Control - Robotiq gripper interface.

Provides high-level gripper control abstraction.
"""

import sys
import os
from typing import Optional

# Add path to robotiq_gripper library
_collect_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'collect')
if os.path.isdir(_collect_path):
    sys.path.insert(0, _collect_path)

try:
    import robotiq_gripper
    HAS_GRIPPER = True
except ImportError:
    HAS_GRIPPER = False
    print("[Warning] Robotiq gripper library not available")


class Gripper:
    """
    High-level gripper controller for Robotiq grippers.

    Provides:
    - Connection management
    - Open/close control
    - Position queries
    """

    def __init__(self, config: dict, robot_ip: str = None):
        """
        Initialize gripper with config.

        Args:
            config: Configuration dict with gripper settings
            robot_ip: Robot IP (if not in config)
        """
        self.config = config
        gripper_cfg = config.get('gripper', {})

        self.robot_ip = robot_ip or config['robot']['ip']
        self.port = gripper_cfg.get('port', 63352)

        # Gripper settings
        self.open_pos = gripper_cfg.get('open_pos', 0)
        self.close_pos = gripper_cfg.get('close_pos', 255)
        self.speed = gripper_cfg.get('speed', 100)
        self.force = gripper_cfg.get('force', 50)
        self.open_wait = gripper_cfg.get('open_wait', 0.5)
        self.close_wait = gripper_cfg.get('close_wait', 1.5)
        self.open_margin = gripper_cfg.get('open_margin', 10)

        self._gripper = None
        self._connected = False
        self._min_pos = 0
        self._max_pos = 255

    def connect(self) -> bool:
        """
        Connect to gripper.

        Returns:
            True if connection succeeded
        """
        if not HAS_GRIPPER:
            print("[Warning] Gripper library not available, running without gripper")
            return False

        if self._connected:
            return True

        try:
            self._gripper = robotiq_gripper.RobotiqGripper()
            self._gripper.connect(self.robot_ip, self.port)
            self._gripper.activate()

            # Auto-calibrate to get actual range
            self._min_pos, self._max_pos = self._gripper.auto_calibrate()
            print(f"Gripper auto-calibrated to [{self._min_pos}, {self._max_pos}]")

            self._connected = True
            print("Gripper connected.")
            return True
        except Exception as e:
            print(f"Gripper connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect gripper."""
        if self._gripper:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
        self._connected = False
        self._gripper = None

    def is_connected(self) -> bool:
        """Check if gripper is connected."""
        return self._connected

    @property
    def gripper(self):
        """Get underlying gripper object (for primitives)."""
        return self._gripper

    def open(self, blocking: bool = True) -> bool:
        """
        Open gripper.

        Args:
            blocking: Wait for motion to complete

        Returns:
            True if successful
        """
        if not self._connected or not self._gripper:
            return False

        try:
            if blocking:
                self._gripper.move_and_wait_for_pos(
                    self.open_pos, self.speed, self.force
                )
            else:
                self._gripper.move(self.open_pos, self.speed, self.force)
            return True
        except Exception as e:
            print(f"Gripper open failed: {e}")
            return False

    def close(self, blocking: bool = True) -> bool:
        """
        Close gripper.

        Args:
            blocking: Wait for motion to complete

        Returns:
            True if successful
        """
        if not self._connected or not self._gripper:
            return False

        try:
            if blocking:
                self._gripper.move_and_wait_for_pos(
                    self.close_pos, self.speed, self.force
                )
            else:
                self._gripper.move(self.close_pos, self.speed, self.force)
            return True
        except Exception as e:
            print(f"Gripper close failed: {e}")
            return False

    def move_to(self, position: int, blocking: bool = True) -> bool:
        """
        Move gripper to specific position.

        Args:
            position: Target position (0-255)
            blocking: Wait for motion to complete

        Returns:
            True if successful
        """
        if not self._connected or not self._gripper:
            return False

        try:
            if blocking:
                self._gripper.move_and_wait_for_pos(
                    position, self.speed, self.force
                )
            else:
                self._gripper.move(position, self.speed, self.force)
            return True
        except Exception as e:
            print(f"Gripper move failed: {e}")
            return False

    def get_position(self) -> int:
        """
        Get current gripper position.

        Returns:
            Position (0-255) or -1 if not connected
        """
        if not self._connected or not self._gripper:
            return -1

        try:
            return self._gripper.get_current_position()
        except Exception:
            return -1

    def is_open(self) -> bool:
        """
        Check if gripper is open.

        Returns:
            True if gripper position is near open position
        """
        if not self._connected:
            return True  # Assume open if no gripper

        pos = self.get_position()
        if pos < 0:
            return True

        return pos <= (self.open_pos + self.open_margin)

    def is_closed(self) -> bool:
        """
        Check if gripper is closed.

        Returns:
            True if gripper position is near close position
        """
        if not self._connected:
            return False

        pos = self.get_position()
        if pos < 0:
            return False

        return pos >= (self.close_pos - self.open_margin)
