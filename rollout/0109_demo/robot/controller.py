"""
Robot Controller - RTDE interface wrapper for UR5e.

Provides high-level robot control abstraction.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation

try:
    import rtde_control
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    print("[Warning] RTDE libraries not available")


class RobotController:
    """
    High-level robot controller wrapping RTDE interfaces.

    Provides:
    - Connection management
    - Cartesian and joint motion
    - TCP pose and joint position queries
    """

    def __init__(self, config: dict):
        """
        Initialize controller with config.

        Args:
            config: Configuration dict with robot settings
        """
        self.config = config
        self.robot_ip = config['robot']['ip']
        self.speed = config['robot'].get('speed', 0.3)
        self.acceleration = config['robot'].get('acceleration', 0.3)
        self.init_pose = config['robot'].get('init_pose', None)

        self._rtde_c = None
        self._rtde_r = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to robot via RTDE.

        Returns:
            True if connection succeeded
        """
        if not HAS_RTDE:
            raise RuntimeError("RTDE libraries not available")

        if self._connected:
            return True

        try:
            print(f"Connecting to robot at {self.robot_ip}...")
            self._rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self._rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self._connected = True
            print("Robot connected.")
            return True
        except Exception as e:
            print(f"Robot connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from robot."""
        if self._rtde_c:
            try:
                self._rtde_c.stopScript()
            except Exception:
                pass
        self._connected = False
        self._rtde_c = None
        self._rtde_r = None

    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self._connected

    @property
    def rtde_c(self):
        """Get RTDE control interface (for primitives)."""
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return self._rtde_c

    @property
    def rtde_r(self):
        """Get RTDE receive interface (for primitives/perception)."""
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return self._rtde_r

    def get_tcp_pose(self) -> List[float]:
        """
        Get current TCP pose.

        Returns:
            [x, y, z, rx, ry, rz] in meters and axis-angle
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return self._rtde_r.getActualTCPPose()

    def get_joint_positions(self) -> List[float]:
        """
        Get current joint positions.

        Returns:
            [q0, q1, q2, q3, q4, q5] in radians
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return self._rtde_r.getActualQ()

    def move_to_pose(self, pose: List[float], speed: float = None, acceleration: float = None):
        """
        Move to Cartesian pose (linear motion).

        Args:
            pose: [x, y, z, rx, ry, rz] target pose
            speed: Motion speed (default: config value)
            acceleration: Motion acceleration (default: config value)
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")

        speed = speed or self.speed
        acceleration = acceleration or self.acceleration
        self._rtde_c.moveL(pose, speed, acceleration)

    def move_to_joints(self, joints: List[float], speed: float = None, acceleration: float = None):
        """
        Move to joint configuration.

        Args:
            joints: [q0, q1, q2, q3, q4, q5] target joints
            speed: Motion speed (default: config value)
            acceleration: Motion acceleration (default: config value)
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")

        speed = speed or self.speed
        acceleration = acceleration or self.acceleration
        self._rtde_c.moveJ(joints, speed, acceleration)

    def move_to_init_pose(self):
        """
        Move robot to initial observation pose from config.

        Pose format in config: [x, y, z, qx, qy, qz, qw] (meters + quaternion)
        """
        if self.init_pose is None:
            print("No init_pose configured, skipping.")
            return

        pose = self.init_pose
        x, y, z = pose[0], pose[1], pose[2]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]

        # Convert quaternion to axis-angle
        rot = Rotation.from_quat([qx, qy, qz, qw])
        rx, ry, rz = rot.as_rotvec()

        rtde_pose = [x, y, z, rx, ry, rz]
        print(f"Moving to init pose: xyz=[{x:.3f}, {y:.3f}, {z:.3f}]")

        self.move_to_pose(rtde_pose)
        print("Reached init pose.")

    def stop(self):
        """Stop current motion."""
        if self._connected and self._rtde_c:
            self._rtde_c.stopL(2.0)  # Deceleration
