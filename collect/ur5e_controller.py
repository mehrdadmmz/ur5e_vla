import rtde_control
import rtde_receive
import robotiq_gripper
import numpy as np
from utils import (
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_normalize
)


def is_identity_quaternion(q, tol=1e-6):
    """Check if quaternion is identity [0, 0, 0, 1]."""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    return np.allclose(q, identity, atol=tol)


class UR5E_CONTROLLER:

    def __init__(self, ip='192.168.56.101', gripper_port=63352, speed=0.4, acceleration=0.4):
        self.speed = speed
        self.acceleration = acceleration

        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)

        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(ip, gripper_port)
        self.gripper.activate()

    def move_to_target(self, target):
        """Move to target pose.

        Args:
            target: [x, y, z, qx, qy, qz, qw] - position + quaternion
        """
        target_xyz = np.array(target[:3])
        target_q = np.array(target[3:])

        target_axis_angle = quaternion_to_axis_angle(target_q)
        final_pose = np.concatenate([target_xyz, target_axis_angle])

        current_joints = self.rtde_r.getActualQ()
        if self.rtde_c.getInverseKinematicsHasSolution(final_pose, current_joints):
            target_joints = self.rtde_c.getInverseKinematics(final_pose, current_joints)
            self.rtde_c.moveJ(target_joints, speed=self.speed, acceleration=self.acceleration)
        else:
            print("IK solution not found for target pose:", final_pose)

    def move_delta(self, delta):
        """Move by a delta in position and orientation.

        Args:
            delta: [dx, dy, dz, dqx, dqy, dqz, dqw] - position delta + quaternion delta
                   The quaternion represents the rotation to apply to current orientation.
                   Use [0, 0, 0, 1] for no rotation change.
        """
        delta = np.array(delta)
        xyz_delta = delta[:3]
        q_delta = delta[3:]

        # Get current TCP pose [x, y, z, rx, ry, rz] (axis-angle)
        cur_pose = self.rtde_r.getActualTCPPose()
        cur_xyz = np.array(cur_pose[:3])
        cur_axis_angle = np.array(cur_pose[3:6])

        # Compute target position
        tar_xyz = cur_xyz + xyz_delta

        # Compute target orientation
        if is_identity_quaternion(q_delta):
            # No rotation change - keep current orientation
            tar_axis_angle = cur_axis_angle
        else:
            # Convert current orientation to quaternion
            cur_q = axis_angle_to_quaternion(cur_axis_angle)

            # Apply delta rotation: new_orientation = delta * current
            tar_q = quaternion_multiply(q_delta, cur_q)
            tar_q = quaternion_normalize(tar_q)

            # Convert back to axis-angle
            tar_axis_angle = quaternion_to_axis_angle(tar_q)

        # Build target pose
        tar_pose = np.concatenate([tar_xyz, tar_axis_angle])

        # Execute move
        cur_joints = self.rtde_r.getActualQ()
        if self.rtde_c.getInverseKinematicsHasSolution(tar_pose, cur_joints):
            tar_joints = self.rtde_c.getInverseKinematics(tar_pose, cur_joints)
            self.rtde_c.moveJ(tar_joints, self.speed, self.acceleration)
        else:
            raise RuntimeError(f"IK solution not found for target pose: {tar_pose}")
