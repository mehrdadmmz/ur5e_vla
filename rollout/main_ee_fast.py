import dataclasses
import logging
import numpy
import queue
import collections
import time
import cv2
import pyrealsense2 as rs
import numpy as np

from openpi_client import action_chunk_broker, image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

import rtde_control
import rtde_receive
import robotiq_gripper


def move_delta( rtde_c, rtde_r, xyz = [0,0,0], q = [0,0,0,1]):
    # ignore q for now
    

    # 1. 读取当前 TCP 位姿（6维）与关节角
    cur_pose_full = rtde_r.getActualTCPPose()  # [x,y,z,rx,ry,rz]
    cur_j = rtde_r.getActualQ()                # 当前关节角

    # 2. 构建目标 TCP 位姿：平移 + 保持姿态
    tar_pose = [
        cur_pose_full[0] + xyz[0],
        cur_pose_full[1] + xyz[1],
        cur_pose_full[2] + xyz[2],
        cur_pose_full[3],
        cur_pose_full[4],
        cur_pose_full[5]
    ]

    # 3. 求解逆向运动学（IK），使用 cur_j 作为 initial guess
    if rtde_c.getInverseKinematicsHasSolution(tar_pose, cur_j):
        tar_j = rtde_c.getInverseKinematics(tar_pose, cur_j)
    else:
        raise RuntimeError(f"目标位姿 {tar_pose} 无法 IK 求解")

    # 4. 使用 servoJ 实时驱动机器人到目标关节角
    rtde_c.moveJ(tar_j, 0.4,0.4)



# organize code structures

def quaternion_to_euler(qx, qy, qz, qw):
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    roll = (roll + np.pi) % (2 * np.pi) - np.pi
    pitch = (pitch + np.pi) % (2 * np.pi) - np.pi
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    return roll, pitch, yaw

def euler_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """Convert Euler angles (Roll, Pitch, Yaw) to quaternion.
    
    Args:
        rpy: 3-element array containing [roll, pitch, yaw] in radians
        
    Returns:
        4-element array [qx, qy, qz, qw] representing the quaternion
    """
    roll, pitch, yaw = rpy
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return np.array([qx, qy, qz, qw])

def axis_angle_to_quaternion(angle, axis):
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    
    axis_normalized = axis / norm
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    
    return np.array([
        axis_normalized[0] * sin_half,
        axis_normalized[1] * sin_half,
        axis_normalized[2] * sin_half,
        np.cos(half_angle)
    ])

def quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation.
    
    Args:
        q: 4-element array [qx, qy, qz, qw] representing the quaternion
        
    Returns:
        3-element array representing the rotation axis multiplied by angle (rx, ry, rz)
    """
    # Normalize quaternion
    q_norm = q / np.linalg.norm(q)
    qx, qy, qz, qw = q_norm
    
    angle = 2 * np.arccos(qw)
    sin_angle = np.sin(angle/2)
    
    if sin_angle < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    
    axis = np.array([qx, qy, qz]) / sin_angle
    return axis * angle

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product).
    
    Args:
        q1: First quaternion [qx, qy, qz, qw]
        q2: Second quaternion [qx, qy, qz, qw]
        
    Returns:
        Product quaternion q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])



@dataclasses.dataclass
class Args:
    host:               str = "0.0.0.0"
    port:               int = 8000
    robot_ip:           str = '192.168.56.101'
    action_horizon:     int = 28 #23

    # num_episodes:       int = 1
    # max_episode_steps:  int = 1000

    speed:              float = 0.8  # rad/s
    acceleration:       float = 1.0  # rad/s^2
    dt:                 float = 0.008  # Time step in seconds
    lookahead_time:     float = 0.1  # Seconds
    gain:               float= 300  # Proportional gain

class CameraSystem:
    # right 344422071118
    # base 337322072205
    # left 346122072239
    def __init__(self, serial_numbers=["337322072205", "243322072780"]):
        self.pipelines = []
        self.configs = []
        
        for serial in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self.pipelines.append(pipeline)
            self.configs.append(config)
        
        for pipeline, config in zip(self.pipelines, self.configs):
            pipeline.start(config)
    
    def get_frames(self):
        frames = []
        images = []
        
        for i, pipeline in enumerate(self.pipelines):
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            frames.append(color_frame)
            
            if color_frame:
                img_rgb = np.asanyarray(color_frame.get_data())

                # If this is the second photo, crop the left 1/3
                if i == 2:
                    height, width, _ = img_rgb.shape
                    crop_x = width // 3  # Start cropping from 1/3 width
                    img_rgb = img_rgb[:, crop_x:]  # Keep right 2/3 of the image

                # Resize and pad after cropping
                img2_rgb = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_rgb, 224, 224)
                )
                images.append(img2_rgb)
            else:
                images.append(None)

        return frames, images

    
    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

def get_input_elements(rtde_r, gripper, camera_images):

    input_elements = {
        "observation/image": camera_images[0],
        "observation/wrist_image": camera_images[1],
        "observation/state": np.zeros(7),  # tcp, euler + gripper
        # "prompt": "grasp white binder from rack."
        "prompt": "grasp pink pencilcase from drawer",
        #"prompt": "grasp pink pencilcase from drawer",
        # "prompt": "place binder on rack",
        # "prompt": "grasp paper from printer",
        # "prompt": "pick black stapler from table",
        # "prompt": "tap id card",
        # "prompt": "stack green box on red box"
        #"prompt": "pick pink can on the table",
    }

    # Get TCP pose and convert to quaternion then Euler angles
    tcp_pose = rtde_r.getActualTCPPose()  # x,y,z,rx,ry,rz
    angle = np.linalg.norm(tcp_pose[3:6])
    axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-8 else tcp_pose[3:6] / angle
    
    tcp_q = axis_angle_to_quaternion(angle, axis)
    tcp_euler = quaternion_to_euler(*tcp_q)   

    # Get gripper status (0 for open, 1 for closed)
    gripper_status_in = 0 if gripper.get_current_position() < 76 else 1
    gripper_status_in = 0 
    
    # Combine position, orientation and gripper status
    input_elements["observation/state"] = np.concatenate([
        tcp_pose[:3],  # x,y,z position
        tcp_euler,     # roll, pitch, yaw
        [gripper_status_in]  # gripper state
    ])
    
    return input_elements

def main(args: Args) -> None:
    # Initialize components
    
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    print("Resetting Robot...")
    # gpfp
    # rtde_c.moveJ([3.14, -2.315, 2.355, 0.419, 1.506, 3.155], 0.5, 0.5)

    # gstb
    # rtde_c.moveJ([3.925, -2.315, 2.355, 0.419, 1.506, 3.155], 0.5, 0.5)
    #rtde_c.moveJ(np.array([225, -122, 100, -103, -80, -7])*0.076, 0.5, 0.5)

    #gstw
    # rtde_c.moveJ([4.26, -1.26, 1.10, 1.35, 1.35, 2.83], 0.5, 0.5)

    # grasp binder
    # rtde_c.moveJ(np.array([267, -75, 130, -53, 90, 180])*3.1415/180, 0.5, 0.5)

    # new up_down pick stapler
    # rtde_c.moveJ(np.array([89, -120, -40, -107, 88, -5])*3.1415/180, 0.5, 0.5)

    # tap id:
    # initial_pos = np.array([220, -45, 115, -70, 45, 0])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    # new gpfp
    # initial_pos = np.array([220, -45, 115, -70, 45, 90])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    #right gpfp
    # initial_pos = np.array([60, -45, 90, -70, -10, 90])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    # place binder:
    # initial_pos = np.array([255, -94, 56, -54, -92, -10 ])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    # # new tap id
    # initial_pos = np.array([220, -45, 115, -70, 45, 90])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    # grasp pencil case
    # initial_pos = np.array([90, -70, 90, -110, -90, -90])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    # # home
    # initial_pos = np.array([177, -134, 124, 11, 90, 187])*3.1415/180
    # rtde_c.moveJ(initial_pos, 0.5, 0.5)

    #sajjad - canssss
    rtde_c.moveL([0.0, 0.45, 0.50, 3.14, 0.0, 0.0], 0.3, 0.3)


    print("Starting the Gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(args.robot_ip, 63352)
    gripper.activate()

    time.sleep(1)
    gripper.move_and_wait_for_pos(0,255,255)

    print("Connecting websocket...")
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    print("Connecting Realsenses...")
    camera_system = CameraSystem()

    print("Preparing to start...")
    time.sleep(1)

    gripper_status = 0 if gripper.get_current_position() < 200 else 1

    action_plan = collections.deque()
    action_chunk = []

    try:
        while True:
            # Get camera frames and show them
            _, camera_images = camera_system.get_frames()
            if camera_images[0] is not None and camera_images[1] is not None:
                cv2.imshow('Camera 1', camera_images[0])
                cv2.imshow('Camera 2', camera_images[1])
                cv2.waitKey(1)
            

            if not action_plan:
                input_elements = get_input_elements(rtde_r, gripper, camera_images)
                action_chunk = ws_client_policy.infer(input_elements)["actions"]
                action_plan.extend(action_chunk[:args.action_horizon])
                # print(action_chunk)
            print(action_plan)
        

            

            
            current_tcp = rtde_r.getActualTCPPose()
            current_position = current_tcp[:3]
            
            current_angle = np.linalg.norm(current_tcp[3:6])
            current_axis = (current_tcp[3:6] / current_angle 
                            if current_angle > 1e-8 
                            else np.array([1.0, 0.0, 0.0]))
            current_orientation_q = axis_angle_to_quaternion(current_angle, current_axis)


            final_gripper_cmd = max([x[-1] for x in action_plan])      # Gripper command (0=open, 1=close)
            final_gripper_cmd = action_plan[-1][-1]
            # final_gripper_cmd = action_chunk[-15][-1]

            # print(final_gripper_cmd)

            target_position, target_orientation_q = current_position, current_orientation_q
            while action_plan:
                cur_action = action_plan.popleft()  # 1x7 array: [tcp_delta_xyz, tcp_delta_rpy, gripper_cmd]

                position_delta = cur_action[:3]    # XYZ position delta
                euler_delta = cur_action[3:6]      # RPY orientation delta
                orientation_delta_q = euler_to_quaternion(euler_delta)

                target_position += position_delta
                target_orientation_q = quaternion_multiply(orientation_delta_q, target_orientation_q)
            final_pose = np.concatenate([target_position, quaternion_to_axis_angle(target_orientation_q)])
            

            # Handle gripper command if state changed
            target_gripper_state = 0 if final_gripper_cmd < 0.5 else 1
            if target_gripper_state != gripper_status:

                print("Gripper state changed!!!!\n\n")
                gripper_position = 0 if target_gripper_state == 0 else 255
                if target_gripper_state != 0:
                    # move_delta(rtde_c, rtde_r, [0.00, 0.00, -0.00])
                    # move_delta(rtde_c, rtde_r, [0.00, 0.00, -0.035]) # for pick cups
                    move_delta(rtde_c, rtde_r, [0.00, 0.00, +0.0]) # for stack boxes
                    time.sleep(1)
                # else:
                #     move_delta(rtde_c, rtde_r, [0.00, -0.00, -0.005])
                #     time.sleep(0.5)

                gripper.move_and_wait_for_pos(gripper_position, 125, 125)
                gripper_status = 0 if gripper.get_current_position() < 76 else 1


            # Check and execute inverse kinematics
            current_joints = rtde_r.getActualQ()
            if rtde_c.getInverseKinematicsHasSolution(final_pose, current_joints):
                target_joints = rtde_c.getInverseKinematics(final_pose, current_joints)
                rtde_c.moveJ(target_joints, speed=0.8, acceleration=0.8)
            else:
                print("IK solution not found for target pose:", final_pose)

    except KeyboardInterrupt:
        print("Shutting down...")
        
    finally:
        camera_system.stop()
        rtde_c.servoStop()
        rtde_c.stopScript()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)