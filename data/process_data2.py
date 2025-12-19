import rosbag2_py
from rclpy.serialization import deserialize_message
import csv
import bisect
import os
import re
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d  
import os
from pathlib import Path




def extract_gripper_status(data_str):
    match = re.search(
        r'(?:.*?\b)?Open:\s*(True|False)\s*Closed:\s*(True|False)\b', 
        data_str, 
        re.IGNORECASE
    )
    if not match:
        return -1

    open_state, closed_state = match.groups()
    open_state = open_state.lower() == "true"
    closed_state = closed_state.lower() == "true"

    if open_state: 
        return 0  # open
    else:
        return 1  # closed

def extract_aligned_data(bag_path="data_0", 
                        image_folder1="image_folder1", 
                        image_folder2="image_folder2", 
                        output_csv="aligned_data.csv"):
    # Create output directories
    os.makedirs(image_folder1, exist_ok=True)
    os.makedirs(image_folder2, exist_ok=True)
    bridge = CvBridge()

    # Open ROS2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Initialize variables to store latest messages
    latest_data = {
        "image1": None,
        "image2": None,
        "joint_state": None,
        "tcp_pose": None,
        "gripper_status": None
    }

    timest = None
    timestep = 50000000  # 50ms in ns
    idx = 0  # 添加计数器变量

    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "id", "image1.png", "image2.png",
            "joint_timestamp", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", 
            "tcp_timestamp", "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw", 
            "gripper_timestamp", "gripper_status"
        ])

    # Parse all data
    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == "/camera3/camera1/color/image_raw":
            latest_data["image1"] = (data, timestamp)
        elif topic == "/camera4/camera2/color/image_raw":
            latest_data["image2"] = (data, timestamp)


        elif topic == "/joint_states":
            latest_data["joint_state"] = (data, timestamp)
        elif topic == "/tcp_pose_broadcaster/pose":
            latest_data["tcp_pose"] = (data, timestamp)
        elif topic == "/gripper_status":
            latest_data["gripper_status"] = (data, timestamp)

        if timest is None:
            if None not in [latest_data["image1"], latest_data["image2"],
                            latest_data["joint_state"], latest_data["tcp_pose"],
                            latest_data["gripper_status"]]:
                timest = timestamp
            continue

        if timestamp - timest < timestep:
            continue

        try:
            pc1_filename = ""
            pc2_filename = ""

            # Save image1
            if latest_data["image1"]:
                msg = deserialize_message(latest_data["image1"][0], Image)
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(f"{image_folder1}/{latest_data['image1'][1]}.png", cv_image)

            # Save image2
            if latest_data["image2"]:
                msg = deserialize_message(latest_data["image2"][0], Image)
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv2.imwrite(f"{image_folder2}/{latest_data['image2'][1]}.png", cv_image)

            # Write row to CSV
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                joints_msg = deserialize_message(latest_data["joint_state"][0], JointState)
                tcp_pose_msg = deserialize_message(latest_data["tcp_pose"][0], PoseStamped)

                writer.writerow([
                    idx,
                    f"{latest_data['image1'][1]}.png" if latest_data["image1"] else "",
                    f"{latest_data['image2'][1]}.png" if latest_data["image2"] else "",

                    latest_data["joint_state"][1],
                    joints_msg.position[5], joints_msg.position[0], joints_msg.position[1],
                    joints_msg.position[2], joints_msg.position[3], joints_msg.position[4],
                    latest_data["tcp_pose"][1],
                    tcp_pose_msg.pose.position.x, tcp_pose_msg.pose.position.y, tcp_pose_msg.pose.position.z,
                    tcp_pose_msg.pose.orientation.x, tcp_pose_msg.pose.orientation.y,
                    tcp_pose_msg.pose.orientation.z, tcp_pose_msg.pose.orientation.w,
                    latest_data["gripper_status"][1],
                    extract_gripper_status(str(latest_data["gripper_status"][0]))
                ])

            idx += 1
            timest += timestep

        except Exception as e:
            print(f"Error processing data at timestamp {timestamp}: {e}")
            continue


    print(f"Data exported to {output_csv} and images to {image_folder1}/{image_folder2}")

def process_data(folder_name, file_name):
    bag_dir = os.path.join(folder_name, file_name)
    output_dir = os.path.join(folder_name,file_name+"_proc")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing bag {i} from {bag_dir}...")
    try:

        extract_aligned_data(
            bag_path=bag_dir,
            image_folder1=          os.path.join(output_dir, "image_folder1"),
            image_folder2=          os.path.join(output_dir, "image_folder2"),
            output_csv=             os.path.join(output_dir, "aligned_data.csv")
        )


        print(f"Successfully processed bag {i}")
    except Exception as e:
        print(f"Error processing bag {i}: {str(e)}")

if __name__ == "__main__":

    for i in range(2, 31):

        home_dir = str(Path.home())
    
        folder = os.path.join(home_dir, f"Desktop/ur5e_train_collect/__collect_data/data")
        file = f"HWM_geomsort_pick_{i}"
        process_data(folder, file)
