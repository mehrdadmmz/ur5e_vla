"""
Convert recorded episodes to HDF5 format for VLA training.
Reads the new recording format (state.csv + timestamp-named images).

Folder structure:
  data/TASK_NAME/0/state.csv, camera1/, camera2/
  data/TASK_NAME/1/state.csv, camera1/, camera2/
  ...

Output:
  data/TASK_NAME/0.hdf5
  data/TASK_NAME/1.hdf5
  ...
"""

import argparse
import csv
import os
import numpy as np
import cv2
import h5py


def load_episode(episode_path):
    """Load a single episode from state.csv and images.

    Args:
        episode_path: Path to episode directory containing state.csv and camera folders

    Returns:
        dict with keys: timestamps, joints, tcp_pos, tcp_quat, gripper_pos, images
    """
    csv_path = os.path.join(episode_path, 'state.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"state.csv not found in {episode_path}")

    # Load CSV data
    timestamps = []
    joints = []
    tcp_pos = []
    tcp_quat = []
    gripper_pos = []
    camera_files = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Detect number of cameras from CSV header
        camera_columns = [col for col in fieldnames if col.startswith('camera') and col.endswith('_file')]
        num_cameras = len(camera_columns)

        for row in reader:
            timestamps.append(float(row['timestamp']))
            joints.append([
                float(row['joint_0']),
                float(row['joint_1']),
                float(row['joint_2']),
                float(row['joint_3']),
                float(row['joint_4']),
                float(row['joint_5']),
            ])
            tcp_pos.append([
                float(row['tcp_x']),
                float(row['tcp_y']),
                float(row['tcp_z']),
            ])
            tcp_quat.append([
                float(row['tcp_qx']),
                float(row['tcp_qy']),
                float(row['tcp_qz']),
                float(row['tcp_qw']),
            ])
            gripper_pos.append(float(row['gripper_pos']))

            # Collect camera filenames
            cam_files = [row[col] for col in camera_columns]
            camera_files.append(cam_files)

    # Load images
    images = {f'camera{i+1}': [] for i in range(num_cameras)}

    for frame_idx, cam_files in enumerate(camera_files):
        for cam_idx, filename in enumerate(cam_files):
            cam_name = f'camera{cam_idx + 1}'
            img_path = os.path.join(episode_path, cam_name, filename)

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                images[cam_name].append(img)
            else:
                print(f"Warning: Image not found: {img_path}")
                # Use black image as placeholder
                images[cam_name].append(np.zeros((480, 640, 3), dtype=np.uint8))

    # Convert to numpy arrays
    for cam_name in images:
        images[cam_name] = np.array(images[cam_name], dtype=np.uint8)

    return {
        'timestamps': np.array(timestamps, dtype=np.float64),
        'joints': np.array(joints, dtype=np.float32),
        'tcp_pos': np.array(tcp_pos, dtype=np.float32),
        'tcp_quat': np.array(tcp_quat, dtype=np.float32),
        'gripper_pos': np.array(gripper_pos, dtype=np.float32),
        'images': images,
        'num_cameras': num_cameras,
    }


def episode_to_hdf5(episode_path, output_path, compress=True):
    """Convert a single episode to HDF5.

    Args:
        episode_path: Path to episode directory
        output_path: Output HDF5 file path
        compress: Whether to use gzip compression
    """
    print(f"Loading episode from {episode_path}...")
    data = load_episode(episode_path)

    n_frames = len(data['timestamps'])
    print(f"  Frames: {n_frames}")
    print(f"  Cameras: {data['num_cameras']}")

    # Build qpos (joints + gripper normalized)
    gripper_normalized = data['gripper_pos'].reshape(-1, 1) / 255.0
    qpos = np.concatenate([data['joints'], gripper_normalized], axis=1)

    # Build full state vector [tcp_pos(3), tcp_quat(4), joints(6), gripper(1)]
    state = np.concatenate([
        data['tcp_pos'],
        data['tcp_quat'],
        data['joints'],
        gripper_normalized
    ], axis=1)

    # Compression settings
    compression = 'gzip' if compress else None
    compression_opts = 4 if compress else None

    print(f"Writing to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['num_frames'] = n_frames
        f.attrs['num_cameras'] = data['num_cameras']

        # Timestamps
        f.create_dataset('timestamps', data=data['timestamps'])

        # Observations group
        obs = f.create_group('observations')

        # Images subgroup
        images_grp = obs.create_group('images')
        for cam_name, img_data in data['images'].items():
            images_grp.create_dataset(
                cam_name,
                data=img_data,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(1, *img_data.shape[1:])  # Chunk by frame for faster access
            )

        # State observations
        obs.create_dataset('qpos', data=qpos)  # (N, 7) joints + gripper
        obs.create_dataset('state', data=state)  # (N, 14) full state
        obs.create_dataset('tcp_pos', data=data['tcp_pos'])
        obs.create_dataset('tcp_quat', data=data['tcp_quat'])
        obs.create_dataset('joints', data=data['joints'])
        obs.create_dataset('gripper_pos', data=data['gripper_pos'])

    # Report file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output size: {file_size:.2f} MB")

    return output_path


def convert_task(task_dir, compress=True, force=False):
    """Convert all episodes in a task directory to HDF5.

    Args:
        task_dir: Path to task directory (e.g., data/HWM_geomsort/)
        compress: Whether to use gzip compression
        force: Overwrite existing HDF5 files

    Folder structure:
        task_dir/0/state.csv -> task_dir/0.hdf5
        task_dir/1/state.csv -> task_dir/1.hdf5
    """
    if not os.path.isdir(task_dir):
        raise ValueError(f"Not a directory: {task_dir}")

    # Find episode directories (numbered folders containing state.csv)
    episodes = []
    for name in os.listdir(task_dir):
        episode_path = os.path.join(task_dir, name)
        if os.path.isdir(episode_path) and os.path.exists(os.path.join(episode_path, 'state.csv')):
            episodes.append((name, episode_path))

    # Sort by episode number (handle both numeric and string names)
    episodes.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else float('inf'), x[0]))

    if not episodes:
        print(f"No episodes found in {task_dir}")
        return

    print(f"Found {len(episodes)} episodes in {task_dir}")

    converted = 0
    skipped = 0

    for episode_name, episode_path in episodes:
        output_path = os.path.join(task_dir, f"{episode_name}.hdf5")

        if os.path.exists(output_path) and not force:
            print(f"  [{episode_name}] Skipping (already exists)")
            skipped += 1
            continue

        print(f"\n[{episode_name}] Converting...")
        try:
            episode_to_hdf5(episode_path, output_path, compress)
            converted += 1
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nDone: {converted} converted, {skipped} skipped")


def print_hdf5_structure(hdf5_path):
    """Print the structure of an HDF5 file."""
    print(f"\nHDF5 Structure: {hdf5_path}")
    print("=" * 60)

    def print_attrs(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: {obj.shape} {obj.dtype}")
        else:
            print(f"{indent}{name}/")
            for key, val in obj.attrs.items():
                print(f"{indent}  @{key}: {val}")

    with h5py.File(hdf5_path, 'r') as f:
        for key, val in f.attrs.items():
            print(f"@{key}: {val}")
        f.visititems(print_attrs)


def main():
    parser = argparse.ArgumentParser(description="Convert episodes to HDF5 format")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Convert task directory
    convert_parser = subparsers.add_parser('convert', help='Convert all episodes in a task directory')
    convert_parser.add_argument('task_dir', help='Path to task directory (e.g., data/HWM_geomsort/)')
    convert_parser.add_argument('--no-compress', action='store_true', help='Disable gzip compression')
    convert_parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing HDF5 files')

    # Inspect HDF5
    inspect_parser = subparsers.add_parser('inspect', help='Print HDF5 structure')
    inspect_parser.add_argument('hdf5_file', help='Path to HDF5 file')

    args = parser.parse_args()

    if args.command == 'convert':
        convert_task(args.task_dir, not args.no_compress, args.force)

    elif args.command == 'inspect':
        print_hdf5_structure(args.hdf5_file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
