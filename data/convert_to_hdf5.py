"""
Convert recorded episodes to HDF5 format for VLA training.

Reads state.csv, timestamp-named RGB/depth frames, and camera intrinsics.
Depth remains raw uint16; multiply it by the stored depth_scale to get meters.
"""

import argparse
import csv
import json
import os
from contextlib import contextmanager
import numpy as np
import cv2
import h5py


CONTEXT_COLUMNS = (
    'chunk_index',
    'phase',
    'active_target_id',
    'active_destination_id',
)
UTF8_DTYPE = h5py.string_dtype(encoding='utf-8')


def load_json_sidecar(path):
    """Return (parsed JSON, original text), or (None, None) when absent."""
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        text = f.read()
    return json.loads(text), text


def string_array(values):
    """Build an HDF5-compatible variable-length UTF-8 string array."""
    return np.asarray(values, dtype=object)


def optional_int(value, default=-1):
    """Parse an optional integer CSV/JSON field."""
    if value in (None, ''):
        return default
    return int(value)


def optional_float(value, default=np.nan):
    """Parse an optional floating-point JSON field."""
    if value in (None, ''):
        return default
    return float(value)


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
    frame_indices = []
    frame_context = {column: [] for column in CONTEXT_COLUMNS}
    has_frame_context = False

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Detect number of cameras from CSV header
        camera_columns = [col for col in fieldnames if col.startswith('camera') and col.endswith('_file')]
        num_cameras = len(camera_columns)
        has_frame_context = all(column in fieldnames for column in CONTEXT_COLUMNS)

        for row in reader:
            frame_indices.append(optional_int(row.get('frame_idx'), len(frame_indices)))
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

            if has_frame_context:
                frame_context['chunk_index'].append(optional_int(row['chunk_index']))
                frame_context['phase'].append(row['phase'] or '')
                frame_context['active_target_id'].append(row['active_target_id'] or '')
                frame_context['active_destination_id'].append(
                    row['active_destination_id'] or ''
                )

    # Load images (color, and depth if camera{i}_depth/ folders exist)
    images = {f'camera{i+1}': [] for i in range(num_cameras)}
    depth_cams = [i + 1 for i in range(num_cameras)
                  if os.path.isdir(os.path.join(episode_path, f'camera{i+1}_depth'))]
    depths = {f'camera{i}': [] for i in depth_cams}

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

            if (cam_idx + 1) in depth_cams:
                depth_path = os.path.join(episode_path, f'{cam_name}_depth', filename)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    print(f"Warning: Depth not found: {depth_path}")
                    depth = np.zeros((480, 640), dtype=np.uint16)
                depths[cam_name].append(depth)

    # Convert to numpy arrays
    for cam_name in images:
        images[cam_name] = np.array(images[cam_name], dtype=np.uint8)
    for cam_name in depths:
        depths[cam_name] = np.array(depths[cam_name], dtype=np.uint16)

    intrinsics, intrinsics_json = load_json_sidecar(
        os.path.join(episode_path, 'intrinsics.json')
    )
    metadata, metadata_json = load_json_sidecar(
        os.path.join(episode_path, 'meta.json')
    )
    task_sequence, task_sequence_json = load_json_sidecar(
        os.path.join(episode_path, 'task_sequence.json')
    )

    return {
        'frame_indices': np.array(frame_indices, dtype=np.int64),
        'timestamps': np.array(timestamps, dtype=np.float64),
        'joints': np.array(joints, dtype=np.float32),
        'tcp_pos': np.array(tcp_pos, dtype=np.float32),
        'tcp_quat': np.array(tcp_quat, dtype=np.float32),
        'gripper_pos': np.array(gripper_pos, dtype=np.float32),
        'images': images,
        'depths': depths,
        'intrinsics': intrinsics,
        'intrinsics_json': intrinsics_json,
        'metadata': metadata,
        'metadata_json': metadata_json,
        'task_sequence': task_sequence,
        'task_sequence_json': task_sequence_json,
        'frame_context': frame_context if has_frame_context else None,
        'num_cameras': num_cameras,
    }


def write_json_dataset(group, name, text):
    """Write an original JSON sidecar as a scalar UTF-8 dataset."""
    if text is not None:
        group.create_dataset(name, data=text, dtype=UTF8_DTYPE)


def write_task_annotations(hdf5_file, data):
    """Preserve task_sequence.json in convenient and lossless forms."""
    task_sequence = data['task_sequence']
    if task_sequence is None:
        return

    task_group = hdf5_file.create_group('task')
    write_json_dataset(task_group, 'task_sequence_json', data['task_sequence_json'])

    scalar_fields = (
        'schema_version',
        'task',
        'condition',
        'clutter_level',
        'assignment_policy',
        'language_instruction',
        'coordinate_frame',
        'units',
        'quat_convention',
        'episode_status',
        'episode_success',
        'recorded_at',
    )
    for key in scalar_fields:
        value = task_sequence.get(key)
        if value is not None:
            task_group.attrs[key] = value

    root_fields = (
        'task',
        'condition',
        'assignment_policy',
        'language_instruction',
        'episode_status',
        'episode_success',
    )
    for key in root_fields:
        value = task_sequence.get(key)
        if value is not None:
            hdf5_file.attrs[key] = value

    cup_order = task_sequence.get('cup_order')
    if isinstance(cup_order, list):
        task_group.create_dataset(
            'cup_order', data=string_array(cup_order), dtype=UTF8_DTYPE
        )

    assignment = task_sequence.get('bowl_assignment')
    if isinstance(assignment, dict):
        colors = list(assignment.keys())
        assignment_group = task_group.create_group('bowl_assignment')
        assignment_group.create_dataset(
            'cup_colors', data=string_array(colors), dtype=UTF8_DTYPE
        )
        assignment_group.create_dataset(
            'bowl_colors',
            data=string_array([assignment[color] for color in colors]),
            dtype=UTF8_DTYPE,
        )

    steps = task_sequence.get('steps')
    if isinstance(steps, list):
        steps_group = task_group.create_group('steps')
        integer_fields = (
            'chunk_index',
            'start_frame',
            'grasp_frame',
            'release_frame',
            'completion_frame',
        )
        float_fields = ('start_ts', 'grasp_ts', 'release_ts', 'completion_ts')
        string_fields = ('target_id', 'destination_id', 'assignment_policy')
        bool_fields = ('color_match', 'success')

        for key in integer_fields:
            values = [optional_int(step.get(key)) for step in steps]
            steps_group.create_dataset(key, data=np.asarray(values, dtype=np.int64))
        for key in float_fields:
            values = [optional_float(step.get(key)) for step in steps]
            steps_group.create_dataset(key, data=np.asarray(values, dtype=np.float64))
        for key in string_fields:
            values = [step.get(key) or '' for step in steps]
            steps_group.create_dataset(
                key, data=string_array(values), dtype=UTF8_DTYPE
            )
        for key in bool_fields:
            values = [bool(step.get(key, False)) for step in steps]
            steps_group.create_dataset(key, data=np.asarray(values, dtype=np.bool_))


def validate_hdf5(hdf5_path, episode_path=None):
    """Raise ValueError if an episode HDF5 is incomplete or inconsistent."""
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if 'num_frames' not in hdf5_file.attrs:
            raise ValueError('missing num_frames attribute')
        n_frames = int(hdf5_file.attrs['num_frames'])
        if n_frames <= 0:
            raise ValueError('num_frames must be positive')

        required = (
            'timestamps',
            'observations/qpos',
            'observations/state',
            'observations/tcp_pos',
            'observations/tcp_quat',
            'observations/joints',
            'observations/gripper_pos',
        )
        for path in required:
            if path not in hdf5_file:
                raise ValueError(f'missing {path}')
            if hdf5_file[path].shape[0] != n_frames:
                actual = hdf5_file[path].shape[0]
                raise ValueError(f'{path} has {actual} rows; expected {n_frames}')

        num_cameras = int(hdf5_file.attrs.get('num_cameras', 0))
        for camera_index in range(1, num_cameras + 1):
            image_path = f'observations/images/camera{camera_index}'
            if image_path not in hdf5_file:
                raise ValueError(f'missing {image_path}')
            if hdf5_file[image_path].shape[0] != n_frames:
                raise ValueError(f'{image_path} frame count does not match')

            source_depth = bool(
                episode_path
                and os.path.isdir(
                    os.path.join(episode_path, f'camera{camera_index}_depth')
                )
            )
            if source_depth:
                depth_path = f'observations/depth/camera{camera_index}'
                if depth_path not in hdf5_file:
                    raise ValueError(f'missing {depth_path}')
                if hdf5_file[depth_path].shape[0] != n_frames:
                    raise ValueError(f'{depth_path} frame count does not match')

        source_has_task_sequence = bool(
            episode_path
            and os.path.exists(os.path.join(episode_path, 'task_sequence.json'))
        )
        if source_has_task_sequence:
            task_sequence_path = os.path.join(episode_path, 'task_sequence.json')
            with open(task_sequence_path, 'r') as source:
                source_task_sequence_json = source.read()
            source_task_sequence = json.loads(source_task_sequence_json)
            required_task_paths = (
                'task/task_sequence_json',
                'task/cup_order',
                'task/bowl_assignment/cup_colors',
                'task/bowl_assignment/bowl_colors',
                'task/steps/chunk_index',
                'task/steps/target_id',
                'task/steps/destination_id',
            )
            for path in required_task_paths:
                if path not in hdf5_file:
                    raise ValueError(f'missing {path}')

            embedded_json = hdf5_file['task/task_sequence_json'][()]
            if isinstance(embedded_json, bytes):
                embedded_json = embedded_json.decode('utf-8')
            if embedded_json != source_task_sequence_json:
                raise ValueError('embedded task_sequence_json does not match source')

            if int(hdf5_file.attrs.get('hdf5_schema_version', 0)) != 2:
                raise ValueError('task-sequence episode requires HDF5 schema version 2')
            source_instruction = source_task_sequence.get('language_instruction', '')
            if hdf5_file.attrs.get('language_instruction', '') != source_instruction:
                raise ValueError('language_instruction does not match source')

            root_fields = (
                'task', 'condition', 'assignment_policy',
                'episode_status', 'episode_success',
            )
            for key in root_fields:
                if hdf5_file.attrs.get(key) != source_task_sequence.get(key):
                    raise ValueError(f'{key} attribute does not match source')

            source_order = source_task_sequence.get('cup_order', [])
            stored_order = hdf5_file['task/cup_order'].asstr()[:].tolist()
            if stored_order != source_order:
                raise ValueError('cup_order does not match source')

            cup_colors = hdf5_file[
                'task/bowl_assignment/cup_colors'
            ].asstr()[:].tolist()
            bowl_colors = hdf5_file[
                'task/bowl_assignment/bowl_colors'
            ].asstr()[:].tolist()
            stored_assignment = dict(zip(cup_colors, bowl_colors))
            if stored_assignment != source_task_sequence.get('bowl_assignment', {}):
                raise ValueError('bowl_assignment does not match source')

            source_steps = source_task_sequence.get('steps', [])
            if hdf5_file['task/steps/chunk_index'].shape[0] != len(source_steps):
                raise ValueError('task step count does not match source')
            step_fields = (
                'chunk_index', 'target_id', 'destination_id',
                'assignment_policy', 'color_match', 'start_frame', 'start_ts',
                'grasp_frame', 'grasp_ts', 'release_frame', 'release_ts',
                'completion_frame', 'completion_ts', 'success',
            )
            string_step_fields = {
                'target_id', 'destination_id', 'assignment_policy',
            }
            for key in step_fields:
                path = f'task/steps/{key}'
                if path not in hdf5_file:
                    raise ValueError(f'missing {path}')
                if key in string_step_fields:
                    stored_values = hdf5_file[path].asstr()[:].tolist()
                else:
                    stored_values = hdf5_file[path][:].tolist()
                source_values = [step.get(key) for step in source_steps]
                if stored_values != source_values:
                    raise ValueError(f'task step field {key} does not match source')

        source_csv = os.path.join(episode_path, 'state.csv') if episode_path else None
        source_has_context = False
        if source_csv and os.path.exists(source_csv):
            with open(source_csv, 'r') as source:
                fieldnames = csv.DictReader(source).fieldnames or []
            source_has_context = all(column in fieldnames for column in CONTEXT_COLUMNS)
        if source_has_context:
            with open(source_csv, 'r') as source:
                source_rows = list(csv.DictReader(source))
            if len(source_rows) != n_frames:
                raise ValueError('source state.csv row count does not match HDF5')

            if 'frame_indices' not in hdf5_file:
                raise ValueError('missing frame_indices')
            source_frame_indices = np.asarray(
                [optional_int(row.get('frame_idx'), index)
                 for index, row in enumerate(source_rows)],
                dtype=np.int64,
            )
            if not np.array_equal(hdf5_file['frame_indices'][:], source_frame_indices):
                raise ValueError('frame_indices do not match source state.csv')

            for key in CONTEXT_COLUMNS:
                path = f'frame_annotations/{key}'
                if path not in hdf5_file:
                    raise ValueError(f'missing {path}')
                if hdf5_file[path].shape[0] != n_frames:
                    raise ValueError(f'{path} row count does not match')
                if key == 'chunk_index':
                    source_values = np.asarray(
                        [optional_int(row.get(key)) for row in source_rows],
                        dtype=np.int64,
                    )
                    if not np.array_equal(hdf5_file[path][:], source_values):
                        raise ValueError(f'{path} does not match source state.csv')
                else:
                    source_values = [(row.get(key) or '') for row in source_rows]
                    if hdf5_file[path].asstr()[:].tolist() != source_values:
                        raise ValueError(f'{path} does not match source state.csv')

        if episode_path and source_has_task_sequence:
            json_sidecars = (
                ('meta.json', 'metadata/meta_json'),
                ('intrinsics.json', 'metadata/intrinsics_json'),
            )
            for filename, dataset_path in json_sidecars:
                source_path = os.path.join(episode_path, filename)
                if not os.path.exists(source_path):
                    continue
                if dataset_path not in hdf5_file:
                    raise ValueError(f'missing {dataset_path}')
                with open(source_path, 'r') as source:
                    source_json = source.read()
                embedded_json = hdf5_file[dataset_path][()]
                if isinstance(embedded_json, bytes):
                    embedded_json = embedded_json.decode('utf-8')
                if embedded_json != source_json:
                    raise ValueError(f'{dataset_path} does not match {filename}')

    return True


@contextmanager
def atomic_hdf5_writer(output_path, episode_path=None):
    """Write and validate a temporary file before replacing the destination."""
    output_path = os.path.abspath(output_path)
    temporary_path = f"{output_path}.tmp"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    try:
        with h5py.File(temporary_path, 'w') as hdf5_file:
            yield hdf5_file
        validate_hdf5(temporary_path, episode_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


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

    output_path = os.path.abspath(output_path)
    print(f"Writing to {output_path}...")
    with atomic_hdf5_writer(output_path, episode_path) as f:
        # Metadata
        f.attrs['hdf5_schema_version'] = 2
        f.attrs['num_frames'] = n_frames
        f.attrs['num_cameras'] = data['num_cameras']

        # Original frame indices and timestamps
        f.create_dataset('frame_indices', data=data['frame_indices'])
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

        # Preserve aligned raw depth for 3-D reconstruction.
        if data['depths']:
            depth_grp = obs.create_group('depth')
            for cam_name, depth_data in data['depths'].items():
                depth_grp.create_dataset(
                    cam_name,
                    data=depth_data,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(1, *depth_data.shape[1:]),
                )

        # Preserve intrinsics and depth scale alongside the frame arrays.
        if data['intrinsics'] is not None:
            info_grp = f.create_group('camera_info')
            for cam_name, intrinsics in data['intrinsics'].items():
                camera_grp = info_grp.create_group(cam_name)
                for key, value in intrinsics.items():
                    if isinstance(value, list):
                        camera_grp.create_dataset(
                            key, data=np.array(value, dtype=np.float64)
                        )
                    else:
                        camera_grp.attrs[key] = value

        # Preserve original JSON metadata losslessly for provenance and for
        # loaders that need fields not promoted into the main schema.
        metadata_group = f.create_group('metadata')
        write_json_dataset(metadata_group, 'meta_json', data['metadata_json'])
        write_json_dataset(
            metadata_group, 'intrinsics_json', data['intrinsics_json']
        )

        # State observations
        obs.create_dataset('qpos', data=qpos)  # (N, 7) joints + gripper
        obs.create_dataset('state', data=state)  # (N, 14) full state
        obs.create_dataset('tcp_pos', data=data['tcp_pos'])
        obs.create_dataset('tcp_quat', data=data['tcp_quat'])
        obs.create_dataset('joints', data=data['joints'])
        obs.create_dataset('gripper_pos', data=data['gripper_pos'])

        # Per-frame task context remains exactly aligned with observations.
        if data['frame_context'] is not None:
            annotations = f.create_group('frame_annotations')
            context = data['frame_context']
            annotations.create_dataset(
                'chunk_index',
                data=np.asarray(context['chunk_index'], dtype=np.int64),
            )
            for key in ('phase', 'active_target_id', 'active_destination_id'):
                annotations.create_dataset(
                    key, data=string_array(context[key]), dtype=UTF8_DTYPE
                )

        write_task_annotations(f, data)

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
