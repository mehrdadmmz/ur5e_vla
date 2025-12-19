# UR5e VLA Data Collection & Rollout

Data collection and policy rollout system for UR5e robot with Robotiq gripper and RealSense cameras.

## Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Project Structure

```
ur5e_vla/
├── collect/           # Data collection
│   ├── main.py        # Main collection script
│   ├── config.yaml    # Hardware configuration
│   ├── plan.yaml      # Task definition
│   ├── plan_executor.py
│   ├── recorder.py    # Camera & state recording
│   ├── ur5e_controller.py
│   ├── robotiq_gripper.py
│   └── utils.py
├── data/              # Recorded episodes
│   ├── convert_to_hdf5.py
│   └── TASK_NAME/
│       ├── 0/         # Episode 0
│       │   ├── state.csv
│       │   ├── camera1/
│       │   └── camera2/
│       ├── 0.hdf5     # Converted
│       └── ...
├── rollout/           # Policy inference
│   ├── replay_episode.py
│   └── ...
└── requirements.txt
```

## Data Collection

### 1. Configure Hardware

Edit `collect/config.yaml`:

```yaml
recording:
  hz: 20
  camera_serials:
    - "337322072205"  # base camera
    - "243322072780"  # wrist camera
  camera_resolution: [640, 480]

robot:
  ip: "192.168.56.101"
  speed: 0.4
  acceleration: 0.4

gripper:
  port: 63352
  speed: 100
  force: 100
```

### 2. Define Task

Edit `collect/plan.yaml`:

```yaml
task:
  name: my_task
  num_rounds: 10
  record: true

poses:
  home: [-0.2, -0.5, 0.3, 0, 0, 0, 1]  # [x, y, z, qx, qy, qz, qw]
  pick_pos: [-0.3, -0.4, 0.15, 0, 0, 0, 1]

actions:
  pick:
    - move_to: {pose: $pose, height: $height}
    - move_delta: [0, 0, -0.05, 0, 0, 0, 1]
    - gripper: close
    - move_delta: [0, 0, 0.10, 0, 0, 0, 1]

sequence:
  - move_to: home
  - pick: {pose: pick_pos, height: 0.20}

reset:
  - gripper: open
  - move_to: home
```

### 3. Run Collection

```bash
cd collect
python main.py --plan plan.yaml --config config.yaml
```

Data saves to `data/TASK_NAME/EPISODE_NUM/`.

## Data Format

### Raw Recording

Each episode folder contains:
- `state.csv` - Robot state at each timestep
- `camera1/`, `camera2/` - Images named by timestamp (e.g., `1734567890.123456.png`)

CSV columns:
```
frame_idx, timestamp,
joint_0..joint_5,
tcp_x, tcp_y, tcp_z, tcp_qx, tcp_qy, tcp_qz, tcp_qw,
gripper_pos,
camera1_file, camera2_file
```

### HDF5 Format

Convert recordings to HDF5 for training:

```bash
python data/convert_to_hdf5.py convert data/TASK_NAME/
```

Structure:
```
@num_frames: N
@num_cameras: 2
timestamps: (N,) float64
observations/
  images/
    camera1: (N, 480, 640, 3) uint8
    camera2: (N, 480, 640, 3) uint8
  qpos: (N, 7) float32      # joints(6) + gripper(1)
  state: (N, 14) float32    # tcp_pos(3) + tcp_quat(4) + joints(6) + gripper(1)
  tcp_pos: (N, 3) float32
  tcp_quat: (N, 4) float32  # [qx, qy, qz, qw]
  joints: (N, 6) float32
  gripper_pos: (N,) float32 # 0-255
```

Inspect an HDF5 file:
```bash
python data/convert_to_hdf5.py inspect data/TASK_NAME/0.hdf5
```

## Replay Episode

Replay a recorded episode on the robot:

```bash
cd rollout
python replay_episode.py ../data/TASK_NAME/0 --speed 0.3
```

Options:
- `--rate 0.5` - Playback speed multiplier
- `--no-gripper` - Skip gripper replay
- `--dry-run` - Print actions without executing
- `--skip-start` - Don't move to start position first

## Plan Primitives

Available actions in `plan.yaml`:

| Primitive | Description |
|-----------|-------------|
| `move_to: pose_name` | Move to named pose |
| `move_to: {pose: name, height: 0.2}` | Move to pose at height |
| `move_delta: [dx, dy, dz, dqx, dqy, dqz, dqw]` | Relative move |
| `gripper: open` | Open gripper |
| `gripper: close` | Close gripper |
| `gripper: 128` | Move gripper to position (0-255) |
| `wait: 0.5` | Wait seconds |

Compound actions can combine primitives with variable substitution (`$pose`, `$height`).
