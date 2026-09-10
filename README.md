# UR5e VLA Data Collection

Data collection system for UR5e robot manipulation tasks. Collects synchronized multi-camera RGB + depth demonstrations with full robot state, converts to HDF5 for VLA training.

## Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| Robot | Universal Robots UR5e | RTDE control/receive |
| Gripper | Robotiq 2F-85 | Socket control on port 63352 |
| Cameras | 3× Intel RealSense D435/D435i | Synchronized RGB + depth at 640×480 |

Camera placement (configured in `collect/config.yaml`):
- **camera1** (`337322072205`): base-mounted
- **camera2** (`243322072780`): wrist-mounted
- **camera3** (`233722072756`): opposite-side countertop

## Setup

### Network

The robot is at `192.168.56.101`. Set your workstation's Ethernet interface to the same subnet (e.g., `192.168.56.1/24`).

Verify connectivity:
```bash
ping -c 2 192.168.56.101
nc -vz -w 2 192.168.56.101 30004   # RTDE
nc -vz -w 2 192.168.56.101 63352   # Gripper
```

### Python environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify imports:
```bash
.venv/bin/python -c "import yaml, numpy, cv2, pyrealsense2, rtde_control, rtde_receive, h5py; print('ok')"
```

### Camera check

```bash
rs-enumerate-devices -s          # list connected RealSense devices
.venv/bin/python test_cam.py     # live preview from all 3 cameras
```

## Repository Structure

```
ur5e_vla/
├── collect/                     # Shared collection infrastructure
│   ├── config.yaml              # Hardware config (cameras, robot IP, gripper)
│   ├── recorder.py              # Multi-camera + state recorder (RGB + depth)
│   ├── robotiq_gripper.py       # Gripper socket driver
│   ├── ur5e_controller.py       # Robot motion primitives
│   ├── plan_executor.py         # YAML plan interpreter
│   ├── main.py                  # Legacy plan-based collection entry point
│   ├── utils.py                 # Shared utilities
│   └── plan_*.yaml              # Per-task plan configs (verify block = source of truth)
├── collect_*.py                 # Per-task standalone collection scripts
├── verify_*.py                  # Per-task setup/calibration scripts
├── convert_ep_hdf5.py           # Batch HDF5 converter (writes into episode dirs)
├── data/
│   ├── convert_to_hdf5.py       # Core HDF5 conversion logic
│   └── run_single.py            # Single-episode converter wrapper
├── test_cam.py                  # Camera preview utility
├── camera3_eye_to_hand/         # Fixed camera3 extrinsic calibration workflow
├── rollout/                     # Policy inference / replay
├── handeye/                     # Hand-eye calibration data
└── requirements.txt
```

## Data Format

### Raw episode layout

Each episode is a directory under `data/<task>/<subset>/data/<episode_number>/`:

```
data/put_cup_in_bowl/clean/data/1/
├── state.csv                    # Robot state per frame
├── meta.json                    # Episode metadata (task, subset, episode, params)
├── task_sequence.json           # Optional task/chunk annotations and language goal
├── intrinsics.json              # Per-camera fx/fy/cx/cy/depth_scale
├── camera1/                     # RGB frames (timestamp-named PNGs)
├── camera2/
├── camera3/
├── camera1_depth/               # Aligned depth frames (uint16 PNGs, mm)
├── camera2_depth/
└── camera3_depth/
```

`state.csv` columns:
```
frame_idx, timestamp, joint_0..joint_5, tcp_x, tcp_y, tcp_z,
tcp_qx, tcp_qy, tcp_qz, tcp_qw, gripper_pos, camera1_file, camera2_file, camera3_file
```

Annotated multi-step tasks append `chunk_index`, `phase`, `active_target_id`,
and `active_destination_id` to each row.

### HDF5 schema

Each episode converts to `episode<n>.hdf5` inside the episode directory:

```
@num_frames: N
@num_cameras: 3
@hdf5_schema_version: 2
frame_indices: (N,) int64
timestamps: (N,) float64
observations/
  images/
    camera1: (N, 480, 640, 3) uint8      # RGB
    camera2: (N, 480, 640, 3) uint8
    camera3: (N, 480, 640, 3) uint8
  depth/
    camera1: (N, 480, 640) uint16         # mm, multiply by depth_scale for meters
    camera2: (N, 480, 640) uint16
    camera3: (N, 480, 640) uint16
  qpos: (N, 7) float32                   # joints(6) + normalized_gripper(1)
  state: (N, 14) float32                 # tcp_pos(3) + tcp_quat(4) + joints(6) + gripper(1)
  tcp_pos: (N, 3) float32                # end-effector position (meters)
  tcp_quat: (N, 4) float32               # end-effector orientation [qx, qy, qz, qw]
  joints: (N, 6) float32                 # joint angles (radians)
  gripper_pos: (N,) float32              # raw gripper position (0-255)
camera_info/
  camera{1,2,3}/
    coeffs: (5,) float64                 # distortion coefficients
    @fx, @fy, @cx, @cy: float            # intrinsic parameters
    @depth_scale: float                   # depth → meters multiplier
metadata/                                 # lossless meta/intrinsics JSON copies
frame_annotations/                       # optional per-frame task context
task/                                    # optional task sequence and step boundaries
```

**Depth note:** RealSense writes `65535` (uint16 max) for invalid pixels. Mask with `depth[depth == 65535] = 0` before use.

### Subset organization

Each task has 5 subsets:
- `clean` — 100 episodes, no distractors
- `d1` through `d4` — 25 episodes each, with increasing clutter/distractors

## Completed Tasks

| Task | Description | Clean | d1 | d2 | d3 | d4 | Total |
|------|-------------|-------|----|----|----|----|-------|
| `put_book_in_box` | Pick up book, place in box | 100 | 25 | 25 | 25 | 25 | 200 |
| `put_bowl_on_rack` | Pick up bowl, place on dish rack | 100 | 25 | 25 | 25 | 25 | 200 |
| `put_cup_in_bowl` | Pick up cup, place into bowl | 100 | 25 | 25 | 25 | 25 | 200 |
| `put_mug_on_coaster` | Pick up mug, place on coaster | 100 | 25 | 25 | 25 | 25 | 200 |
| `stack_two_cubes` | Pick up cube, stack on another cube | 100 | 25 | 25 | 25 | 25 | 200 |
| `place_three_cups_in_bowls` | Place three colored cups into three bowls | 100 | 25 | 25 | 25 | 25 | 200 |
| **Total** | | **600** | **150** | **150** | **150** | **150** | **1200** |

Dataset available on HuggingFace: [`mzxuan/real_world_data`](https://huggingface.co/datasets/mzxuan/real_world_data)

## Workflow: Adding a New Task

See [TASKS.md](TASKS.md) for the full guide. Summary:

1. **Create a verify script** (`verify_<objects>.py`) — interactive tool to jog the robot, find pick/place heights, and run one full trial cycle.
2. **Create a plan YAML** (`collect/plan_<task>.yaml`) — the `verify:` block is the single source of truth for all motion heights and positions.
3. **Create a collect script** (`collect_<task>.py`) — copy an existing one as template. Reads heights from the plan YAML.
4. **Verify** the setup: `.venv/bin/python verify_<objects>.py --plan collect/plan_<task>.yaml`
5. **Collect clean** (100 eps): `.venv/bin/python -u collect_<task>.py -n 100`
6. **Collect cluttered** (25 eps × 4): `.venv/bin/python -u collect_<task>.py --subset d1 -n 25 --reset-pause 10`
7. **Convert to HDF5**: `.venv/bin/python convert_ep_hdf5.py --task <task> --subsets clean,d1,d2,d3,d4`

The multi-step three-cup task uses `collect_three_cups_bowls.py`,
`verify_three_cups_bowls.py`, and
`collect/plan_place_three_cups_in_bowls.yaml`. Its dataset-specific integrity
check is `audit_three_cups_dataset.py`.

## Camera3 Extrinsic Calibration

The isolated [`camera3_eye_to_hand`](camera3_eye_to_hand/) workflow estimates
the fixed wall camera pose `T_base_camera3` using the calibrated wrist camera
and a ChArUco board. See its README for the capture and validation procedure.

The accepted calibration is
[`result_4`](camera3_eye_to_hand/results/result_4/); other local trial runs are
ignored so rerunning calibration does not clutter version control.

## Uploading to HuggingFace

Use the `hf` CLI with Xet high-performance transfer:

```bash
export HF_TOKEN=<your_token>
export HF_XET_HIGH_PERFORMANCE=1
export PATH=.venv/bin:$PATH

hf upload <repo_id> data/<task>/clean <task>/clean \
    --repo-type=dataset \
    --include="**/*.hdf5" --include="**/*.json" --include="**/*.csv" \
    --commit-message="clean <task> (100 eps)"
```

Uploads only HDF5 + metadata (not raw PNGs). Idempotent — safe to re-run on interruption.

## Key Files

| File | Purpose |
|------|---------|
| `collect/recorder.py` | Multi-camera RGB+depth recorder with synchronized state logging |
| `collect/robotiq_gripper.py` | Robotiq 2F-85 socket driver |
| `collect/config.yaml` | Camera serials, robot IP, recording Hz |
| `collect/plan_<task>.yaml` | Per-task config; `verify:` block = source of truth for heights |
| `collect_<task>.py` | Per-task collection loop (pick → place → reset) |
| `verify_<objects>.py` | Interactive setup: hover placement + one trial cycle |
| `convert_ep_hdf5.py` | Batch converter: raw episodes → HDF5 (writes into episode dirs) |
| `data/convert_to_hdf5.py` | Core conversion logic (called by `convert_ep_hdf5.py`) |
| `test_cam.py` | Quick camera preview / connectivity check |
| `camera3_eye_to_hand/` | Read-only robot-state workflow for calibrating fixed camera3 |
| `audit_three_cups_dataset.py` | Cross-check three-cup metadata, frames, and geometry |
