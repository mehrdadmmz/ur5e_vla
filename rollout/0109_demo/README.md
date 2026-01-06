# Block World Planner - UR5e Robot Control System

A web-based interface for controlling a UR5e robot to manipulate blocks using PDDL planning, ArUco marker perception, and real-time 3D visualization.

## Features

- **Real-time 3D Visualization**: See blocks and gripper positions in a Three.js scene
- **PDDL Planning**: Automatic task planning using PDDLStream + Fast Downward
- **ArUco Perception**: Block detection and tracking via wrist-mounted RealSense camera
- **Goal-based Execution**: Select goals (tower, bridge, pyramid, etc.) and watch the robot execute
- **Voice Commands**: Optional voice control via OpenAI Whisper API
- **WebSocket Updates**: Real-time state synchronization between robot and UI

## Prerequisites

### System Requirements
- Ubuntu 20.04+ (tested on 22.04)
- Python 3.10+
- Node.js 18+ and npm
- CMake and C++ compiler (g++)

### Hardware
- UR5e robot with RTDE interface
- Robotiq gripper (optional but recommended)
- Intel RealSense camera (D400 series)
- ArUco markers on blocks (DICT_5X5_50)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ur5e_vla/rollout/0109_demo
```

### 2. Build PDDLStream Planner

The Fast Downward planner needs to be compiled:

```bash
cd planner/pddlstream/downward
./build.py
cd ../../..
```

This takes ~5 minutes and creates the `builds/` folder.

### 3. Install Python Dependencies

```bash
pip install fastapi uvicorn websockets pydantic pyyaml numpy opencv-python pyrealsense2
```

For the robot interface:
```bash
pip install rtde-control rtde-receive
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Configure the System

Edit `config.yaml` to match your setup:

```yaml
robot:
  ip: "192.168.56.101"  # Your robot's IP address

camera:
  serial: "243322072780"  # Your RealSense serial number
  calibration_file: "path/to/calibration.npy"  # Hand-eye calibration

table_z: -0.15  # Table surface height in robot coordinates
```

### 6. (Optional) Voice Commands

Create `backend/.env` with your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

## Running the System

### Start the Backend

```bash
cd backend
python main.py --config ../config.yaml
```

The API server runs at `http://localhost:8000`

### Start the Frontend (Development)

In a new terminal:

```bash
cd frontend
npm run dev
```

The UI is available at `http://localhost:5173`

### Production Build

```bash
cd frontend
npm run build
```

Then the backend serves the built frontend at `http://localhost:8000`

## Usage

1. **Connect**: Click "Connect" in the UI to initialize the robot and camera
2. **Observe**: The system automatically detects blocks via ArUco markers
3. **Select Goal**: Choose a goal configuration (tower, bridge, etc.)
4. **Execute**: Click "Set" to start planning and execution
5. **Control**: Use Pause/Continue/Quit buttons as needed
6. **Home**: Click "Home" to return robot to initial position

## Project Structure

```
0109_demo/
├── backend/           # FastAPI backend
│   ├── main.py        # Entry point
│   ├── planner_adapter.py  # Robot/planner integration
│   ├── state_manager.py    # State coordination
│   ├── command_bridge.py   # Command handling
│   └── routes/        # API endpoints
├── frontend/          # React + Three.js UI
│   └── src/
│       ├── components/    # UI components
│       └── context/       # State management
├── planner/           # PDDL planning
│   ├── pddlstream/    # PDDLStream library
│   └── domain.pddl    # Block world domain
├── robot/             # Robot control
│   ├── controller.py  # RTDE interface
│   ├── primitives.py  # Motion primitives
│   └── gripper.py     # Gripper control
├── perception/        # Vision system
│   ├── aruco_monitor.py   # ArUco detection
│   └── predicates.py      # Logical state extraction
└── config.yaml        # System configuration
```

## Available Goals

| Goal | Description |
|------|-------------|
| `tower` | Stack 3 blocks vertically |
| `bridge` | Two base blocks with plank on top |
| `two_towers` | Two separate 2-block towers |
| `pyramid` | 3-block pyramid structure |
| `double_bridge` | Extended bridge with 4 bases |

## Troubleshooting

### Robot Connection Failed
- Verify robot IP in `config.yaml`
- Ensure robot is in Remote Control mode
- Check network connectivity: `ping <robot-ip>`

### Camera Not Found
- Verify serial number in `config.yaml`
- Check USB connection: `rs-enumerate-devices`
- Ensure proper permissions: add user to `video` group

### Planner Fails
- Ensure Fast Downward is built: check `planner/pddlstream/downward/builds/` exists
- Verify domain.pddl syntax
- Check console for planning errors

### Blocks Not Detected
- Ensure good lighting (no glare on markers)
- Verify marker size in config matches physical markers
- Check camera calibration

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/connect` | POST | Initialize robot/camera |
| `/api/system/status` | GET | Get connection status |
| `/api/system/config` | GET | Get system config |
| `/api/control/home` | POST | Move to home position |
| `/api/control/pause` | POST | Pause execution |
| `/api/control/continue` | POST | Resume execution |
| `/api/control/goal` | POST | Set goal and start |
| `/api/state/current` | GET | Get full state |
| `/api/state/observe` | POST | Trigger observation |
| `/ws` | WebSocket | Real-time updates |

## License

MIT License - See LICENSE file for details.
