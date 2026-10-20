# Three-camera calibration (accepted, 2026-10-20)

The calibration currently in use for all three RealSense cameras. Older
results in `handeye/handeye_data/` and `camera3_eye_to_hand/results/` are
superseded. Do not use them.

| Camera | Serial | Mount | File | Position |
|---|---|---|---|---|
| camera1 | `337322072205` | base (fixed) | `T_base_camera1.npy` | (−0.027, 0.199, 0.054) m in base |
| camera2 | `243322072780` | wrist | `T_ee_camera2.npy` | (−33.0, −44.7, 73.4) mm from flange |
| camera3 | `233722072756` | wall (fixed) | `T_base_camera3.npy` | (0.048, 1.049, 0.569) m in base |

`calibration.yaml` has all three matrices, quality numbers and intrinsics in
one file.

## Using them

`T_A_B` is the pose of frame B in frame A: `p_A = T_A_B @ p_B`. Camera frames
are the RealSense **colour** optical frames at 640x480.

```python
T_base_cam1 = np.load("calibration/three_cam_calibration/T_base_camera1.npy")
T_base_cam3 = np.load("calibration/three_cam_calibration/T_base_camera3.npy")
T_ee_cam2   = np.load("calibration/three_cam_calibration/T_ee_camera2.npy")

p_base = T_base_cam1 @ p_cam1                   # fixed cameras
p_base = T_base_ee @ T_ee_cam2 @ p_cam2         # wrist; T_base_ee = current TCP pose
```

The wrist result assumes pendant TCP "TCP" with **zero offset (tool flange)**
and Feature "Base". Keep that setting.

## Quality

| | Captures | Reprojection | Consistency | Approx. 1-sigma |
|---|---|---|---|---|
| camera2 (automatic, `auto_wrist.py`) | 21/23 | 0.56 px | 0.38 mm | 0.13 mm |
| camera3 | 20/20 | 0.21 px | 1.16 mm | 0.41 mm |
| camera1 | 15/15 | 0.38 px | 2.58 mm | 0.44 mm |

Independent live cross-check (`validation/`), board at new positions with the
robot moved between measurements:

- camera2 vs camera3: mean **2.1 mm**, max 3.7 mm
- camera1 vs camera2: mean **2.1 mm**, max 3.8 mm

## Contents

```
T_ee_camera2.npy, T_base_camera1.npy, T_base_camera3.npy   the results
calibration.yaml                                           all three in one file
camera{1,2,3}/report.yaml                                  full per-camera report
camera{1,2,3}/captures.npz                                 raw captures (re-solvable)
validation/*.csv                                           cross-check logs
```

Board: 7x5 ChArUco, DICT_6X6_250, printed at 93.6% scale, so squares are
**23.4 mm** (not the 25 mm it was designed for). Measured: 7 squares = 164 mm,
5 squares = 117 mm.

Re-solve any camera from its captures, without hardware:

```bash
.venv/bin/python calibration/calibrate_wrist.py \
    --from-captures calibration/three_cam_calibration/camera2/captures.npz
```
