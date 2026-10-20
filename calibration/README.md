# Camera calibration (all three cameras)

Finds the pose of every RealSense camera relative to the UR5e. The wrist camera
is calibrated first (hand-eye), then the two fixed cameras are calibrated
through it, and finally a live cross-check confirms they all agree.

**Calibration in use: [`three_cam_calibration/`](three_cam_calibration/)**
(2026-10-20). Its README has the matrices, quality numbers and usage. The older
results in `handeye/handeye_data/` and `camera3_eye_to_hand/results/` are
superseded.

| Camera | Serial | Mount | Result | Meaning |
|---|---|---|---|---|
| camera1 | `337322072205` | base (fixed) | `T_base_camera1` | camera1 pose in the robot base |
| camera2 | `243322072780` | wrist | `T_ee_camera2` | camera2 pose in the TCP/flange frame |
| camera3 | `233722072756` | wall (fixed) | `T_base_camera3` | camera3 pose in the robot base |

## Conventions

- `T_A_B` is the pose of frame B in frame A: `p_A = T_A_B @ p_B`, 4x4, metres.
- Camera frames are the RealSense **colour** optical frames (x right, y down,
  z forward). Recorded depth is aligned to colour, so depth points use the same
  frame.
- `ee` is the UR TCP with pendant TCP **"TCP" (zero offset = tool flange)** and
  Feature **"Base"**. Keep this setting for calibration, collection and rollout.
- Fixed cameras: `p_base = T_base_cameraN @ p_cameraN`.
  Wrist camera: `p_base = T_base_ee @ T_ee_camera2 @ p_camera2`, with
  `T_base_ee` the current TCP pose.
- The rollout config field `T_cam_base` is used as camera-to-base, so paste
  `T_base_cameraN` into it **as-is (do not invert)**.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Serials, board size, capture checks, auto-wrist motion limits |
| `calibrate_wrist.py` | Step 1, manual: you move the robot, press `s` |
| `auto_wrist.py` | Step 1, automatic: **the robot moves itself** around the board (recommended) |
| `calibrate_fixed.py` | Steps 2–3: `--camera camera3` / `--camera camera1` |
| `validate.py` | Step 4: live cross-check between cameras |
| `geometry.py` | Solvers (closed-form start + bundle adjustment); no hardware |
| `hardware.py` | Cameras, ChArUco detection, read-only robot pose, stationary capture |
| `session.py` | Shared keyboard loop for the interactive scripts |
| `tests/` | Offline solver tests on synthetic data |
| `three_cam_calibration/` | The accepted results |

Only `auto_wrist.py` moves the robot. Every other script only reads its pose.

## Before you start

- **Robot:** pendant TCP "TCP" (zeros), Feature "Base". Use **Local** mode for
  Freedrive and **Remote Control** mode only for `auto_wrist.py`.
- **Network:** robot at `192.168.56.101`. If it does not answer, the laptop may
  have put the adapter on another profile: `nmcli con up UR5e`.
- **Cameras:** close `test_cam.py` and anything else using them.
- **Board:** one 7x5 ChArUco board (`DICT_6X6_250`) for every step. Ours was
  printed at 93.6% scale, so its squares are **23.4 mm**, not the 25 mm it was
  designed for (7 squares = 164 mm, 5 squares = 117 mm). If you reprint it,
  measure the full 7-square length, divide by 7, and update `board:` in
  `config.yaml`; scale `marker_length_m` by the same ratio. A wrong size scales
  every camera distance. The wrist report's "board size check" (from robot
  motion) should agree with the ruler.
- Keep the board flat and still, and avoid glare.

## Step 1: wrist camera2

### Automatic (recommended)

**The robot moves. Keep a hand near the e-stop.**

1. Tape the board flat on the table.
2. Freedrive the robot so the wrist camera looks roughly straight down at the
   board from about 30–50 cm, with free space all around. This is "home".
3. Plan and check without moving:
   ```bash
   .venv/bin/python calibration/auto_wrist.py --dry-run
   ```
4. Clear the table around the board, set the pendant speed slider to ~50% for a
   first run, switch to **Remote Control**, then run it and press Enter:
   ```bash
   .venv/bin/python calibration/auto_wrist.py
   ```

It finds the board from home and plans ~26 camera poses on a dome around it
(tilts 0/15/25/35°, 8 directions, 28/36 cm, ±30° wrist twist), always aimed at
the board. Each pose is skipped if the gripper would come within 8 cm of the
board along the path, the board would leave the image, it is more than
35 cm / 70° from home, it needs a joint swing over 90°, or the controller
reports no IK solution or a safety-limit violation. Moves are straight lines at
8 cm/s, returning home between poses. **Esc** or **q** in the camera window
stops the arm. When done it solves and saves like the manual step.

### Manual

```bash
.venv/bin/python calibration/calibrate_wrist.py
```

Tape the board down along all edges. Freedrive the camera to 25–40 cm from the
board and press `s` once the screen says "robot stopped". Repeat 20–30 times.
**Tilt matters most:** lean the tool 20–35° toward the board from the front,
back, left and right, at a few heights, with some wrist twist. The on-screen
line lists tilt directions still missing (+x -x +y -y). Twisting alone does not
help. Then `c` to check, `w` to save.

**Good result:** reprojection < 0.5 px, board consistency < 2 mm, all four tilt
directions above 20°.

## Step 2: camera3 (wall)

```bash
.venv/bin/python calibration/calibrate_fixed.py --camera camera3
```

Left window: wrist camera. Right window: camera3.

1. Put the board on a **riser** (a box or book) so it is closer to the high wall
   camera, and **tilt it 10–30° toward camera3**.
2. Freedrive the wrist camera to look at it from 35–45 cm. Both windows must
   show green corners (ideally 24/24 steady).
3. Hands off, then press `s`.
4. Repeat 20–25 times, changing the board's position, tilt and rotation each
   time. **Move the robot to a new pose every 5–6 captures** so any small wrist
   error averages out.
5. `c` to check, `w` to save.

**Good result:** board disagreement < 3 mm.

## Step 3: camera1 (base)

```bash
.venv/bin/python calibration/calibrate_fixed.py --camera camera1
```

Same as step 2, with the board tilted toward camera1. camera1 sits low and
looks across the table, so it needs a steep tilt (30–40°). Keep the wrist
camera's view of the board under about 45° off-axis (lower the wrist and tilt it
toward the board's face) for the best accuracy.

## Step 4: cross-check

Run once per pair (all three at once works too if one board position suits
every camera):

```bash
.venv/bin/python calibration/validate.py --cameras camera2,camera3
.venv/bin/python calibration/validate.py --cameras camera1,camera2
```

Place the board where both cameras see it, press `s`, Freedrive the robot
somewhere else, press `s` again, and so on for 5–6 measurements; then `q`. Each
line is how far apart the two cameras place the board. Coloured crosses show
where the other camera puts the board corners.

**Good result:** mean of a few mm, max under ~5 mm. The angle numbers are
naturally noisier for the far, slanted wall camera.

## Keys (interactive scripts)

| Key | Action |
|---|---|
| `s` | capture (averages 10 frames; refused if the robot or board moves) |
| `d` | delete the last capture |
| `c` | calculate and print the result + per-capture table |
| `w` | calculate and save |
| `q` | quit (press twice if there are unsaved captures) |

## Outputs and accepting a new calibration

New runs go to `calibration/results/` (git-ignored) and never touch the
accepted folder:

```
calibration/results/camera2/result_N/T_ee_camera2.npy
calibration/results/camera3/result_N/T_base_camera3.npy
calibration/results/camera1/result_N/T_base_camera1.npy
    + report.yaml (matrix, quality, change vs the accepted one) + captures.npz
calibration/results/validation/validation_<time>.csv
```

Steps 2–4 use the newest run of each camera in `results/`, otherwise the
accepted one in `three_cam_calibration/`. Pass `--wrist PATH` (fixed) or
`--camera1/--camera2/--camera3 PATH` (validate) to choose explicitly. Unsaved
captures are kept in `results/<camera>/unsaved_session.npz`.

To accept new results: copy them into `three_cam_calibration/`, update its
README and `calibration.yaml`, then delete `calibration/results/`.

## Re-solving without hardware

The per-capture table marks automatic `outlier`s (already dropped) and
`suspicious` captures. To drop more and re-solve from saved captures:

```bash
.venv/bin/python calibration/calibrate_wrist.py \
    --from-captures calibration/results/camera2/result_1/captures.npz --exclude 3,7 --save
.venv/bin/python calibration/calibrate_fixed.py --camera camera3 \
    --from-captures calibration/results/camera3/result_1/captures.npz --exclude 4 --save
```

## How it is solved

- **Capture:** one `s` averages ChArUco corners over 10 frames after dropping
  possibly stale frames. It is refused if the TCP moves, the board is not seen,
  or the reprojection error exceeds 2 px.
- **Detection:** finer adaptive thresholding and 8 px per marker cell, so the
  small, distant board is still fully detected (camera3: 24/24 corners instead
  of 13–18).
- **Wrist:** a closed-form hand-eye estimate (Park–Martin) is refined by bundle
  adjustment of the camera pose and the board pose against every corner. The
  report adds tilt coverage and a board-size check implied by robot motion.
- **Fixed:** per-capture estimates `T_base_ee · T_ee_camera2 · T_camera2_board ·
  inv(T_cameraN_board)` are robustly averaged. Then the camera pose and one
  board pose per capture are refined jointly against the corners seen by both
  cameras.
- Captures whose pixel error is far above the rest are dropped automatically.
- Lens intrinsics are the RealSense factory values at 640x480. Extrinsics do not
  depend on resolution, so the results also hold at other colour resolutions.

## Lessons from the 2026-10-20 calibration

- The board had been printed at 93.6% scale. The previous camera3 result
  assumed 25 mm squares and was about 2 cm off because of it.
- A manual wrist session with mostly small tilts and wrist twists gave 4.6 mm
  consistency. The automatic run, with 24–36° tilts in all four directions,
  gave 0.38 mm.
- In both fixed-camera sessions the robot stayed in one pose, so those results
  lean on the wrist calibration at that pose. The cross-check at several robot
  poses still agreed to about 2 mm, but moving the robot during capture is
  better.

## Tests

```bash
.venv/bin/python -m unittest discover -s calibration/tests -v
```
