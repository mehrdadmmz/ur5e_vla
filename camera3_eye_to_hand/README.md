# Camera3 eye-to-hand calibration

This folder contains an isolated calibration workflow for the fixed wall camera
(`camera3`, serial `233722072756`). It uses the already-calibrated wrist camera
(`camera2`, serial `243322072780`) as the robot-base reference.

Nothing in this folder changes or overwrites the existing wrist calibration or
rollout configuration.

## Safety and assumptions

- `calibrate_camera3.py` imports only the RTDE receive interface. It reads the
  current TCP pose and has no code capable of commanding robot motion.
- Camera3 must remain rigidly mounted during and after calibration.
- Both cameras must see the same ChArUco board.
- Each camera must detect at least six ChArUco corners for a valid pose.
- The robot and board must remain completely stationary while `s` captures a
  sample. The two RealSense streams are read sequentially, not hardware synced.
- The new board is 7 columns x 5 rows, uses `DICT_6X6_250`, has 25 mm squares
  and 18 mm markers, starting at marker ID 0. Print it at 100% / actual size
  and confirm a square measures exactly 25 mm. Do not use the old 4x4 PDF with
  this configuration.

## Geometry

Transform names use `T_A_B`: the pose of frame B expressed in frame A.

For every paired observation, the program calculates:

```text
T_base_camera3 =
    T_base_ee
    @ T_ee_camera2
    @ T_camera2_board
    @ inverse(T_camera3_board)
```

Camera3 is fixed, so all valid board poses should produce the same
`T_base_camera3`. The program robustly averages those candidates and rejects
translation or rotation outliers using the thresholds in `config.yaml`.

## Before running

1. Check the two serial numbers in `config.yaml`.
2. Confirm `calibration.wrist_extrinsic_file` points to the desired camera2
   hand-eye result.
3. Connect the UR5e and both cameras.
4. Put the robot at a safe stationary pose where camera2 and camera3 can both
   see the board.

## Run

From the repository root:

```bash
.venv/bin/python camera3_eye_to_hand/calibrate_camera3.py \
  --config camera3_eye_to_hand/config.yaml
```

The preview shows camera2 on the left and camera3 on the right.

| Key | Action |
|---|---|
| `s` | Save one stationary board pose, averaged over several frames |
| `d` | Delete the last saved pose |
| `c` | Calculate and print the current calibration |
| `w` | Calculate and save the calibration and raw captures |
| `q` | Quit without automatically saving |

Collect at least 15 poses; 20-40 is preferable. Move the board between captures
to vary its position, distance, and orientation while keeping it visible in
both cameras. Stop moving it before pressing `s`. The wrist may also be moved
manually between captures, but it must be stopped before capture.

## Outputs

Pressing `w` creates a new numbered run directory automatically:

```text
results/result_1/
results/result_2/
results/result_3/
...
```

Each numbered directory contains three timestamped files:

- `camera3_to_base_<timestamp>.npy`: the 4x4 `T_base_camera3` matrix.
- `camera3_to_base_<timestamp>.yaml`: readable matrix, serials, intrinsics and
  consistency metrics.
- `camera3_to_base_<timestamp>_captures.npz`: source candidates and diagnostics.

Earlier result directories are never overwritten.

### Accepted reference result

`results/result_4/` is the reviewed calibration currently selected for use.
It contains the transform matrix, readable report, and raw capture diagnostics.
Its fit used 25/25 inliers with 3.29 mm translation RMSE and 0.486 degree
rotation RMSE. New calibration runs do not replace this reference automatically.

## Validation and rollout integration

Before using the result for robot motion, repeat captures from several wrist
poses and verify that the estimated camera3 transform remains consistent.
Inspect the translation and rotation residuals in the YAML report, then perform
a held-out board-position test in the robot base frame.

The existing rollout calls its fixed-camera configuration field `T_cam_base`,
but its multiplication actually expects `T_base_camera`. When integration is
done later, copy the saved matrix as-is into that field; do not invert it.

## Offline math tests

The transform solver can be tested without cameras or a robot:

```bash
.venv/bin/python -m unittest discover \
  -s camera3_eye_to_hand/tests -v
```
