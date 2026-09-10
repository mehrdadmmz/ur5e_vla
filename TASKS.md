# Adding a New Collection Task

Guide for adding a new manipulation task using the standalone `collect_<task>.py` pattern.

## What already works (don't touch)

The shared infrastructure supports depth + N cameras + HDF5 conversion out of the box. Adding a task does **not** require changes to:

- `collect/recorder.py` — supports `record_depth=True` (aligned depth PNGs + `intrinsics.json`)
- `data/convert_to_hdf5.py` — picks up depth + intrinsics automatically
- `test_cam.py` — auto-detects connected RealSense cameras

## Per-task files to create

For a new task called e.g. `put_block_on_plate`:

| File | Purpose |
|------|---------|
| `verify_<objects>.py` | Interactive setup: hover for placement + one full trial cycle. Prints end-state coords for seeding collect. |
| `collect/plan_<task>.yaml` | Config. The `verify:` block is source of truth for heights (pick z's, hover z, lift z, geometry, initial xy). Read by both verify and collect. |
| `collect_<task>.py` | Standalone collect loop. Copy an existing collect script as template; edit motion sequence for the new task. |

Also update:

| File | Change |
|------|--------|
| `collect/log.json` | Add `"<task>": 0` counter entry |

## Runtime artifacts (auto-generated, do not commit)

Each collect run creates under `data/<task>/<subset>/`:

- `data/<ep>/` — per-episode raw folder (color + depth PNGs, `state.csv`, `meta.json`, `intrinsics.json`)
- `state.json` — last object positions for resuming without re-verify
- `positions.jsonl` — per-episode start/reset positions log

After conversion:
- `data/<ep>/episode<ep>.hdf5` — HDF5 file inside the episode directory

These are all covered by `.gitignore`.

Calibration attempts under `camera3_eye_to_hand/results/` are also ignored by
default. Force-add only a reviewed, accepted result when it should become a
versioned reference calibration.

## Flow to add + run a new task

1. **Write** verify + plan + collect script (copy existing files as templates).
2. **Verify** the setup:
   ```bash
   .venv/bin/python verify_<objects>.py --plan collect/plan_<task>.yaml
   ```
   Physically place objects during phase-1 hover. Note the end-state coords printed at the end.
3. **Collect clean** (100 episodes):
   ```bash
   .venv/bin/python -u collect_<task>.py -n 100 --<obj1>=X,Y --<obj2>=X,Y
   ```
   Objects seed from CLI (matches verify's end-state); subsequent runs use `--start-episode N` and resume from the auto-saved state file.
4. **Collect cluttered** (25 episodes × 4 subsets):
   ```bash
   .venv/bin/python -u collect_<task>.py --subset d1 -n 25 --<obj1>=X,Y --<obj2>=X,Y --reset-pause 10
   ```
   Place obstacles on the table. `--reset-pause` gives you time to reposition obstacles between episodes.
5. **Convert to HDF5**:
   ```bash
   .venv/bin/python convert_ep_hdf5.py --task <task> --subsets clean,d1,d2,d3,d4
   ```

For `place_three_cups_in_bowls`, run the additional read-only dataset audit
after collection and conversion:

```bash
.venv/bin/python audit_three_cups_dataset.py --expect-rewritten
```

## Design guardrails

- Keep the plan yaml's `verify:` block as **the** source of truth for heights. Both verify and collect read it via `load_verify_defaults()`, so tuning happens in one place.
- Don't auto-copy initial object xy from the plan into collect's argparse — those are verify's phase-1 hover targets. Letting them land in collect's args triggers the "user CLI override" branch and wipes the resumable state file. Filter them out via `PLAN_SKIP`.
- Depth values are raw uint16 (RealSense depth units). Multiply by `camera_info/camera{i}/@depth_scale` (in the HDF5) to get meters. Mask `65535` as invalid.
- Frame timing is ~19 Hz real at 20 Hz requested, with occasional 300–600 ms hiccups from disk I/O on the 6 PNG writes per frame (3 color + 3 depth). Timestamps are preserved.
