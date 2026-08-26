"""
Collect task: put_book_in_box.

Task-specific vs prior amir collect scripts:
  - File box sits at a FIXED xy on the table -- never picked up during reset.
  - Book always lands at the same xyz inside the box (fixed-z release), so
    the in-box pose is deterministic.
  - Bookshelf source has 5 DISCRETE x slots and a continuous y range within
    each slot. Each reset samples a fresh (x_slot, y) so the model sees the
    book placed at many different shelf positions.
  - Fixed-z release everywhere (no moveUntilContact): both shelf slots and
    box walls would trigger contact incorrectly during descent.

Per episode:
  start: book at (x_slot, y); file box at its fixed spot.
  TASK (recorded): pick book from shelf -> place in file box (fixed z release).
  RESET (NOT recorded): pick book off file box (fixed z regrasp) -> place at
                        new (x_slot, y) sampled from the shelf grid.
                        File box is NOT touched.

Motion parameters come from the `verify:` block of a plan yaml (default
`collect/plan_put_book_in_box.yaml`), shared with `verify_book_box.py`.

Usage:
    /home/joshua/Desktop/ur5e_vla/.venv/bin/python -u collect_book_box.py -n 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rtde_control
import rtde_receive
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "collect"))
import robotiq_gripper
from recorder import DataRecorder

RX, RY, RZ = 3.14159, 0.0, 0.0
DATA_ROOT = os.path.join(THIS_DIR, "data")
TASK_NAME = "put_book_in_box"
DEFAULT_PLAN = os.path.join(THIS_DIR, "collect", "plan_put_book_in_box.yaml")

# Subset paths are computed at runtime from --subset. Layout:
#   data/<task>/<subset>/data/<ep>/     (episode recordings)
#   data/<task>/<subset>/state.json     (resumable per-subset state)
#   data/<task>/<subset>/positions.jsonl


def sample_shelf_slot(rng, x_choices, y_min, y_max):
    """Sample a bookshelf slot: uniform pick from x_choices, uniform y in [y_min, y_max]."""
    x = float(rng.choice(x_choices))
    y = float(rng.uniform(y_min, y_max))
    return (x, y)


def load_verify_defaults(plan_path):
    """Read the `verify:` section of a plan yaml. Returns {} if missing."""
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    return plan.get("verify", {}) or {}


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return None


def save_state(state_path, state):
    with open(state_path, "w") as f:
        json.dump(state, f)


def parse_xy(s):
    return tuple(float(v) for v in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--n-episodes", "-n", type=int, default=50)

    ap.add_argument("--pick-z-book", type=float, default=0.309,
                    help="TCP z when picking book from shelf")
    ap.add_argument("--release-z-on-shelf", type=float, default=None,
                    help="TCP z when placing book back on shelf during reset "
                         "(defaults to pick_z_book)")
    ap.add_argument("--release-z-in-box", type=float, default=0.304,
                    help="TCP z when releasing book in the file box (fixed, from jog)")
    ap.add_argument("--pick-z-book-in-box", type=float, default=None,
                    help="TCP z when regrasping book off the box "
                         "(defaults to release_z_in_box)")
    ap.add_argument("--hover-z", type=float, default=0.45)
    ap.add_argument("--lift-z", type=float, default=0.50)

    ap.add_argument("--home", default="-0.10,0.55,0.35")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200)
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0)

    # Random gripper init pose at the start of each task recording.
    ap.add_argument("--init-x-min", type=float, default=-0.15)
    ap.add_argument("--init-x-max", type=float, default=+0.15)
    ap.add_argument("--init-y-min", type=float, default=+0.45)
    ap.add_argument("--init-y-max", type=float, default=+0.70)
    ap.add_argument("--init-z-min", type=float, default=+0.35)
    ap.add_argument("--init-z-max", type=float, default=+0.50)

    ap.add_argument("--config", default=os.path.join(THIS_DIR, "collect", "config.yaml"))
    ap.add_argument("--plan", default=DEFAULT_PLAN,
                    help="Plan yaml whose `verify:` section supplies defaults. "
                         "Any CLI arg overrides.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--reset-state", action="store_true",
                    help="ignore existing state; re-do initial hover placement")

    # Task-specific: file box fixed position + shelf slot parameters.
    ap.add_argument("--book", type=str, default=None, help="'x,y' override for book initial position")
    ap.add_argument("--file-box", type=str, default=None,
                    help="'x,y' for the FIXED file box position "
                         "(defaults to plan value; do not change between runs)")
    ap.add_argument("--shelf-x-choices", type=str, default=None,
                    help="comma-separated list of 5 shelf-slot x's (m). Default: from plan.")
    ap.add_argument("--shelf-y-min", type=float, default=0.725)
    ap.add_argument("--shelf-y-max", type=float, default=0.800)

    ap.add_argument("--record-depth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--subset", default="clean",
                    choices=["clean", "d1", "d2", "d3", "d4"],
                    help="Which subset to write to. Episodes land in "
                         "data/<task>/<subset>/data/<ep>/. State & positions "
                         "log are per-subset so you can resume each independently.")
    ap.add_argument("--reset-pause", type=float, default=0.0,
                    help="Extra seconds to sleep AFTER the reset motion (before "
                         "the next episode starts). Use during cluttered runs "
                         "to give yourself time to re-place obstacles by hand.")
    args = ap.parse_args()

    # Per-subset paths (derived from --subset).
    SUBSET_ROOT = os.path.join(DATA_ROOT, TASK_NAME, args.subset)
    EPISODES_DIR = os.path.join(SUBSET_ROOT, "data")
    STATE_PATH = os.path.join(SUBSET_ROOT, "state.json")
    POSITIONS_LOG = os.path.join(SUBSET_ROOT, "positions.jsonl")

    # Load defaults from the plan's `verify:` block. CLI values always win.
    # Skip `book` -- that's verify's phase-1 hover target. Applying it here
    # would trigger the "user-supplied override" branch every run and wipe
    # the resumable state file.
    PLAN_SKIP = {"book"}
    if args.plan and os.path.exists(args.plan):
        user_supplied = {a.split("=")[0].lstrip("-").replace("-", "_")
                         for a in sys.argv[1:] if a.startswith("--")}
        defaults = load_verify_defaults(args.plan)
        for key, val in defaults.items():
            attr = key.replace("-", "_")
            if attr in PLAN_SKIP:
                continue
            if hasattr(args, attr) and attr not in user_supplied:
                setattr(args, attr, val)
        print(f"[loaded verify defaults from {args.plan}]")
    else:
        print(f"[warning] plan not found: {args.plan} -- using CLI defaults only")

    if args.pick_z_book_in_box is None:
        args.pick_z_book_in_box = args.release_z_in_box
    if args.release_z_on_shelf is None:
        args.release_z_on_shelf = args.pick_z_book

    # Normalise shelf_x_choices: could be list (from yaml) or str (from CLI).
    if isinstance(args.shelf_x_choices, str):
        args.shelf_x_choices = [float(v) for v in args.shelf_x_choices.split(",")]
    elif args.shelf_x_choices is None:
        args.shelf_x_choices = [0.080, 0.014, -0.053, -0.124, -0.189]

    file_box = parse_xy(args.file_box) if args.file_box else parse_xy("0.211,0.700")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cam_serials = cfg["recording"]["camera_serials"]
    rec_hz = cfg["recording"]["hz"]
    cam_res = tuple(cfg["recording"]["camera_resolution"])

    hx, hy, hz_home = (float(v) for v in args.home.split(","))
    rng = np.random.default_rng(args.seed)

    os.makedirs(EPISODES_DIR, exist_ok=True)

    params = {
        "task": TASK_NAME,
        "subset": args.subset,
        "reset_pause": args.reset_pause,
        "plan": args.plan,
        "pick_z_book": args.pick_z_book,
        "release_z_on_shelf": args.release_z_on_shelf,
        "release_z_in_box": args.release_z_in_box,
        "pick_z_book_in_box": args.pick_z_book_in_box,
        "hover_z": args.hover_z,
        "lift_z": args.lift_z,
        "home": args.home,
        "speed": args.speed,
        "accel": args.accel,
        "grip_pos": args.grip_pos,
        "transit_grip_pos": args.transit_grip_pos,
        "file_box": list(file_box),                    # FIXED for the task
        "shelf_x_choices": list(args.shelf_x_choices),
        "shelf_y_min": args.shelf_y_min,
        "shelf_y_max": args.shelf_y_max,
        "record_depth": args.record_depth,
    }
    print(json.dumps(params, indent=2))
    print(f"state file    -> {STATE_PATH}")
    print(f"positions log -> {POSITIONS_LOG}")
    print(f"cameras       -> {cam_serials}")

    print(f"Connecting to UR5e at {args.robot_ip} ...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    grip = robotiq_gripper.RobotiqGripper()
    grip.connect(args.robot_ip, 63352)
    grip.activate()
    grip.move_and_wait_for_pos(args.transit_grip_pos, 255, 255)

    recorder = DataRecorder(
        rtde_r=rtde_r, gripper=grip,
        camera_serials=cam_serials, hz=rec_hz, camera_resolution=cam_res,
        record_depth=args.record_depth,
    )
    recorder.test()

    def go(x, y, z):
        rtde_c.moveL([x, y, z, RX, RY, RZ], args.speed, args.accel)

    def pick(label, x, y, pick_z=None):
        pz = args.pick_z_book if pick_z is None else pick_z
        hz = max(args.hover_z, pz + 0.10)
        print(f"  PICK  {label} ({x:+.3f},{y:+.3f}) z={pz:.3f} hover={hz:.3f}")
        go(x, y, hz)
        grip.move_and_wait_for_pos(0, 125, 125)
        go(x, y, pz)
        grip.move_and_wait_for_pos(args.grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    def place_fixed_z(label, x, y, release_z):
        """Fixed-z release (deterministic; no moveUntilContact -- would false-trip
        on shelf slot walls / box rim)."""
        hz = max(args.hover_z, release_z + 0.10)
        print(f"  PLACE {label} ({x:+.3f},{y:+.3f}) hover={hz:.3f} release_z={release_z:.3f} (fixed)")
        go(x, y, hz)
        go(x, y, release_z)
        grip.move_and_wait_for_pos(args.transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    def record_motion(ep_idx, motion_fn, meta):
        out_dir = os.path.join(EPISODES_DIR, str(ep_idx))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        recorder.start(out_dir)
        try:
            motion_fn()
        finally:
            recorder.stop()
        print(f"  -> recorded to {out_dir}")

    def sample_init():
        return (
            float(rng.uniform(args.init_x_min, args.init_x_max)),
            float(rng.uniform(args.init_y_min, args.init_y_max)),
            float(rng.uniform(args.init_z_min, args.init_z_max)),
        )

    # ---------- initialise state ----------
    if args.book:
        book = parse_xy(args.book)
        state = {"book": list(book), "file_box": list(file_box)}
        print(f"\n[using user-supplied initial book position]")
        print(f"  book     @ ({book[0]:+.3f},{book[1]:+.3f})")
        print(f"  file box @ ({file_box[0]:+.3f},{file_box[1]:+.3f})  [FIXED]")
        save_state(STATE_PATH, state)
    elif args.start_episode > 1 and not args.reset_state and os.path.exists(STATE_PATH):
        state = load_state(STATE_PATH)
        print(f"\n[resuming from {STATE_PATH}] state={state}")
    else:
        book = sample_shelf_slot(rng, args.shelf_x_choices, args.shelf_y_min, args.shelf_y_max)
        print(f"\n=== INITIAL placement (manual) ===")
        print(f"  hover at book: ({book[0]:+.3f},{book[1]:+.3f}) -- {args.hover_pause}s")
        go(book[0], book[1], args.hover_z)
        time.sleep(args.hover_pause)
        state = {"book": list(book), "file_box": list(file_box)}
        save_state(STATE_PATH, state)

    # ---------- main loop ----------
    try:
        for ep in range(args.start_episode, args.start_episode + args.n_episodes):
            print(f"\n{'='*60}\nEPISODE {ep}\n{'='*60}")
            book = tuple(state["book"])

            book_new = sample_shelf_slot(rng, args.shelf_x_choices, args.shelf_y_min, args.shelf_y_max)
            init_pose = sample_init()
            print(f"start: book={book}  file_box={file_box} [fixed]")
            print(f"reset target: book_new={book_new}")
            print(f"init {tuple(round(v, 3) for v in init_pose)}")

            with open(POSITIONS_LOG, "a") as f:
                f.write(json.dumps({
                    "episode": ep,
                    "start": {"book": list(book)},
                    "fixed": {"file_box": list(file_box)},
                    "reset_targets": {"book_new": list(book_new)},
                    "init_pose": list(init_pose),
                    "timestamp": time.time(),
                }) + "\n")

            # ----- RECORDED motion: pick book, place in file box -----
            print(f"--- {TASK_NAME} ---")
            go(*init_pose); time.sleep(0.5)

            def task_motion():
                pick("book", *book, pick_z=args.pick_z_book)
                place_fixed_z("book_in_box", *file_box, release_z=args.release_z_in_box)
                go(hx, hy, hz_home)

            meta = {
                **params,
                "episode": ep,
                "start": {"book": list(book)},
                "end":   {"book": list(file_box)},
                "init_pose": list(init_pose),
            }
            record_motion(ep, task_motion, meta)

            # ----- RESET motion (NOT recorded): file box stays put -----
            print(f"--- RESET (not recorded) ---")
            pick("book_from_box", *file_box, pick_z=args.pick_z_book_in_box)
            place_fixed_z("book_new", *book_new, release_z=args.release_z_on_shelf)
            go(hx, hy, hz_home)

            state = {"book": list(book_new), "file_box": list(file_box)}
            save_state(STATE_PATH, state)

            # Extra pause after the reset lets the operator physically re-place
            # obstacles for the next cluttered episode.
            if args.reset_pause > 0:
                print(f"  [reset-pause] sleeping {args.reset_pause}s -- move obstacles now")
                time.sleep(args.reset_pause)

        print(f"\nDONE: {args.n_episodes} episodes collected.")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving state...")
        save_state(STATE_PATH, state)
    except Exception as e:
        print(f"\nERROR: {e}. Saving state before exit...")
        save_state(STATE_PATH, state)
        raise
    finally:
        try:
            recorder.stop(); recorder.close()
        except Exception:
            pass
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
