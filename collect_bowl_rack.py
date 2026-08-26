"""
Collect task: put_bowl_on_rack.

Task-specific vs other amir collect scripts:
  - Dish rack sits at a FIXED xy on the table -- never picked up during reset.
  - Bowl always lands at the same xyz on the rack (from arm jog), so the
    on-rack pose is deterministic (fixed-z release, not moveUntilContact).
  - Bowl xy for each episode is sampled uniformly inside a small disk
    (default 4 cm radius) around a fixed center, so the arm never risks
    hitting the still-standing rack during reset placement.
  - Only ONE object varies episode-to-episode (the bowl), so state/positions
    log carry only the bowl xy plus the fixed rack position for eval.

Per episode:
  start: bowl at some xy in the jitter disk; rack at its fixed spot.
  TASK (recorded): pick bowl -> place on rack (fixed release z).
  RESET (NOT recorded): pick bowl off rack -> place at bowl_new (sampled
                        in the disk); moveUntilContact on the table.
                        Rack is NOT touched.

Motion parameters come from the `verify:` block of a plan yaml (default
`collect/plan_put_bowl_on_rack.yaml`), shared with `verify_bowl_rack.py`
so verify + collect always agree.

Usage:
    /home/joshua/Desktop/ur5e_vla/.venv/bin/python -u collect_bowl_rack.py -n 50
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
TASK_NAME = "put_bowl_on_rack"
DEFAULT_PLAN = os.path.join(THIS_DIR, "collect", "plan_put_bowl_on_rack.yaml")

# Subset paths are computed at runtime from --subset. Layout:
#   data/<task>/<subset>/data/<ep>/     (episode recordings)
#   data/<task>/<subset>/state.json     (resumable per-subset state)
#   data/<task>/<subset>/positions.jsonl


def sample_bowl_in_disk(rng, center, radius):
    """Sample bowl xy uniformly in a disk of `radius` around `center`.
    Uses r = R*sqrt(u) so density is uniform over the disk."""
    r = radius * float(np.sqrt(rng.uniform()))
    theta = float(rng.uniform(0.0, 2 * np.pi))
    return (center[0] + r * np.cos(theta), center[1] + r * np.sin(theta))


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

    ap.add_argument("--pick-z-table", type=float, default=0.122,
                    help="TCP z when picking bowl from table")
    ap.add_argument("--release-z-on-rack", type=float, default=0.149,
                    help="TCP z when releasing bowl on the rack (fixed, from jog)")
    ap.add_argument("--pick-z-bowl-on-rack", type=float, default=None,
                    help="TCP z when regrasping bowl off the rack "
                         "(defaults to release_z_on_rack)")
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--lift-z", type=float, default=0.30)

    ap.add_argument("--home", default="-0.10,0.55,0.25")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--descend-speed", type=float, default=0.05)
    ap.add_argument("--descend-accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200)
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0)

    # Random gripper init pose at the start of each task recording.
    ap.add_argument("--init-x-min", type=float, default=-0.15)
    ap.add_argument("--init-x-max", type=float, default=+0.15)
    ap.add_argument("--init-y-min", type=float, default=+0.45)
    ap.add_argument("--init-y-max", type=float, default=+0.70)
    ap.add_argument("--init-z-min", type=float, default=+0.25)
    ap.add_argument("--init-z-max", type=float, default=+0.40)

    ap.add_argument("--config", default=os.path.join(THIS_DIR, "collect", "config.yaml"))
    ap.add_argument("--plan", default=DEFAULT_PLAN,
                    help="Plan yaml whose `verify:` section supplies defaults. "
                         "Any CLI arg overrides.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--reset-state", action="store_true",
                    help="ignore existing state; re-do initial hover placement")

    # Task-specific: dish rack fixed position + bowl-jitter parameters.
    ap.add_argument("--bowl", type=str, default=None,
                    help="'x,y' override for bowl initial position")
    ap.add_argument("--dish-rack", type=str, default=None,
                    help="'x,y' for the FIXED dish rack position "
                         "(defaults to plan value; do not change between runs)")
    ap.add_argument("--bowl-center", type=str, default=None,
                    help="'x,y' center of the bowl-jitter disk")
    ap.add_argument("--bowl-jitter-radius", type=float, default=0.04,
                    help="radius (m) of the disk in which bowl_new is sampled each reset")

    ap.add_argument("--record-depth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--subset", default="clean",
                    choices=["clean", "d1", "d2", "d3", "d4"],
                    help="Which subset to write to. Episodes land in "
                         "data/<task>/<subset>/data/<ep>/. State & positions "
                         "log are per-subset so you can resume each independently.")
    ap.add_argument("--reset-pause", type=float, default=0.0,
                    help="Extra seconds to sleep AFTER the reset motion (before "
                         "the next episode). Use during cluttered runs to give "
                         "yourself time to re-place obstacles by hand.")
    args = ap.parse_args()

    # Per-subset paths (derived from --subset).
    SUBSET_ROOT = os.path.join(DATA_ROOT, TASK_NAME, args.subset)
    EPISODES_DIR = os.path.join(SUBSET_ROOT, "data")
    STATE_PATH = os.path.join(SUBSET_ROOT, "state.json")
    POSITIONS_LOG = os.path.join(SUBSET_ROOT, "positions.jsonl")

    # Load defaults from the plan's `verify:` block. CLI values always win.
    # Skip `bowl` -- that's verify's phase-1 hover target. Applying it here
    # would trigger the "user-supplied override" branch on every run and
    # wipe the resumable state file.
    PLAN_SKIP = {"bowl"}
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

    if args.pick_z_bowl_on_rack is None:
        args.pick_z_bowl_on_rack = args.release_z_on_rack

    dish_rack = parse_xy(args.dish_rack) if args.dish_rack else parse_xy("0.172,0.819")
    bowl_center = parse_xy(args.bowl_center) if args.bowl_center else dish_rack  # fallback
    if args.bowl_center is None:
        # Prefer bowl_center from the plan; fall back to plan's initial bowl xy.
        plan_defaults = load_verify_defaults(args.plan) if args.plan else {}
        bowl_center = parse_xy(plan_defaults.get("bowl_center") or plan_defaults.get("bowl") or "-0.003,0.569")

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
        "pick_z_table": args.pick_z_table,
        "release_z_on_rack": args.release_z_on_rack,
        "pick_z_bowl_on_rack": args.pick_z_bowl_on_rack,
        "hover_z": args.hover_z,
        "lift_z": args.lift_z,
        "home": args.home,
        "speed": args.speed,
        "accel": args.accel,
        "descend_speed": args.descend_speed,
        "descend_accel": args.descend_accel,
        "grip_pos": args.grip_pos,
        "transit_grip_pos": args.transit_grip_pos,
        "dish_rack": list(dish_rack),          # FIXED for the task
        "bowl_center": list(bowl_center),
        "bowl_jitter_radius": args.bowl_jitter_radius,
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

    def descend_contact():
        rtde_c.moveUntilContact(
            [0.0, 0.0, -args.descend_speed, 0.0, 0.0, 0.0],
            [0.0] * 6,
            args.descend_accel,
        )

    def pick(label, x, y, pick_z=None):
        pz = args.pick_z_table if pick_z is None else pick_z
        hz = max(args.hover_z, pz + 0.07)
        print(f"  PICK  {label} ({x:+.3f},{y:+.3f}) z={pz:.3f} hover={hz:.3f}")
        go(x, y, hz)
        grip.move_and_wait_for_pos(0, 125, 125)
        go(x, y, pz)
        grip.move_and_wait_for_pos(args.grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    def place_on_rack(label, x, y, release_z):
        """Fixed-z release on the rack (deterministic landing, matches jog)."""
        hz = max(args.hover_z, release_z + 0.10)
        print(f"  PLACE {label} ({x:+.3f},{y:+.3f}) hover={hz:.3f} release_z={release_z:.3f} (fixed)")
        go(x, y, hz)
        go(x, y, release_z)
        grip.move_and_wait_for_pos(args.transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    def place_on_table(label, x, y):
        """moveUntilContact release on the table."""
        hz = args.hover_z
        print(f"  PLACE {label} ({x:+.3f},{y:+.3f}) hover={hz:.3f} -> moveUntilContact")
        go(x, y, hz)
        descend_contact()
        z_now = rtde_r.getActualTCPPose()[2]
        print(f"    contact at z={z_now:.4f}")
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
    if args.bowl:
        bowl = parse_xy(args.bowl)
        state = {"bowl": list(bowl), "dish_rack": list(dish_rack)}
        print(f"\n[using user-supplied initial bowl position]")
        print(f"  bowl      @ ({bowl[0]:+.3f},{bowl[1]:+.3f})")
        print(f"  dish rack @ ({dish_rack[0]:+.3f},{dish_rack[1]:+.3f})  [FIXED]")
        save_state(STATE_PATH, state)
    elif args.start_episode > 1 and not args.reset_state and os.path.exists(STATE_PATH):
        state = load_state(STATE_PATH)
        print(f"\n[resuming from {STATE_PATH}] state={state}")
    else:
        bowl = sample_bowl_in_disk(rng, bowl_center, args.bowl_jitter_radius)
        print(f"\n=== INITIAL placement (manual) ===")
        print(f"  hover at bowl: ({bowl[0]:+.3f},{bowl[1]:+.3f}) -- {args.hover_pause}s")
        go(bowl[0], bowl[1], args.hover_z)
        time.sleep(args.hover_pause)
        state = {"bowl": list(bowl), "dish_rack": list(dish_rack)}
        save_state(STATE_PATH, state)

    # ---------- main loop ----------
    try:
        for ep in range(args.start_episode, args.start_episode + args.n_episodes):
            print(f"\n{'='*60}\nEPISODE {ep}\n{'='*60}")
            bowl = tuple(state["bowl"])

            bowl_new = sample_bowl_in_disk(rng, bowl_center, args.bowl_jitter_radius)
            init_pose = sample_init()
            print(f"start: bowl={bowl}  dish_rack={dish_rack} [fixed]")
            print(f"reset target: bowl_new={bowl_new}")
            print(f"init {tuple(round(v, 3) for v in init_pose)}")

            with open(POSITIONS_LOG, "a") as f:
                f.write(json.dumps({
                    "episode": ep,
                    "start": {"bowl": list(bowl)},
                    "fixed": {"dish_rack": list(dish_rack)},
                    "reset_targets": {"bowl_new": list(bowl_new)},
                    "init_pose": list(init_pose),
                    "timestamp": time.time(),
                }) + "\n")

            # ----- RECORDED motion: pick bowl, place on rack -----
            print(f"--- {TASK_NAME} ---")
            go(*init_pose); time.sleep(0.5)

            def task_motion():
                pick("bowl", *bowl, pick_z=args.pick_z_table)
                place_on_rack("bowl_on_rack", *dish_rack, release_z=args.release_z_on_rack)
                go(hx, hy, hz_home)

            meta = {
                **params,
                "episode": ep,
                "start": {"bowl": list(bowl)},
                "end":   {"bowl": list(dish_rack)},
                "init_pose": list(init_pose),
            }
            record_motion(ep, task_motion, meta)

            # ----- RESET motion (NOT recorded): rack stays put -----
            print(f"--- RESET (not recorded) ---")
            pick("bowl_from_rack", *dish_rack, pick_z=args.pick_z_bowl_on_rack)
            place_on_table("bowl_new", *bowl_new)
            go(hx, hy, hz_home)

            state = {"bowl": list(bowl_new), "dish_rack": list(dish_rack)}
            save_state(STATE_PATH, state)

            # Extra pause after reset lets the operator re-place obstacles
            # for the next cluttered episode.
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
