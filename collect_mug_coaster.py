"""
Collect task: put_mug_on_coaster.

Same overall flow as `collect_cup_bowl.py` -- pick object, place on
target, reset for the next round -- but the target is a flat coaster
(~1.5 cm thick), not a bowl. moveUntilContact hits the coaster surface,
then the gripper releases the mug on top.

Per episode:
  start:  mug + coaster on table at 2 random separated positions
  TASK (recorded): pick mug -> place on coaster (moveUntilContact)
  RESET (NOT recorded): pick mug off coaster -> place at mug_new
                        pick coaster (at pick_z_coaster) -> place at coaster_new
                        return home

Motion parameters come from the `verify:` block of a plan yaml (default
`collect/plan_put_mug_on_coaster.yaml`), shared with
`verify_mug_coaster.py` so verify + collect always agree on numbers.

Usage:
    /home/joshua/Desktop/ur5e_vla/.venv/bin/python -u collect_mug_coaster.py -n 50
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
TASK_NAME = "put_mug_on_coaster"
DEFAULT_PLAN = os.path.join(THIS_DIR, "collect", "plan_put_mug_on_coaster.yaml")

# Subset paths are computed at runtime from --subset. Layout:
#   data/<task>/<subset>/data/<ep>/     (episode recordings)
#   data/<task>/<subset>/state.json     (resumable per-subset state)
#   data/<task>/<subset>/positions.jsonl

# Workspace trapezoid (matches verify_mug_coaster.py).
YMIN, YMAX = 0.50, 0.80
HW_CLOSE, HW_FAR = 0.10, 0.18
MUG_RADIUS = 0.04
COASTER_RADIUS = 0.05
MIN_MUG_COASTER = 0.14
MIN_RESET_SHIFT = 0.10
MIN_MUG_TO_PREV_COASTER = 0.20


def halfwidth(y):
    return HW_CLOSE + (y - YMIN) / (YMAX - YMIN) * (HW_FAR - HW_CLOSE)


def in_trapezoid(p):
    x, y = p
    return YMIN <= y <= YMAX and abs(x) <= halfwidth(y)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def sample_in_trapezoid(rng):
    y = rng.uniform(YMIN, YMAX)
    hw = halfwidth(y)
    return float(rng.uniform(-hw, hw)), float(y)


def sample_pair(rng, max_outer=2000, max_inner=300,
                prev_mug=None, prev_coaster=None,
                min_shift=MIN_RESET_SHIFT,
                min_mug_to_prev_coaster=MIN_MUG_TO_PREV_COASTER):
    """Sample a (mug, coaster) pair in the trapezoid.

    Matches verify_mug_coaster.py: new poses must each be >= min_shift from
    their previous position, and the new mug must be >= min_mug_to_prev_coaster
    from the still-standing coaster (mug is placed BEFORE the coaster is
    moved during reset).
    """
    for _ in range(max_outer):
        mug = sample_in_trapezoid(rng)
        if prev_mug is not None and dist(mug, prev_mug) < min_shift:
            continue
        if prev_coaster is not None and dist(mug, prev_coaster) < min_mug_to_prev_coaster:
            continue
        for _ in range(max_inner):
            coaster = sample_in_trapezoid(rng)
            if prev_coaster is not None and dist(coaster, prev_coaster) < min_shift:
                continue
            if dist(mug, coaster) >= MIN_MUG_COASTER:
                return mug, coaster
    raise RuntimeError("could not sample mug/coaster pair")


def sample_reset_targets(rng, prev_mug, prev_coaster):
    """Sample (mug_new, coaster_new) with progressive constraint relaxation.

    Falls back through looser levels if the strict constraints are infeasible
    for the current workspace geometry. Only MIN_MUG_COASTER (safety) is never
    relaxed; the pose-diversity constraints are.
    """
    levels = [
        (MIN_RESET_SHIFT, MIN_MUG_TO_PREV_COASTER, "strict"),
        (0.05,            0.16,                     "medium"),
        (0.03,            MIN_MUG_COASTER,          "loose"),
        (0.00,            MIN_MUG_COASTER,          "safety-only"),
    ]
    last_err = None
    for min_shift, min_mug_to_prev_coaster, label in levels:
        try:
            pair = sample_pair(
                rng,
                prev_mug=prev_mug, prev_coaster=prev_coaster,
                min_shift=min_shift,
                min_mug_to_prev_coaster=min_mug_to_prev_coaster,
            )
            if label != "strict":
                print(f"  [sampler] fell back to '{label}' constraints "
                      f"(min_shift={min_shift}, min_mug_to_prev_coaster={min_mug_to_prev_coaster})")
            return pair
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"could not sample reset targets at any level: {last_err}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--n-episodes", "-n", type=int, default=50)

    # Geometry defaults (overridden by plan verify: section unless CLI-provided).
    ap.add_argument("--mug-height", type=float, default=0.10)
    ap.add_argument("--coaster-thickness", type=float, default=0.015)

    # Pick / hover / lift z (overridden by plan verify: section).
    ap.add_argument("--pick-z-table", type=float, default=0.180,
                    help="TCP z when picking the mug from the table")
    ap.add_argument("--pick-z-coaster", type=float, default=0.140,
                    help="TCP z when picking the coaster during reset")
    ap.add_argument("--pick-z-mug-on-coaster", type=float, default=None,
                    help="TCP z when picking mug back off the coaster "
                         "(defaults to pick_z_table + coaster_thickness)")
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--lift-z", type=float, default=0.30)

    ap.add_argument("--home", default="-0.10,0.55,0.25")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--descend-speed", type=float, default=0.05)
    ap.add_argument("--descend-accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200,
                    help="Robotiq target when closing. Bump up if slipping.")
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0,
                    help="seconds hovering over each object at initial manual placement")

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
                         "Any CLI arg overrides the plan's value.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--reset-state", action="store_true",
                    help="ignore existing state; re-do initial hover placement")
    ap.add_argument("--mug", type=str, default=None,
                    help="'x,y' override for mug initial position")
    ap.add_argument("--coaster", type=str, default=None,
                    help="'x,y' override for coaster initial position")
    ap.add_argument("--record-depth", action=argparse.BooleanOptionalAction, default=True,
                    help="Also record aligned depth (uint16 PNGs + intrinsics.json). "
                         "Doubles disk footprint; needed for point-cloud reconstruction.")
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
    # Skip `mug` / `coaster` -- those are verify's phase-1 hover targets. If we
    # applied them here, the "user-supplied override" branch below would fire
    # on every run and wipe the resumable state file.
    PLAN_SKIP = {"mug", "coaster"}
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

    if args.pick_z_mug_on_coaster is None:
        args.pick_z_mug_on_coaster = args.pick_z_table + args.coaster_thickness

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cam_serials = cfg["recording"]["camera_serials"]
    rec_hz = cfg["recording"]["hz"]
    cam_res = tuple(cfg["recording"]["camera_resolution"])

    hx, hy, hz_home = (float(v) for v in args.home.split(","))
    rng = np.random.default_rng(args.seed)

    os.makedirs(EPISODES_DIR, exist_ok=True)

    # Effective parameter set -- saved into every episode's meta.json.
    params = {
        "task": TASK_NAME,
        "subset": args.subset,
        "reset_pause": args.reset_pause,
        "plan": args.plan,
        "mug_height": args.mug_height,
        "coaster_thickness": args.coaster_thickness,
        "pick_z_table": args.pick_z_table,
        "pick_z_coaster": args.pick_z_coaster,
        "pick_z_mug_on_coaster": args.pick_z_mug_on_coaster,
        "hover_z": args.hover_z,
        "lift_z": args.lift_z,
        "home": args.home,
        "speed": args.speed,
        "accel": args.accel,
        "descend_speed": args.descend_speed,
        "descend_accel": args.descend_accel,
        "grip_pos": args.grip_pos,
        "transit_grip_pos": args.transit_grip_pos,
        "min_mug_coaster": MIN_MUG_COASTER,
        "min_reset_shift": MIN_RESET_SHIFT,
        "min_mug_to_prev_coaster": MIN_MUG_TO_PREV_COASTER,
        "record_depth": args.record_depth,
        "workspace": {"ymin": YMIN, "ymax": YMAX,
                      "hw_close": HW_CLOSE, "hw_far": HW_FAR},
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

    def place(label, x, y, surface_z=0.0):
        hz = max(args.hover_z, surface_z + 0.20)
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
    if args.mug and args.coaster:
        mug = tuple(float(v) for v in args.mug.split(","))
        coaster = tuple(float(v) for v in args.coaster.split(","))
        state = {"mug": list(mug), "coaster": list(coaster)}
        print(f"\n[using user-supplied initial positions]")
        print(f"  mug     @ ({mug[0]:+.3f},{mug[1]:+.3f})")
        print(f"  coaster @ ({coaster[0]:+.3f},{coaster[1]:+.3f})")
        save_state(STATE_PATH, state)
    elif args.start_episode > 1 and not args.reset_state and os.path.exists(STATE_PATH):
        state = load_state(STATE_PATH)
        print(f"\n[resuming from {STATE_PATH}] state={state}")
    else:
        mug, coaster = sample_pair(rng)
        print(f"\n=== INITIAL placement (manual) ===")
        for label, p in [("mug", mug), ("coaster", coaster)]:
            print(f"  hover at {label}: ({p[0]:+.3f},{p[1]:+.3f}) -- {args.hover_pause}s")
            go(p[0], p[1], args.hover_z)
            time.sleep(args.hover_pause)
        state = {"mug": list(mug), "coaster": list(coaster)}
        save_state(STATE_PATH, state)

    # ---------- main loop ----------
    try:
        for ep in range(args.start_episode, args.start_episode + args.n_episodes):
            print(f"\n{'='*60}\nEPISODE {ep}\n{'='*60}")
            mug = tuple(state["mug"])
            coaster = tuple(state["coaster"])

            mug_new, coaster_new = sample_reset_targets(rng, prev_mug=mug, prev_coaster=coaster)
            init_pose = sample_init()
            print(f"start: mug={mug}  coaster={coaster}")
            print(f"reset targets: mug_new={mug_new}  coaster_new={coaster_new}")
            print(f"init {tuple(round(v, 3) for v in init_pose)}")

            with open(POSITIONS_LOG, "a") as f:
                f.write(json.dumps({
                    "episode": ep,
                    "start": {"mug": list(mug), "coaster": list(coaster)},
                    "reset_targets": {"mug_new": list(mug_new), "coaster_new": list(coaster_new)},
                    "init_pose": list(init_pose),
                    "timestamp": time.time(),
                }) + "\n")

            # ----- RECORDED motion: pick mug, place on coaster -----
            print(f"--- {TASK_NAME} ---")
            go(*init_pose); time.sleep(0.5)

            def task_motion():
                pick("mug", *mug, pick_z=args.pick_z_table)
                place("mug_on_coaster", *coaster, surface_z=args.coaster_thickness)
                go(hx, hy, hz_home)

            meta = {
                **params,
                "episode": ep,
                "start": {"mug": list(mug), "coaster": list(coaster)},
                "end":   {"mug": list(coaster), "coaster": list(coaster)},
                "init_pose": list(init_pose),
            }
            record_motion(ep, task_motion, meta)

            # ----- RESET motion (NOT recorded): randomize for next ep -----
            print(f"--- RESET (not recorded) ---")
            pick("mug_from_coaster", *coaster, pick_z=args.pick_z_mug_on_coaster)
            place("mug_new", *mug_new, surface_z=0.0)
            pick("coaster", *coaster, pick_z=args.pick_z_coaster)
            place("coaster_new", *coaster_new, surface_z=0.0)
            go(hx, hy, hz_home)

            state = {"mug": list(mug_new), "coaster": list(coaster_new)}
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
