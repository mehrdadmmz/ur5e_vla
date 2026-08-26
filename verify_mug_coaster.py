"""
Verify mug+coaster-task setup before collecting.

  Phase 1 -- hover at sampled mug + coaster positions so the user places
             the physical objects under the gripper.
  Phase 2 -- run ONE recorded-style motion (pick mug, place on coaster) +
             the reset motion (pick mug back off coaster, move coaster),
             so the verify ends in a fresh separated state ready for
             collection.

Coaster is 10 cm diameter -- wider than the Robotiq max open (85mm), so the
grip on the coaster is friction against the outside wall (same as bowl in
cup+bowl task). Very sensitive to centering (~5 mm).

Shares config with the collect script via `--plan <plan_yaml>`: defaults for
heights, pick z's, initial xy, hover pause, etc. are loaded from the plan's
`verify:` block. Any CLI flag still overrides. Same block is read by
`collect_mug_coaster.py` so verify + collect always agree on numbers.

The end-state (final mug + coaster xy) is printed and saved to
/tmp/mug_coaster_trial_end.json -- paste those coords into
`collect_mug_coaster.py --mug=... --coaster=...` to seed collection.
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

RX, RY, RZ = 3.14159, 0.0, 0.0

YMIN, YMAX = 0.50, 0.80
HW_CLOSE, HW_FAR = 0.10, 0.18
MUG_RADIUS = 0.04
COASTER_RADIUS = 0.05
MIN_MUG_COASTER = 0.14        # separation at spawn
MIN_RESET_SHIFT = 0.10        # new pose must be at least this far from its prior pose
MIN_MUG_TO_PREV_COASTER = 0.20  # new mug must clear the still-standing coaster


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


def load_verify_defaults(plan_path):
    """Read the `verify:` section of a plan yaml. Returns {} if missing."""
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    return plan.get("verify", {}) or {}


def sample_pair(rng, max_outer=2000, max_inner=300,
                prev_mug=None, prev_coaster=None,
                min_shift=MIN_RESET_SHIFT,
                min_mug_to_prev_coaster=MIN_MUG_TO_PREV_COASTER):
    """Sample a (mug, coaster) pair in the trapezoid.

    If prev_mug / prev_coaster are given, the new poses must each be at least
    `min_shift` metres from their previous position. Additionally, the new
    mug must be at least `min_mug_to_prev_coaster` metres from the *previous*
    coaster position, since during reset the mug is placed before the coaster
    is moved -- we need clearance so the mug doesn't collide with the still-
    standing coaster.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--mug-height", type=float, default=0.10)
    ap.add_argument("--coaster-thickness", type=float, default=0.015)
    ap.add_argument("--pick-z-table", type=float, default=0.180)
    ap.add_argument("--pick-z-coaster", type=float, default=0.140,
                    help="TCP z when grasping the coaster during reset")
    ap.add_argument("--pick-z-mug-on-coaster", type=float, default=None,
                    help="TCP z when regrasping the mug from on top of the coaster "
                         "(defaults to pick_z_table + coaster_thickness)")
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--lift-z", type=float, default=0.30)
    ap.add_argument("--home", default="-0.10,0.55,0.25")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--descend-speed", type=float, default=0.05)
    ap.add_argument("--descend-accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200,
                    help="Robotiq target when closing (0=open, 255=fully closed). "
                         "Bump up if mug or coaster slips.")
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0,
                    help="seconds at each phase-1 hover for manual placement")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mug", type=str, default=None, help="'x,y' override")
    ap.add_argument("--coaster", type=str, default=None, help="'x,y' override")
    ap.add_argument("--plan", type=str, default=None,
                    help="Path to a collect plan yaml. Reads defaults from its "
                         "`verify:` section so the same file drives verify and "
                         "the collect script. Any CLI arg still overrides.")
    args = ap.parse_args()

    if args.plan:
        defaults = load_verify_defaults(args.plan)
        user_supplied = {a.split("=")[0].lstrip("-").replace("-", "_")
                         for a in sys.argv[1:] if a.startswith("--")}
        for key, val in defaults.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr) and attr not in user_supplied:
                setattr(args, attr, val)

    if args.pick_z_mug_on_coaster is None:
        args.pick_z_mug_on_coaster = args.pick_z_table + args.coaster_thickness

    rng = np.random.default_rng(args.seed)
    if args.mug and args.coaster:
        mug = tuple(float(v) for v in args.mug.split(","))
        coaster = tuple(float(v) for v in args.coaster.split(","))
        print("[using user-supplied positions]")
    else:
        mug, coaster = sample_pair(rng)

    hx, hy, hz_home = (float(v) for v in args.home.split(","))

    print(f"sampled initial positions:")
    print(f"  mug     @ ({mug[0]:+.3f},{mug[1]:+.3f})")
    print(f"  coaster @ ({coaster[0]:+.3f},{coaster[1]:+.3f})")
    print(f"  dist(mug, coaster) = {dist(mug, coaster):.3f}m  (min {MIN_MUG_COASTER})")

    print(f"\nConnecting...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    grip = robotiq_gripper.RobotiqGripper()
    grip.connect(args.robot_ip, 63352)
    grip.activate()
    grip.move_and_wait_for_pos(args.transit_grip_pos, 255, 255)

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

    try:
        print(f"\n=== PHASE 1: HOVER ({args.hover_pause}s each) -- place objects ===")
        for label, p in [("mug", mug), ("coaster", coaster)]:
            print(f"  hover at {label}: ({p[0]:+.3f},{p[1]:+.3f})")
            go(p[0], p[1], args.hover_z)
            time.sleep(args.hover_pause)

        print(f"\n=== PHASE 2: TRIAL ===")
        print("-- TASK: put_mug_on_coaster --")
        pick("mug", *mug)
        place("mug_on_coaster", *coaster, surface_z=args.coaster_thickness)

        mug_new, coaster_new = sample_pair(rng, prev_mug=mug, prev_coaster=coaster)
        print(f"-- RESET --")
        print(f"  mug_new={mug_new}  coaster_new={coaster_new}")
        pick("mug_from_coaster", *coaster, pick_z=args.pick_z_mug_on_coaster)
        place("mug_new", *mug_new, surface_z=0.0)
        pick("coaster", *coaster, pick_z=args.pick_z_coaster)
        place("coaster_new", *coaster_new, surface_z=0.0)

        print(f"\nReturning home -> ({hx},{hy},{hz_home})")
        go(hx, hy, hz_home)
        end_state = {"mug": list(mug_new), "coaster": list(coaster_new)}
        json.dump(end_state, open("/tmp/mug_coaster_trial_end.json", "w"))
        print(f"\nEND-STATE (objects are physically here now):")
        print(f"  mug     @ ({mug_new[0]:+.4f}, {mug_new[1]:+.4f})")
        print(f"  coaster @ ({coaster_new[0]:+.4f}, {coaster_new[1]:+.4f})")
        print(f"saved to /tmp/mug_coaster_trial_end.json")
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
