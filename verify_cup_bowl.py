"""
Verify cup+bowl-task setup before collecting.

  Phase 1 -- hover at sampled cup + bowl positions so the user places
             the physical objects under the gripper.
  Phase 2 -- run ONE recorded-style motion (pick cup, place in bowl) +
             the reset motion (pick cup back out, move bowl), so the
             verify ends in a fresh separated state ready for collection.

Shares config with the collect script via `--plan <plan_yaml>`: defaults
for heights, pick z's, initial xy, hover pause, etc. are loaded from the
plan's `verify:` block. Any CLI flag still overrides. Same block is read
by `collect_cup_bowl.py` so verify + collect always agree on numbers.

The end-state (final cup + bowl xy) is printed and saved to
/tmp/cup_bowl_trial_end.json -- paste those coords into
`collect_cup_bowl.py --cup=... --bowl=...` to seed collection.
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
CUP_RADIUS = 0.055
BOWL_RADIUS = 0.08
MIN_CUP_BOWL = 0.16
MIN_RESET_SHIFT = 0.10       # new pose must be at least this far from its prior pose
MIN_CUP_TO_PREV_BOWL = 0.25  # new cup must be this far from the *still-standing* bowl


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
                prev_cup=None, prev_bowl=None,
                min_shift=MIN_RESET_SHIFT,
                min_cup_to_prev_bowl=MIN_CUP_TO_PREV_BOWL):
    """Sample a (cup, bowl) pair in the trapezoid.

    If prev_cup / prev_bowl are given, the new poses must each be at least
    `min_shift` metres from their previous position. Additionally, the new
    cup must be at least `min_cup_to_prev_bowl` metres from the *previous*
    bowl position, since during reset the cup is placed before the bowl is
    moved -- we need clearance so the cup doesn't collide with the still-
    standing bowl.
    """
    for _ in range(max_outer):
        cup = sample_in_trapezoid(rng)
        if prev_cup is not None and dist(cup, prev_cup) < min_shift:
            continue
        if prev_bowl is not None and dist(cup, prev_bowl) < min_cup_to_prev_bowl:
            continue
        for _ in range(max_inner):
            bowl = sample_in_trapezoid(rng)
            if prev_bowl is not None and dist(bowl, prev_bowl) < min_shift:
                continue
            if dist(cup, bowl) >= MIN_CUP_BOWL:
                return cup, bowl
    raise RuntimeError("could not sample cup/bowl pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--cup-height", type=float, default=0.11)
    ap.add_argument("--bowl-height", type=float, default=0.06)
    ap.add_argument("--pick-z-table", type=float, default=0.130)
    ap.add_argument("--pick-z-bowl", type=float, default=None,
                    help="pick z for grabbing the bowl during reset "
                         "(defaults to --pick-z-table)")
    ap.add_argument("--pick-z-cup-in-bowl", type=float, default=None)
    ap.add_argument("--hover-z", type=float, default=0.20)
    ap.add_argument("--lift-z", type=float, default=0.30)
    ap.add_argument("--home", default="-0.10,0.55,0.25")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--descend-speed", type=float, default=0.05)
    ap.add_argument("--descend-accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200)
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0,
                    help="seconds at each phase-1 hover for manual placement")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cup", type=str, default=None, help="'x,y' override")
    ap.add_argument("--bowl", type=str, default=None, help="'x,y' override")
    ap.add_argument("--plan", type=str, default=None,
                    help="Path to a collect plan yaml. Reads defaults from its "
                         "`verify:` section so the same file drives verify and "
                         "main.py. Any CLI arg still overrides the file.")
    args = ap.parse_args()

    # If a plan is provided, its `verify:` section supplies defaults for any
    # arg the user did NOT pass on the command line. CLI values always win.
    if args.plan:
        defaults = load_verify_defaults(args.plan)
        user_supplied = {a.split("=")[0].lstrip("-").replace("-", "_")
                         for a in sys.argv[1:] if a.startswith("--")}
        for key, val in defaults.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr) and attr not in user_supplied:
                setattr(args, attr, val)

    if args.pick_z_cup_in_bowl is None:
        args.pick_z_cup_in_bowl = args.pick_z_table + args.bowl_height
    if args.pick_z_bowl is None:
        args.pick_z_bowl = args.pick_z_table

    rng = np.random.default_rng(args.seed)
    if args.cup and args.bowl:
        cup = tuple(float(v) for v in args.cup.split(","))
        bowl = tuple(float(v) for v in args.bowl.split(","))
        print("[using user-supplied positions]")
    else:
        cup, bowl = sample_pair(rng)

    hx, hy, hz_home = (float(v) for v in args.home.split(","))

    print(f"sampled initial positions:")
    print(f"  cup  @ ({cup[0]:+.3f},{cup[1]:+.3f})")
    print(f"  bowl @ ({bowl[0]:+.3f},{bowl[1]:+.3f})")
    print(f"  dist(cup, bowl) = {dist(cup, bowl):.3f}m  (min {MIN_CUP_BOWL})")

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
        # --- PHASE 1: hover ---
        print(f"\n=== PHASE 1: HOVER ({args.hover_pause}s each) -- place objects ===")
        for label, p in [("cup", cup), ("bowl", bowl)]:
            print(f"  hover at {label}: ({p[0]:+.3f},{p[1]:+.3f})")
            go(p[0], p[1], args.hover_z)
            time.sleep(args.hover_pause)

        # --- PHASE 2: trial cycle (RECORDED-like motion + RESET) ---
        print(f"\n=== PHASE 2: TRIAL ===")
        print("-- TASK: put_cup_in_bowl --")
        pick("cup", *cup)
        place("cup_in_bowl", *bowl, surface_z=args.bowl_height)

        # --- RESET ---
        cup_new, bowl_new = sample_pair(rng, prev_cup=cup, prev_bowl=bowl)
        print(f"-- RESET --")
        print(f"  cup_new={cup_new}  bowl_new={bowl_new}")
        pick("cup_from_bowl", *bowl, pick_z=args.pick_z_cup_in_bowl)
        place("cup_new", *cup_new, surface_z=0.0)
        pick("bowl", *bowl, pick_z=args.pick_z_bowl)
        place("bowl_new", *bowl_new, surface_z=0.0)

        print(f"\nReturning home -> ({hx},{hy},{hz_home})")
        go(hx, hy, hz_home)
        end_state = {"cup": list(cup_new), "bowl": list(bowl_new)}
        json.dump(end_state, open("/tmp/cup_bowl_trial_end.json", "w"))
        print(f"\nEND-STATE (objects are physically here now):")
        print(f"  cup  @ ({cup_new[0]:+.4f}, {cup_new[1]:+.4f})")
        print(f"  bowl @ ({bowl_new[0]:+.4f}, {bowl_new[1]:+.4f})")
        print(f"saved to /tmp/cup_bowl_trial_end.json")
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
