"""
Verify stack-two-cubes task setup before collecting.

  Phase 1 -- hover at sampled cube_a + cube_b positions so the user places
             the physical cubes under the gripper.
  Phase 2 -- run ONE recorded-style motion (pick cube_a, stack on cube_b) +
             the reset motion (pick cube_a back off cube_b, move cube_b),
             so the verify ends in a fresh separated state ready for
             collection.

Both cubes are Rubik's cubes (~5.7 cm edge). Grip is straightforward -- the
gripper wraps around the cube's outside faces. Placement lands cube_a on
top of cube_b via moveUntilContact.

Shares config with the collect script via `--plan <plan_yaml>`. Same block
is read by `collect_two_cubes.py` so verify + collect always agree.

The end-state (final cube_a + cube_b xy) is printed and saved to
/tmp/two_cubes_trial_end.json -- paste those coords into
`collect_two_cubes.py --cube-a=... --cube-b=...` to seed collection.
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
CUBE_RADIUS = 0.03            # half-edge for spatial separation
MIN_CUBE_CUBE = 0.12          # separation at spawn (gripper clearance around each cube)
MIN_RESET_SHIFT = 0.10        # new pose must be at least this far from its prior pose
MIN_CUBE_A_TO_PREV_CUBE_B = 0.20  # new cube_a must clear the still-standing cube_b


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
                prev_cube_a=None, prev_cube_b=None,
                min_shift=MIN_RESET_SHIFT,
                min_cube_a_to_prev_cube_b=MIN_CUBE_A_TO_PREV_CUBE_B):
    """Sample a (cube_a, cube_b) pair in the trapezoid.

    If prev_* are given, new poses must each be >= min_shift from their
    previous position. Additionally, the new cube_a must be >=
    min_cube_a_to_prev_cube_b from the *previous* cube_b, since during reset
    cube_a is placed before cube_b is moved -- we need clearance so cube_a
    doesn't collide with the still-standing cube_b.
    """
    for _ in range(max_outer):
        cube_a = sample_in_trapezoid(rng)
        if prev_cube_a is not None and dist(cube_a, prev_cube_a) < min_shift:
            continue
        if prev_cube_b is not None and dist(cube_a, prev_cube_b) < min_cube_a_to_prev_cube_b:
            continue
        for _ in range(max_inner):
            cube_b = sample_in_trapezoid(rng)
            if prev_cube_b is not None and dist(cube_b, prev_cube_b) < min_shift:
                continue
            if dist(cube_a, cube_b) >= MIN_CUBE_CUBE:
                return cube_a, cube_b
    raise RuntimeError("could not sample cube_a/cube_b pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--cube-height", type=float, default=0.057,
                    help="Rubik's cube edge length (standard 3x3 is ~5.7 cm)")
    ap.add_argument("--pick-z-table", type=float, default=0.150)
    ap.add_argument("--pick-z-cube-on-cube", type=float, default=None,
                    help="TCP z when regrasping cube_a from top of cube_b "
                         "(defaults to pick_z_table + cube_height)")
    ap.add_argument("--hover-z", type=float, default=0.25)
    ap.add_argument("--lift-z", type=float, default=0.30)
    ap.add_argument("--home", default="-0.10,0.55,0.25")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.40)
    ap.add_argument("--descend-speed", type=float, default=0.05)
    ap.add_argument("--descend-accel", type=float, default=0.40)
    ap.add_argument("--grip-pos", type=int, default=200,
                    help="Robotiq target when closing (0=open, 255=fully closed).")
    ap.add_argument("--transit-grip-pos", type=int, default=0)
    ap.add_argument("--hover-pause", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cube-a", type=str, default=None, help="'x,y' override for cube_a")
    ap.add_argument("--cube-b", type=str, default=None, help="'x,y' override for cube_b")
    ap.add_argument("--plan", type=str, default=None,
                    help="Path to a collect plan yaml. Reads defaults from its "
                         "`verify:` section. Any CLI arg still overrides.")
    args = ap.parse_args()

    if args.plan:
        defaults = load_verify_defaults(args.plan)
        user_supplied = {a.split("=")[0].lstrip("-").replace("-", "_")
                         for a in sys.argv[1:] if a.startswith("--")}
        for key, val in defaults.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr) and attr not in user_supplied:
                setattr(args, attr, val)

    if args.pick_z_cube_on_cube is None:
        args.pick_z_cube_on_cube = args.pick_z_table + args.cube_height

    rng = np.random.default_rng(args.seed)
    if args.cube_a and args.cube_b:
        cube_a = tuple(float(v) for v in args.cube_a.split(","))
        cube_b = tuple(float(v) for v in args.cube_b.split(","))
        print("[using user-supplied positions]")
    else:
        cube_a, cube_b = sample_pair(rng)

    hx, hy, hz_home = (float(v) for v in args.home.split(","))

    print(f"sampled initial positions:")
    print(f"  cube_a @ ({cube_a[0]:+.3f},{cube_a[1]:+.3f})")
    print(f"  cube_b @ ({cube_b[0]:+.3f},{cube_b[1]:+.3f})")
    print(f"  dist(cube_a, cube_b) = {dist(cube_a, cube_b):.3f}m  (min {MIN_CUBE_CUBE})")

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
        for label, p in [("cube_a", cube_a), ("cube_b", cube_b)]:
            print(f"  hover at {label}: ({p[0]:+.3f},{p[1]:+.3f})")
            go(p[0], p[1], args.hover_z)
            time.sleep(args.hover_pause)

        print(f"\n=== PHASE 2: TRIAL ===")
        print("-- TASK: stack_two_cubes --")
        pick("cube_a", *cube_a)
        # cube_b top surface = table + cube_height
        place("cube_a_on_cube_b", *cube_b, surface_z=args.cube_height)

        cube_a_new, cube_b_new = sample_pair(rng, prev_cube_a=cube_a, prev_cube_b=cube_b)
        print(f"-- RESET --")
        print(f"  cube_a_new={cube_a_new}  cube_b_new={cube_b_new}")
        pick("cube_a_from_cube_b", *cube_b, pick_z=args.pick_z_cube_on_cube)
        place("cube_a_new", *cube_a_new, surface_z=0.0)
        pick("cube_b", *cube_b)
        place("cube_b_new", *cube_b_new, surface_z=0.0)

        print(f"\nReturning home -> ({hx},{hy},{hz_home})")
        go(hx, hy, hz_home)
        end_state = {"cube_a": list(cube_a_new), "cube_b": list(cube_b_new)}
        json.dump(end_state, open("/tmp/two_cubes_trial_end.json", "w"))
        print(f"\nEND-STATE (objects are physically here now):")
        print(f"  cube_a @ ({cube_a_new[0]:+.4f}, {cube_a_new[1]:+.4f})")
        print(f"  cube_b @ ({cube_b_new[0]:+.4f}, {cube_b_new[1]:+.4f})")
        print(f"saved to /tmp/two_cubes_trial_end.json")
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
