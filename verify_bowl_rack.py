"""
Verify put_bowl_on_rack task setup before collecting.

  Phase 1 -- hover at the bowl xy so the user places the bowl under the
             gripper. Also hovers over the dish rack once so the user can
             confirm the fixed rack position matches physical reality.
  Phase 2 -- run ONE recorded-style motion (pick bowl, place on rack) +
             the reset motion (pick bowl off rack, place at bowl_new
             sampled in a small disk around the bowl center), so the
             verify ends in a fresh state ready for collection.

Task-specific properties (vs cup+bowl, apple+bowl, mug+coaster, two cubes):
  - Only ONE object moves each episode (the bowl). The dish rack sits at
    a fixed xy and is never picked up during reset.
  - Bowl placement on rack always lands at the same xyz (from arm jog),
    so the on-rack pose is consistent for eval reproducibility.
  - Between-episode bowl positions are constrained to a small disk
    (default 4 cm radius) around a fixed bowl_center, so the arm never
    risks colliding with the still-standing rack.

Shares config with the collect script via `--plan <plan_yaml>`. Same block
is read by `collect_bowl_rack.py` so verify + collect always agree.

End-state (final bowl xy) is printed and saved to
/tmp/bowl_rack_trial_end.json.
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


def load_verify_defaults(plan_path):
    """Read the `verify:` section of a plan yaml. Returns {} if missing."""
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    return plan.get("verify", {}) or {}


def sample_bowl_in_disk(rng, center, radius):
    """Sample a bowl xy uniformly in a disk of `radius` around `center`.
    Uses r = R*sqrt(u) so the density is uniform over the disk (not just
    over (r, theta))."""
    r = radius * float(np.sqrt(rng.uniform()))
    theta = float(rng.uniform(0.0, 2 * np.pi))
    return (center[0] + r * np.cos(theta), center[1] + r * np.sin(theta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--pick-z-table", type=float, default=0.122)
    ap.add_argument("--release-z-on-rack", type=float, default=0.149)
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bowl", type=str, default=None, help="'x,y' override for bowl initial position")
    ap.add_argument("--dish-rack", type=str, default=None,
                    help="'x,y' override for the FIXED dish rack position")
    ap.add_argument("--bowl-center", type=str, default=None,
                    help="'x,y' override for the center of the bowl-jitter disk")
    ap.add_argument("--bowl-jitter-radius", type=float, default=0.04)
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

    if args.pick_z_bowl_on_rack is None:
        args.pick_z_bowl_on_rack = args.release_z_on_rack

    def parse_xy(s):
        return tuple(float(v) for v in s.split(","))

    rng = np.random.default_rng(args.seed)
    bowl = parse_xy(args.bowl) if args.bowl else parse_xy("-0.003,0.569")
    dish_rack = parse_xy(args.dish_rack) if args.dish_rack else parse_xy("0.172,0.819")
    bowl_center = parse_xy(args.bowl_center) if args.bowl_center else bowl

    hx, hy, hz_home = (float(v) for v in args.home.split(","))

    print(f"initial positions:")
    print(f"  bowl       @ ({bowl[0]:+.3f},{bowl[1]:+.3f})")
    print(f"  dish rack  @ ({dish_rack[0]:+.3f},{dish_rack[1]:+.3f})   [FIXED]")
    print(f"  bowl_center @ ({bowl_center[0]:+.3f},{bowl_center[1]:+.3f})  jitter_radius={args.bowl_jitter_radius}")

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
        """moveUntilContact release on the table (no known height)."""
        hz = args.hover_z
        print(f"  PLACE {label} ({x:+.3f},{y:+.3f}) hover={hz:.3f} -> moveUntilContact")
        go(x, y, hz)
        descend_contact()
        z_now = rtde_r.getActualTCPPose()[2]
        print(f"    contact at z={z_now:.4f}")
        grip.move_and_wait_for_pos(args.transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    try:
        print(f"\n=== PHASE 1: HOVER ({args.hover_pause}s each) ===")
        print(f"  hover at bowl: place the bowl under the gripper")
        go(bowl[0], bowl[1], args.hover_z)
        time.sleep(args.hover_pause)
        print(f"  hover at dish rack: confirm the rack is at this xy")
        go(dish_rack[0], dish_rack[1], args.hover_z)
        time.sleep(args.hover_pause)

        print(f"\n=== PHASE 2: TRIAL ===")
        print("-- TASK: put_bowl_on_rack --")
        pick("bowl", *bowl)
        place_on_rack("bowl_on_rack", *dish_rack, release_z=args.release_z_on_rack)

        bowl_new = sample_bowl_in_disk(rng, bowl_center, args.bowl_jitter_radius)
        print(f"-- RESET --")
        print(f"  bowl_new={bowl_new}  (within {args.bowl_jitter_radius}m of center {bowl_center})")
        pick("bowl_from_rack", *dish_rack, pick_z=args.pick_z_bowl_on_rack)
        place_on_table("bowl_new", *bowl_new)

        print(f"\nReturning home -> ({hx},{hy},{hz_home})")
        go(hx, hy, hz_home)
        end_state = {"bowl": list(bowl_new), "dish_rack": list(dish_rack)}
        json.dump(end_state, open("/tmp/bowl_rack_trial_end.json", "w"))
        print(f"\nEND-STATE (objects are physically here now):")
        print(f"  bowl      @ ({bowl_new[0]:+.4f}, {bowl_new[1]:+.4f})")
        print(f"  dish rack @ ({dish_rack[0]:+.4f}, {dish_rack[1]:+.4f})  [FIXED]")
        print(f"saved to /tmp/bowl_rack_trial_end.json")
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
