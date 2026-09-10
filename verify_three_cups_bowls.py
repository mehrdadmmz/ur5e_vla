"""Verify place_three_cups_in_bowls setup and record accepted heights.

Two phases:
  1. Corner probe -- drive TCP (open gripper, hover z, low speed) to each of
     the 4 workspace corners. Any corner that fails motion planning is
     reported; you can shrink the polygon before collection.
  2. Trial cycle -- for each color (blue, pink, yellow) pick the cup at its
     initial xy and release into the same-color bowl. This validates
     pick_z_cup and release_z_cup_in_bowl at three different (x, y).

Writes accepted values into the plan YAML under `verified:` and mirrors the
accepted heights into the `verify:` block used during collection.

Usage:
    .venv/bin/python -u verify_three_cups_bowls.py \\
        --plan collect/plan_place_three_cups_in_bowls.yaml
"""
import argparse
import os
import sys
import time
from datetime import datetime

import rtde_control
import rtde_receive
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "collect"))
import robotiq_gripper

RX, RY, RZ = 3.14159, 0.0, 0.0
DEFAULT_PLAN = os.path.join(THIS_DIR, "collect", "plan_place_three_cups_in_bowls.yaml")
COLORS = ["blue", "pink", "yellow"]


def load_plan(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_plan(path, plan):
    # Atomic write via rename.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(plan, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


def parse_xy(s):
    return tuple(float(v) for v in s.split(","))


def prompt(msg):
    try:
        return input(msg).strip().lower()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--skip-corners", action="store_true",
                    help="Skip the workspace corner probe (not recommended).")
    ap.add_argument("--skip-trial", action="store_true",
                    help="Skip the per-color pick-place trial.")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    v = plan["verify"]

    # Heights (from plan; can be updated interactively at end of trial).
    hover_z            = float(v["hover_z"])
    pick_z_cup         = float(v["pick_z_cup"])
    release_z_in_bowl  = float(v["release_z_cup_in_bowl"])
    lift_z             = float(v["lift_z"])
    speed              = float(v.get("speed", 0.20))
    accel              = float(v.get("accel", 0.40))
    descend_speed      = float(v.get("descend_speed", 0.05))
    descend_accel      = float(v.get("descend_accel", 0.40))
    grip_pos           = int(v.get("grip_pos", 200))
    transit_grip_pos   = int(v.get("transit_grip_pos", 0))
    home_parts         = tuple(float(x) for x in v["home"].split(","))
    hx, hy             = home_parts[0], home_parts[1]
    hz_home            = home_parts[2] if len(home_parts) >= 3 else lift_z

    cups  = {c: parse_xy(v[f"cup_{c}"])  for c in COLORS}
    bowls = {c: parse_xy(v[f"bowl_{c}"]) for c in COLORS}

    corners = [tuple(c) for c in plan["workspace"]["corners"]]

    print(f"\nplan: {args.plan}")
    print(f"cups  {cups}")
    print(f"bowls {bowls}")
    print(f"corners {corners}")
    print(f"heights: hover={hover_z}  pick_cup={pick_z_cup}  "
          f"release_in_bowl={release_z_in_bowl}  lift={lift_z}")

    print(f"\nConnecting to UR5e at {args.robot_ip} ...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    grip = robotiq_gripper.RobotiqGripper()
    grip.connect(args.robot_ip, 63352)
    grip.activate()
    grip.move_and_wait_for_pos(transit_grip_pos, 255, 255)

    def go(x, y, z, slow=False):
        s = speed * 0.5 if slow else speed
        a = accel * 0.5 if slow else accel
        rtde_c.moveL([x, y, z, RX, RY, RZ], s, a)

    def pick_cup(x, y):
        go(x, y, hover_z)
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        go(x, y, pick_z_cup)
        grip.move_and_wait_for_pos(grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, lift_z)

    def release_in_bowl(x, y):
        go(x, y, hover_z)
        go(x, y, release_z_in_bowl)
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, lift_z)

    corner_results = []
    trial_results = []

    try:
        # ---------- PHASE 1: corner probe ----------
        if not args.skip_corners:
            print(f"\n=== PHASE 1: WORKSPACE CORNER PROBE (hover_z={hover_z}) ===")
            print("The arm will drive to each corner at HOVER height, slowly, gripper open.")
            print("Kill (Ctrl-C) if it looks like a collision is imminent.")
            resp = prompt("Ready to probe corners? [y/N] ")
            if resp != "y":
                print("skipping corner probe.")
            else:
                for i, (x, y) in enumerate(corners, 1):
                    label = f"corner {i}/{len(corners)} ({x:+.3f},{y:+.3f})"
                    print(f"  -> {label}")
                    try:
                        go(x, y, hover_z, slow=True)
                        time.sleep(0.5)
                        ok = prompt(f"    {label} reachable? [y/N] ") == "y"
                    except Exception as e:
                        print(f"    FAILED motion: {e}")
                        ok = False
                    corner_results.append({
                        "corner_index": i - 1,
                        "xy": [x, y],
                        "reachable": ok,
                    })
                go(hx, hy, hz_home)

        # ---------- PHASE 2: per-color trial ----------
        if not args.skip_trial:
            print(f"\n=== PHASE 2: PER-COLOR PICK + PLACE TRIAL ===")
            print("Physically place cups and bowls at their initial positions:")
            for c in COLORS:
                cx, cy = cups[c]; bx, by = bowls[c]
                print(f"  cup_{c}  @ ({cx:+.3f},{cy:+.3f})   "
                      f"bowl_{c} @ ({bx:+.3f},{by:+.3f})")
            resp = prompt("Ready to run trial? [y/N] ")
            if resp != "y":
                print("skipping trial.")
            else:
                for c in COLORS:
                    cx, cy = cups[c]; bx, by = bowls[c]
                    print(f"\n  --- {c.upper()}: cup ({cx:+.3f},{cy:+.3f}) "
                          f"-> bowl ({bx:+.3f},{by:+.3f}) ---")
                    pick_ok = release_ok = False
                    try:
                        pick_cup(cx, cy)
                        pick_ok = prompt(f"    cup_{c} grasped cleanly? [y/N] ") == "y"
                        release_in_bowl(bx, by)
                        release_ok = prompt(f"    cup_{c} released into bowl_{c} cleanly? [y/N] ") == "y"
                    except Exception as e:
                        print(f"    trial FAILED: {e}")
                    trial_results.append({
                        "color": c,
                        "cup_xy": [cx, cy],
                        "bowl_xy": [bx, by],
                        "pick_ok": pick_ok,
                        "release_ok": release_ok,
                    })
                    go(hx, hy, hz_home)

        # ---------- Ask if heights need adjustment ----------
        print(f"\n=== FINALIZE HEIGHTS ===")
        print(f"current: hover_z={hover_z}  pick_z_cup={pick_z_cup}  "
              f"release_z_cup_in_bowl={release_z_in_bowl}")
        for name in ("hover_z", "pick_z_cup", "release_z_cup_in_bowl"):
            new = prompt(f"  new value for {name} (blank to keep): ")
            if new:
                try:
                    val = float(new)
                    if name == "hover_z":
                        hover_z = val
                    elif name == "pick_z_cup":
                        pick_z_cup = val
                    elif name == "release_z_cup_in_bowl":
                        release_z_in_bowl = val
                except ValueError:
                    print(f"    invalid, keeping current")

        # Also let user report the actually-verified workspace polygon (in case
        # some corners were unreachable and they want to shrink).
        verified_polygon = corners
        resp = prompt("Restrict workspace to a subset of corners? [y/N] ")
        if resp == "y":
            print(f"  current corners: {corners}")
            new_corners = prompt("  enter as: x1,y1;x2,y2;x3,y3;x4,y4  (m, base frame): ")
            if new_corners:
                try:
                    verified_polygon = [
                        [float(a) for a in pair.split(",")]
                        for pair in new_corners.split(";")
                    ]
                except Exception as e:
                    print(f"    invalid, keeping raw corners: {e}")

        # Write verified: block back to plan
        plan["verified"] = {
            "verified_at": datetime.utcnow().isoformat() + "Z",
            "hover_z": hover_z,
            "pick_z_cup": pick_z_cup,
            "release_z_cup_in_bowl": release_z_in_bowl,
            "lift_z": lift_z,
            "workspace_polygon": verified_polygon,
            "corner_probe": corner_results,
            "trial_cycle":  trial_results,
        }
        # Also mirror latest heights into verify: so the same yaml stays consistent
        plan["verify"]["hover_z"] = hover_z
        plan["verify"]["pick_z_cup"] = pick_z_cup
        plan["verify"]["release_z_cup_in_bowl"] = release_z_in_bowl
        save_plan(args.plan, plan)
        print(f"\nWrote verified: block to {args.plan}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
