"""
Verify put_book_in_box task setup before collecting.

  Phase 1 -- hover at the book xy so the user places the book (standing)
             under the gripper. Also hovers over the file box once so the
             user can confirm the fixed box position matches physical reality.
  Phase 2 -- run ONE recorded-style motion (pick book, place in box) +
             the reset motion (pick book off box, place at a random shelf
             slot), so the verify ends in a fresh state ready for collection.

Task-specific properties:
  - Bookshelf has 5 DISCRETE x slots and a continuous y range.
  - File box sits at a FIXED xy on the table and is never picked up.
  - Book always lands at the same xyz inside the file box (fixed-z release).
  - Reset samples a fresh (x_slot, y) for the book new position.

Shares config with the collect script via `--plan <plan_yaml>`. Same block
is read by `collect_book_box.py` so verify + collect always agree.

End-state (final book xy) is printed and saved to
/tmp/book_box_trial_end.json.
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


def sample_shelf_slot(rng, x_choices, y_min, y_max):
    """Sample a bookshelf slot: uniform pick from x_choices, uniform y in [y_min, y_max]."""
    x = float(rng.choice(x_choices))
    y = float(rng.uniform(y_min, y_max))
    return (x, y)


def parse_xy(s):
    return tuple(float(v) for v in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--pick-z-book", type=float, default=0.309)
    ap.add_argument("--release-z-on-shelf", type=float, default=None,
                    help="TCP z when placing book back on shelf during reset "
                         "(defaults to pick_z_book)")
    ap.add_argument("--release-z-in-box", type=float, default=0.304)
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--book", type=str, default=None, help="'x,y' override for book initial position")
    ap.add_argument("--file-box", type=str, default=None, help="'x,y' override for the FIXED file box")
    ap.add_argument("--shelf-x-choices", type=str, default=None,
                    help="comma-separated list of the 5 slot x's (m). "
                         "Default: from plan.")
    ap.add_argument("--shelf-y-min", type=float, default=0.725)
    ap.add_argument("--shelf-y-max", type=float, default=0.800)
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

    if args.pick_z_book_in_box is None:
        args.pick_z_book_in_box = args.release_z_in_box
    if args.release_z_on_shelf is None:
        args.release_z_on_shelf = args.pick_z_book

    # Normalise shelf_x_choices: could be list (from yaml) or str (from CLI).
    if isinstance(args.shelf_x_choices, str):
        args.shelf_x_choices = [float(v) for v in args.shelf_x_choices.split(",")]
    elif args.shelf_x_choices is None:
        args.shelf_x_choices = [0.080, 0.014, -0.053, -0.124, -0.189]

    rng = np.random.default_rng(args.seed)
    book = parse_xy(args.book) if args.book else parse_xy("0.080,0.725")
    file_box = parse_xy(args.file_box) if args.file_box else parse_xy("0.211,0.700")

    hx, hy, hz_home = (float(v) for v in args.home.split(","))

    print(f"initial positions:")
    print(f"  book       @ ({book[0]:+.3f},{book[1]:+.3f})")
    print(f"  file box   @ ({file_box[0]:+.3f},{file_box[1]:+.3f})   [FIXED]")
    print(f"  shelf slots x = {args.shelf_x_choices}")
    print(f"  shelf y range  = [{args.shelf_y_min}, {args.shelf_y_max}]")

    print(f"\nConnecting...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    grip = robotiq_gripper.RobotiqGripper()
    grip.connect(args.robot_ip, 63352)
    grip.activate()
    grip.move_and_wait_for_pos(args.transit_grip_pos, 255, 255)

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
        """Fixed-z release (deterministic, matches jog). No moveUntilContact --
        the shelf slots and box walls would trigger contact incorrectly."""
        hz = max(args.hover_z, release_z + 0.10)
        print(f"  PLACE {label} ({x:+.3f},{y:+.3f}) hover={hz:.3f} release_z={release_z:.3f} (fixed)")
        go(x, y, hz)
        go(x, y, release_z)
        grip.move_and_wait_for_pos(args.transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x, y, args.lift_z)

    try:
        print(f"\n=== PHASE 1: HOVER ({args.hover_pause}s each) ===")
        print(f"  hover at book: place the book (standing) under the gripper")
        go(book[0], book[1], args.hover_z)
        time.sleep(args.hover_pause)
        print(f"  hover at file box: confirm the box is at this xy")
        go(file_box[0], file_box[1], args.hover_z)
        time.sleep(args.hover_pause)

        print(f"\n=== PHASE 2: TRIAL ===")
        print("-- TASK: put_book_in_box --")
        pick("book", *book)
        place_fixed_z("book_in_box", *file_box, release_z=args.release_z_in_box)

        book_new = sample_shelf_slot(rng, args.shelf_x_choices, args.shelf_y_min, args.shelf_y_max)
        print(f"-- RESET --")
        print(f"  book_new={book_new}  (slot x from {args.shelf_x_choices}, y uniform in [{args.shelf_y_min}, {args.shelf_y_max}])")
        pick("book_from_box", *file_box, pick_z=args.pick_z_book_in_box)
        place_fixed_z("book_new", *book_new, release_z=args.release_z_on_shelf)

        print(f"\nReturning home -> ({hx},{hy},{hz_home})")
        go(hx, hy, hz_home)
        end_state = {"book": list(book_new), "file_box": list(file_box)}
        json.dump(end_state, open("/tmp/book_box_trial_end.json", "w"))
        print(f"\nEND-STATE (objects are physically here now):")
        print(f"  book     @ ({book_new[0]:+.4f}, {book_new[1]:+.4f})")
        print(f"  file box @ ({file_box[0]:+.4f}, {file_box[1]:+.4f})  [FIXED]")
        print(f"saved to /tmp/book_box_trial_end.json")
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
