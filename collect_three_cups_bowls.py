"""Collect task: place_three_cups_in_bowls.

Per episode, three cups (blue/pink/yellow) are placed sequentially into three
bowls. Two assignment policies (deterministically interleaved per subset seed):

  forced_same_color  -- each cup goes to its same-color bowl
  unrestricted       -- random bijection (each bowl gets exactly one cup;
                        same-color hits by chance are allowed and recorded
                        via color_match=True)

RECORDED per episode: 3 pick-place chunks (cup -> bowl), with recorder's
context dict driving 4 extra state.csv columns (chunk_index, phase,
active_target_id, active_destination_id).

RESET (not recorded): robot picks each cup back out of its bowl and moves all
six objects to fresh randomly-sampled xy for the next episode. The six moves
are randomly ordered with one safety dependency: a bowl becomes eligible only
after the cup inside it has been removed. Then a --reset-pause window lets you
swap out clutter by hand.

Initial placement (first ep or --reset-state): place all six objects at the
configured initial coordinates before starting. All later episodes use the
robot-driven reset.

Usage:
  .venv/bin/python -u collect_three_cups_bowls.py --subset clean -n 100 --reset-pause 15
  .venv/bin/python -u collect_three_cups_bowls.py --subset d1 -n 25 --reset-pause 15
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

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
TASK_NAME = "place_three_cups_in_bowls"
DEFAULT_PLAN = os.path.join(THIS_DIR, "collect", "plan_place_three_cups_in_bowls.yaml")
COLORS = ["blue", "pink", "yellow"]
SCHEMA_VERSION = 1
CALIBRATION_HOLD_FRAMES = 15   # ~0.75s at 20 Hz


# ------------------------------------------------------------------ plan I/O

def load_plan(path):
    with open(path) as f:
        return yaml.safe_load(f)


def catalog_by_id(plan):
    out = {}
    for group in ("task_objects", "clutter_pool"):
        for obj in plan["object_catalog"].get(group, []):
            out[obj["id"]] = obj
    return out


def parse_xy(s):
    return tuple(float(v) for v in s.split(","))


# ------------------------------------------------------------------ geometry

def point_in_polygon(pt, polygon):
    """Ray-casting point-in-simple-polygon."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def polygon_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def point_segment_dist(pt, a, b):
    """Shortest 2-D distance from pt to line segment a-b."""
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return dist(pt, a)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return dist(pt, (ax + t * dx, ay + t * dy))


def fits_in_polygon(pt, polygon, radius):
    """True when the whole circular footprint fits inside the polygon."""
    if not point_in_polygon(pt, polygon):
        return False
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if point_segment_dist(pt, a, b) < radius:
            return False
    return True


def position_is_clear(pt, radius, polygon, occupied):
    """Check a target against the table boundary and current table state.

    occupied maps object id -> ((x, y), footprint_radius_m, required_gap_m).
    The stored gap lets final destinations keep the full layout margin while
    old positions that exist only during reset use a smaller transit margin.
    """
    if not fits_in_polygon(pt, polygon, radius):
        return False
    return all(
        dist(pt, other_xy) >= radius + other_radius + required_gap
        for other_xy, other_radius, required_gap in occupied.values()
    )


def sample_clear_position(rng, radius, polygon, occupied, max_attempts,
                          previous=None, min_shift=0.03):
    """Sample one destination that is clear in the current table state."""
    xmin, xmax, ymin, ymax = polygon_bbox(polygon)
    xmin += radius
    xmax -= radius
    ymin += radius
    ymax -= radius
    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError(f"polygon too small for footprint radius {radius}")

    for _ in range(max_attempts):
        pt = (float(rng.uniform(xmin, xmax)),
              float(rng.uniform(ymin, ymax)))
        if previous is not None and dist(pt, previous) < min_shift:
            continue
        if position_is_clear(pt, radius, polygon, occupied):
            return pt
    raise RuntimeError("could not sample a collision-free reset destination")


def sample_layout(rng, ordered_objects, polygon, min_gap, max_attempts, max_restarts,
                  prev_positions=None, min_shift=0.03):
    """Rejection-sample xy for each object in placement order.

    ordered_objects: [(id, footprint_radius_m), ...]
    prev_positions: optional {id: (x,y)} — new pose for that id must be at
                    least min_shift from its previous pose (pose diversity).
    """
    for _ in range(max_restarts):
        placed = {}
        ok = True
        for obj_id, radius in ordered_objects:
            xmin, xmax, ymin, ymax = polygon_bbox(polygon)
            xmin += radius; xmax -= radius; ymin += radius; ymax -= radius
            if xmin >= xmax or ymin >= ymax:
                raise RuntimeError(f"polygon too small for {obj_id} (r={radius})")
            for _ in range(max_attempts):
                x = float(rng.uniform(xmin, xmax))
                y = float(rng.uniform(ymin, ymax))
                if not fits_in_polygon((x, y), polygon, radius):
                    continue
                collision = False
                for other_id, (ox, oy) in placed.items():
                    other_r = next(r for i, r in ordered_objects if i == other_id)
                    if dist((x, y), (ox, oy)) < (radius + other_r + min_gap):
                        collision = True
                        break
                if collision:
                    continue
                if prev_positions is not None:
                    prev = prev_positions.get(obj_id)
                    if prev is not None and dist((x, y), prev) < min_shift:
                        continue
                placed[obj_id] = (x, y)
                break
            else:
                ok = False
                break
        if ok:
            return placed
    raise RuntimeError(f"failed to sample layout after {max_restarts} restarts")


def validate_reset_move_order(move_order, bowl_assignment):
    """Require all six objects exactly once and every cup before its bowl."""
    if (set(bowl_assignment) != set(COLORS) or
            sorted(bowl_assignment.values()) != sorted(COLORS)):
        raise RuntimeError("reset plan has an incomplete cup-to-bowl assignment")

    expected = ({f"cup_{color}" for color in COLORS} |
                {f"bowl_{color}" for color in COLORS})
    if len(move_order) != len(expected) or set(move_order) != expected:
        raise RuntimeError(f"invalid reset move order: {move_order}")

    positions = {obj_id: index for index, obj_id in enumerate(move_order)}
    for cup_color, bowl_color in bowl_assignment.items():
        cup_id = f"cup_{cup_color}"
        bowl_id = f"bowl_{bowl_color}"
        if positions[cup_id] > positions[bowl_id]:
            raise RuntimeError(
                f"unsafe reset order: {bowl_id} appears before {cup_id}"
            )


def sample_reset_move_order(rng, bowl_assignment):
    """Uniformly sample one of the 90 dependency-valid reset orders.

    Start with a uniform permutation, then orient each disjoint cup/bowl pair
    by swapping its two positions when necessary. Every valid order has exactly
    2^3 source permutations, so the resulting distribution remains uniform.
    """
    move_order = ([f"cup_{color}" for color in COLORS] +
                  [f"bowl_{color}" for color in COLORS])
    rng.shuffle(move_order)

    for cup_color, bowl_color in bowl_assignment.items():
        cup_id = f"cup_{cup_color}"
        bowl_id = f"bowl_{bowl_color}"
        cup_index = move_order.index(cup_id)
        bowl_index = move_order.index(bowl_id)
        if bowl_index < cup_index:
            move_order[cup_index], move_order[bowl_index] = (
                move_order[bowl_index], move_order[cup_index]
            )

    validate_reset_move_order(move_order, bowl_assignment)
    return move_order


def plan_sequential_reset(rng, move_order, bowl_assignment,
                          current, catalog, polygon, min_gap, max_attempts,
                          max_restarts, min_shift=0.03,
                          transition_gap=0.03):
    """Plan reset destinations against the objects present at each move.

    Immediately after the demonstration, each cup is inside its assigned bowl.
    Thus the initial occupied table state is the three bowls. A bowl's old
    footprint is removed only when that bowl is picked up. The move order may
    otherwise be any permutation satisfying cup-before-containing-bowl.

    This transition-aware planning is essential: checking only the six final
    destinations can choose a cup target on top of an as-yet-unmoved bowl.
    """
    validate_reset_move_order(move_order, bowl_assignment)

    last_error = None
    for _ in range(max_restarts):
        occupied = {}
        layout = {}

        # A cup sitting inside a bowl does not add another external footprint.
        for color in COLORS:
            bowl_id = f"bowl_{color}"
            occupied[bowl_id] = (
                tuple(current[bowl_id]),
                float(catalog[bowl_id]["footprint_radius_m"]),
                transition_gap,
            )

        try:
            for obj_id in move_order:
                radius = float(catalog[obj_id]["footprint_radius_m"])
                # Free a bowl's source footprint only at the point in the
                # physical sequence where that bowl will actually be lifted.
                if obj_id.startswith("bowl_"):
                    del occupied[obj_id]
                target = sample_clear_position(
                    rng, radius, polygon, occupied, max_attempts,
                    previous=tuple(current[obj_id]), min_shift=min_shift,
                )
                layout[obj_id] = target
                occupied[obj_id] = (target, radius, min_gap)

            return layout
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        f"failed to plan a collision-free sequential reset after "
        f"{max_restarts} restarts: {last_error}"
    )


def validate_sequential_reset(layout, move_order, bowl_assignment,
                              current, catalog, polygon, min_gap,
                              transition_gap):
    """Preflight the exact reset sequence before allowing robot motion."""
    validate_reset_move_order(move_order, bowl_assignment)
    occupied = {
        f"bowl_{color}": (
            tuple(current[f"bowl_{color}"]),
            float(catalog[f"bowl_{color}"]["footprint_radius_m"]),
            transition_gap,
        )
        for color in COLORS
    }

    def require_clear(obj_id):
        target = tuple(layout[obj_id])
        radius = float(catalog[obj_id]["footprint_radius_m"])
        if not fits_in_polygon(target, polygon, radius):
            raise RuntimeError(f"unsafe reset target for {obj_id}: outside table padding")
        for other_id, (other_xy, other_radius, required_gap) in occupied.items():
            edge_gap = dist(target, other_xy) - radius - other_radius
            if edge_gap < required_gap:
                raise RuntimeError(
                    f"unsafe reset target for {obj_id}: edge gap to {other_id} "
                    f"is {edge_gap:.3f} m, requires {required_gap:.3f} m"
                )
        return target, radius

    for obj_id in move_order:
        if obj_id.startswith("bowl_"):
            del occupied[obj_id]
        target, radius = require_clear(obj_id)
        occupied[obj_id] = (target, radius, min_gap)


# ------------------------------------------------------------------ schedule

def assignment_schedule(subset, total, forced, unrestricted, seed,
                        batch_size=None, forced_by_batch=None):
    """Deterministic interleaved F/U schedule.

    When forced_by_batch is provided, each batch is shuffled independently.
    This makes separate 10-episode invocations preserve their exact quota.
    """
    assert forced + unrestricted == total

    if forced_by_batch is not None:
        if not batch_size or batch_size <= 0:
            raise ValueError("batch_size must be positive when forced_by_batch is set")
        expected_batches = (total + batch_size - 1) // batch_size
        if len(forced_by_batch) != expected_batches:
            raise ValueError(
                f"forced_by_batch has {len(forced_by_batch)} entries; "
                f"expected {expected_batches}"
            )
        tags = []
        for batch_index, batch_forced in enumerate(forced_by_batch):
            this_size = min(batch_size, total - len(tags))
            if not 0 <= batch_forced <= this_size:
                raise ValueError(f"invalid forced count for batch {batch_index + 1}")
            batch_tags = (["forced"] * int(batch_forced) +
                          ["unrestricted"] * (this_size - int(batch_forced)))
            digest = hashlib.sha256(
                f"{subset}:{seed}:batch:{batch_index}".encode("utf-8")
            ).digest()
            batch_rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            batch_rng.shuffle(batch_tags)
            tags.extend(batch_tags)
        if tags.count("forced") != forced or tags.count("unrestricted") != unrestricted:
            raise ValueError("per-batch policy counts do not match subset totals")
        return tags

    tags = ["forced"] * forced + ["unrestricted"] * unrestricted
    # Python's built-in hash is randomized for every interpreter process,
    # which would change the schedule after a stopped/resumed collection.
    digest = hashlib.sha256(f"{subset}:{seed}".encode("utf-8")).digest()
    schedule_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(schedule_seed)
    rng.shuffle(tags)
    return tags


def clutter_roster_schedule(plan, subset, total, catalog):
    """Expand clutter blocks into episodes in their exact configured order."""
    schedules = plan.get("clutter_schedules", {})
    if subset not in schedules:
        raise ValueError(f"missing clutter_schedules entry for {subset}")

    expected_level = 0 if subset == "clean" else int(subset.removeprefix("d"))
    rosters = []
    for entry in schedules[subset]:
        object_ids = list(entry.get("objects", []))
        count = int(entry.get("count", 0))
        if count <= 0:
            raise ValueError(f"invalid clutter count for {subset}: {entry}")
        if len(object_ids) != expected_level:
            raise ValueError(
                f"{subset} roster {object_ids} has {len(object_ids)} objects; "
                f"expected {expected_level}"
            )
        if len(set(object_ids)) != len(object_ids):
            raise ValueError(f"duplicate object in {subset} roster: {object_ids}")
        unknown = [obj_id for obj_id in object_ids if obj_id not in catalog]
        if unknown:
            raise ValueError(f"unknown clutter objects in {subset}: {unknown}")
        rosters.extend([tuple(object_ids)] * count)

    if len(rosters) != total:
        raise ValueError(
            f"{subset} clutter quotas total {len(rosters)} episodes; expected {total}"
        )

    return [list(roster) for roster in rosters]


def exact_assignment_instruction(bowl_assignment):
    """Describe the complete cup-to-bowl goal in a stable color order."""
    parts = [
        f"the {color} cup in the {bowl_assignment[color]} bowl"
        for color in COLORS
    ]
    return f"place {parts[0]}, {parts[1]}, and {parts[2]}"


def generic_forced_instruction_episodes(plan, seed):
    """Keep alternating forced episodes generic across the full dataset.

    With the configured 61 forced demonstrations this deterministically gives
    31 generic same-color instructions and 30 exact assignment instructions.
    The ordering matches audit_three_cups_dataset.py and remains stable when a
    stopped collection is resumed.
    """
    forced_refs = []
    for subset in ("clean", "d1", "d2", "d3", "d4"):
        counts = plan["subset_counts"][subset]
        subset_schedule = assignment_schedule(
            subset,
            counts["total"],
            counts["forced"],
            counts["unrestricted"],
            seed,
            batch_size=counts.get("batch_size"),
            forced_by_batch=counts.get("forced_by_batch"),
        )
        forced_refs.extend(
            (subset, episode)
            for episode, policy in enumerate(subset_schedule, 1)
            if policy == "forced"
        )
    return set(forced_refs[::2])


# ------------------------------------------------------------------ atomic io

def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_state(path, state):
    write_json_atomic(path, state)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--config", default=os.path.join(THIS_DIR, "collect", "config.yaml"))
    ap.add_argument("--robot-ip", default="192.168.56.101")
    ap.add_argument("--subset", required=True,
                    choices=["clean", "d1", "d2", "d3", "d4"])
    ap.add_argument("--n-episodes", "-n", type=int, default=None,
                    help="How many eps this run (default = subset total).")
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed. Assignment schedule = (subset, seed).")
    ap.add_argument("--assignment", choices=["auto", "forced", "unrestricted"], default="auto",
                    help="`auto` uses the per-subset deterministic schedule. "
                         "Otherwise forces every ep in this run to one policy.")
    ap.add_argument("--record-depth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reset-pause", type=float, default=None,
                    help="Seconds to pause for clutter setup. Defaults to the "
                         "per-subset value in the plan YAML.")
    ap.add_argument("--reset-state", action="store_true",
                    help="Ignore saved state.json; redo initial manual hover placement.")
    ap.add_argument("--reset-after-last", action="store_true",
                    help="Also reshuffle after the final episode. Useful for a one-episode "
                         "verification run that must finish with all six objects separated.")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    v = plan["verify"]
    catalog = catalog_by_id(plan)
    if args.reset_pause is None:
        pause_defaults = plan.get("reset_pause_seconds", {})
        if args.subset not in pause_defaults:
            raise ValueError(f"missing reset pause default for {args.subset}")
        args.reset_pause = float(pause_defaults[args.subset])
    if args.reset_pause < 0:
        raise ValueError("--reset-pause cannot be negative")

    hover_z            = float(v["hover_z"])
    pick_z_cup         = float(v["pick_z_cup"])
    pick_z_bowl        = float(v["pick_z_bowl"])
    pick_z_cup_in_bowl = float(v["pick_z_cup_in_bowl"])
    release_z_in_bowl  = float(v["release_z_cup_in_bowl"])
    lift_z             = float(v["lift_z"])
    hover_pause        = float(v.get("hover_pause", 6.0))
    speed              = float(v.get("speed", 0.20))
    accel              = float(v.get("accel", 0.40))
    descend_speed      = float(v.get("descend_speed", 0.05))
    descend_accel      = float(v.get("descend_accel", 0.40))
    grip_pos           = int(v.get("grip_pos", 200))
    transit_grip_pos   = int(v.get("transit_grip_pos", 0))
    home_parts         = tuple(float(x) for x in v["home"].split(","))
    hx, hy             = home_parts[0], home_parts[1]
    hz_home            = home_parts[2] if len(home_parts) >= 3 else lift_z

    polygon = plan["workspace"]["corners"]

    # Counts & schedule
    counts = plan["subset_counts"][args.subset]
    total, forced_n, unr_n = counts["total"], counts["forced"], counts["unrestricted"]
    schedule = assignment_schedule(
        args.subset, total, forced_n, unr_n, args.seed,
        batch_size=counts.get("batch_size"),
        forced_by_batch=counts.get("forced_by_batch"),
    )
    clutter_schedule = clutter_roster_schedule(
        plan, args.subset, total, catalog,
    )
    generic_forced_episodes = generic_forced_instruction_episodes(plan, args.seed)

    n_eps = args.n_episodes if args.n_episodes is not None else total
    end_ep = min(args.start_episode + n_eps - 1, total)
    if args.start_episode > total or end_ep < args.start_episode:
        raise SystemExit(f"{args.subset} has only {total} eps; requested range invalid")

    subset_root = os.path.join(DATA_ROOT, TASK_NAME, args.subset)
    episodes_dir = os.path.join(subset_root, "data")
    state_path = os.path.join(subset_root, "state.json")
    positions_log = os.path.join(subset_root, "positions.jsonl")
    os.makedirs(episodes_dir, exist_ok=True)

    sampling = plan.get("sampling", {})
    min_gap = float(sampling.get("min_object_gap_m", 0.02))
    max_attempts = int(sampling.get("max_attempts_per_object", 100))
    max_restarts = int(sampling.get("max_layout_restarts", 20))
    min_reset_shift = float(sampling.get("min_reset_shift_m", 0.03))
    transition_gap = float(sampling.get("transition_object_gap_m", min_gap))

    language = plan["language_instructions"]

    if args.assignment == "auto":
        scheduled_slice = schedule[args.start_episode - 1:end_ep]
        run_policy_counts = {
            "forced": scheduled_slice.count("forced"),
            "unrestricted": scheduled_slice.count("unrestricted"),
        }
    else:
        run_policy_counts = {
            "forced": (end_ep - args.start_episode + 1
                       if args.assignment == "forced" else 0),
            "unrestricted": (end_ep - args.start_episode + 1
                             if args.assignment == "unrestricted" else 0),
        }

    clutter_quota_summary = []
    seen_rosters = set()
    for roster in clutter_schedule:
        key = tuple(roster)
        if key in seen_rosters:
            continue
        seen_rosters.add(key)
        clutter_quota_summary.append({
            "objects": roster,
            "episodes": sum(tuple(item) == key for item in clutter_schedule),
        })

    # Cameras
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cam_serials = cfg["recording"]["camera_serials"]
    rec_hz = cfg["recording"]["hz"]
    cam_res = tuple(cfg["recording"]["camera_resolution"])

    # Print effective params
    params = {
        "task": TASK_NAME, "subset": args.subset,
        "start_ep": args.start_episode, "end_ep": end_ep,
        "counts": counts,
        "assignment_override": args.assignment,
        "scheduled_run_counts": run_policy_counts,
        "seed": args.seed,
        "clutter_quotas": clutter_quota_summary,
        "first_episode_clutter": clutter_schedule[args.start_episode - 1],
        "reset_pause": args.reset_pause,
        "reset_order_algorithm": "random_dependency_aware",
        "polygon": polygon,
        "heights": {
            "hover_z": hover_z, "pick_z_cup": pick_z_cup, "pick_z_bowl": pick_z_bowl,
            "pick_z_cup_in_bowl": pick_z_cup_in_bowl,
            "release_z_cup_in_bowl": release_z_in_bowl, "lift_z": lift_z,
        },
    }
    print(json.dumps(params, indent=2))
    print(f"state: {state_path}")
    print(f"positions: {positions_log}")
    print(f"cameras: {cam_serials}")

    # Connect
    print(f"\nConnecting to UR5e at {args.robot_ip} ...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    grip = robotiq_gripper.RobotiqGripper()
    grip.connect(args.robot_ip, 63352)
    grip.activate()
    grip.move_and_wait_for_pos(transit_grip_pos, 255, 255)

    context = {
        "chunk_index": -1, "phase": "idle",
        "active_target_id": "", "active_destination_id": "",
    }
    recorder = DataRecorder(
        rtde_r=rtde_r, gripper=grip,
        camera_serials=cam_serials, hz=rec_hz, camera_resolution=cam_res,
        record_depth=args.record_depth,
        context=context,
    )
    recorder.test()

    # --- motion primitives ---
    def go(x, y, z):
        rtde_c.moveL([x, y, z, RX, RY, RZ], speed, accel)

    def hold_frames(n):
        time.sleep(n / rec_hz)

    def set_ctx(**kv):
        context.update(kv)

    def pick_at(x, y, z_pick):
        set_ctx(phase="approach")
        go(x, y, hover_z)
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        set_ctx(phase="grasp")
        go(x, y, z_pick)
        grip.move_and_wait_for_pos(grip_pos, 125, 125)
        time.sleep(0.15)
        grasp_frame, grasp_ts = recorder.mark()
        set_ctx(phase="transport")
        go(x, y, lift_z)
        return grasp_frame, grasp_ts

    def release_at(x, y, z_release):
        set_ctx(phase="transport")
        go(x, y, hover_z)
        set_ctx(phase="place")
        go(x, y, z_release)
        set_ctx(phase="release")
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        release_frame, release_ts = recorder.mark()
        set_ctx(phase="settle")
        time.sleep(0.5)
        go(x, y, lift_z)
        set_ctx(phase="hold_calibration")
        hold_frames(CALIBRATION_HOLD_FRAMES)
        completion_frame, completion_ts = recorder.mark()
        return release_frame, release_ts, completion_frame, completion_ts

    def unrecorded_pick_place(from_xy, pick_z, to_xy):
        """Robot-driven reset move (not recorded)."""
        x1, y1 = from_xy; x2, y2 = to_xy
        go(x1, y1, hover_z)
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        go(x1, y1, pick_z)
        grip.move_and_wait_for_pos(grip_pos, 125, 125)
        time.sleep(0.15)
        go(x1, y1, lift_z)
        go(x2, y2, hover_z)
        go(x2, y2, hover_z)  # settle to hover
        # Release using moveUntilContact so the bowl (heavier) sits flush
        rtde_c.moveUntilContact([0.0, 0.0, -descend_speed, 0.0, 0.0, 0.0],
                                [0.0]*6, descend_accel)
        grip.move_and_wait_for_pos(transit_grip_pos, 125, 125)
        time.sleep(0.15)
        go(x2, y2, lift_z)

    # ---------- initial state ----------
    if args.reset_state or not os.path.exists(state_path):
        # Trust that objects are physically at the plan's initial coordinates.
        # No hover phase -- go straight into ep 1 from these positions.
        initial = {f"cup_{c}":  parse_xy(v[f"cup_{c}"])  for c in COLORS}
        initial.update({f"bowl_{c}": parse_xy(v[f"bowl_{c}"]) for c in COLORS})
        print(f"\n[initial positions from plan yaml -- assuming objects are placed]")
        for oid, xy in initial.items():
            print(f"  {oid:14s} ({xy[0]:+.3f},{xy[1]:+.3f})")
        state = {oid: list(xy) for oid, xy in initial.items()}
        state_pose_source = "human_placed_from_plan"
        save_state(state_path, state)
        go(hx, hy, hz_home)
    else:
        state = load_state(state_path)
        state_pose_source = "robot_placed_from_plan"
        print(f"\n[resuming from {state_path}]")
        print(json.dumps(state, indent=2))

    # A new subset has no preceding reset pause in which to place clutter.
    # Give the operator the subset-specific setup window before episode 1.
    if args.start_episode == 1 and clutter_schedule[0]:
        initial_clutter = clutter_schedule[0]
        initial_names = " + ".join(
            str(catalog[obj_id].get("name", obj_id)) for obj_id in initial_clutter
        )
        print(f"\n[initial clutter setup for episode 1: {initial_names}]")
        if args.reset_pause > 0:
            print(f"  [setup-pause {args.reset_pause}s] place clutter now")
            time.sleep(args.reset_pause)

    ep_rng = np.random.default_rng(args.seed)
    for _ in range(args.start_episode - 1):
        ep_rng.integers(0, 2**31 - 1)  # burn draws for skipped eps

    # ---------- main loop ----------
    try:
        for ep in range(args.start_episode, end_ep + 1):
            print(f"\n{'='*70}\nEPISODE {ep}/{total} ({args.subset})\n{'='*70}")

            clutter_ids = list(clutter_schedule[ep - 1])
            clutter_names = " + ".join(
                str(catalog[obj_id].get("name", obj_id)) for obj_id in clutter_ids
            ) or "none"
            print(f"  clutter: {clutter_names}")

            ep_dir = os.path.join(episodes_dir, str(ep))
            if os.path.isdir(ep_dir) and any(os.scandir(ep_dir)):
                raise SystemExit(f"REFUSING to overwrite non-empty ep dir: {ep_dir}")

            # Policy / ordering
            policy = args.assignment if args.assignment != "auto" else schedule[ep - 1]
            ep_seed = int(ep_rng.integers(0, 2**31 - 1))
            layout_rng = np.random.default_rng(ep_seed)

            cup_order = list(COLORS); layout_rng.shuffle(cup_order)
            if policy == "forced":
                bowl_assignment = {c: c for c in COLORS}
            else:
                bowls_perm = list(COLORS); layout_rng.shuffle(bowls_perm)
                bowl_assignment = dict(zip(COLORS, bowls_perm))

            if policy == "forced" and (args.subset, ep) in generic_forced_episodes:
                lang = language["forced_same_color"]
            else:
                lang = exact_assignment_instruction(bowl_assignment)

            # Current positions from state
            current = {oid: tuple(xy) for oid, xy in state.items()}

            print(f"  policy: {policy}    language: \"{lang}\"")
            print(f"  cup order: {cup_order}")
            for c in COLORS:
                match = "MATCH" if c == bowl_assignment[c] else "diff"
                print(f"     cup_{c:6s} -> bowl_{bowl_assignment[c]:6s}  [{match}]")

            # Record
            set_ctx(chunk_index=-1, phase="setup",
                    active_target_id="", active_destination_id="")
            os.makedirs(ep_dir, exist_ok=True)
            recorder.start(ep_dir)
            steps_log = []
            episode_status = "in_progress"
            try:
                set_ctx(phase="hold_calibration")
                hold_frames(CALIBRATION_HOLD_FRAMES)

                for chunk_index, cup_color in enumerate(cup_order):
                    bowl_color = bowl_assignment[cup_color]
                    target_id = f"cup_{cup_color}"
                    dest_id = f"bowl_{bowl_color}"
                    cx, cy = current[target_id]
                    bx, by = current[dest_id]
                    set_ctx(chunk_index=chunk_index, phase="approach",
                            active_target_id=target_id, active_destination_id=dest_id)
                    start_frame, start_ts = recorder.mark()
                    print(f"  [chunk {chunk_index}] {target_id} -> {dest_id}  "
                          f"cup=({cx:+.3f},{cy:+.3f})  bowl=({bx:+.3f},{by:+.3f})")
                    grasp_frame, grasp_ts = pick_at(cx, cy, pick_z_cup)
                    release_frame, release_ts, completion_frame, completion_ts = \
                        release_at(bx, by, release_z_in_bowl)
                    steps_log.append({
                        "chunk_index": chunk_index,
                        "target_id": target_id, "destination_id": dest_id,
                        "assignment_policy": ("forced_same_color" if policy == "forced"
                                              else "unrestricted"),
                        "color_match": (cup_color == bowl_color),
                        "start_frame": start_frame, "start_ts": start_ts,
                        "grasp_frame": grasp_frame, "grasp_ts": grasp_ts,
                        "release_frame": release_frame, "release_ts": release_ts,
                        "completion_frame": completion_frame, "completion_ts": completion_ts,
                        "success": True,
                    })

                set_ctx(chunk_index=-1, phase="idle",
                        active_target_id="", active_destination_id="")
                go(hx, hy, hz_home)
                episode_status = "completed"
            except Exception as e:
                episode_status = f"failed: {type(e).__name__}: {e}"
                print(f"  EPISODE FAILED: {e}")
                raise
            finally:
                recorder.stop()

            # ----- Task_sequence.json (authoritative) -----
            # `current` = positions at episode START (cups on table, bowls on table)
            objects_json = []
            for c in COLORS:
                for oid, otype in ((f"cup_{c}", "cup"), (f"bowl_{c}", "bowl")):
                    cat = catalog[oid]
                    x, y = current[oid]
                    objects_json.append({
                        "id": oid, "type": otype, "color": c, "group": "task_object",
                        "geometry_class": cat.get("geometry_class"),
                        "footprint_radius_m": cat.get("footprint_radius_m"),
                        "planned_pose": {
                            "position": [x, y, 0.0],
                            "quaternion": [0.0, 0.0, 0.0, 1.0],
                        },
                        "measured_pose": None,
                        "pose_source": state_pose_source,
                    })
            clutter_json = [{
                "id": cl_id, "name": catalog[cl_id].get("name"),
                "geometry_class": catalog[cl_id].get("geometry_class"),
                "planned_pose": None,
                "measured_pose": None,
                "pose_source": "human_placed",
            } for cl_id in clutter_ids]

            task_seq = {
                "schema_version": SCHEMA_VERSION,
                "task": TASK_NAME, "condition": args.subset,
                "clutter_level": len(clutter_ids),
                "assignment_policy": ("forced_same_color" if policy == "forced"
                                      else "unrestricted"),
                "language_instruction": lang,
                "coordinate_frame": "robot_base", "units": "m", "quat_convention": "xyzw",
                "objects": objects_json,
                "clutter_objects": clutter_json,
                "steps": steps_log,
                "episode_status": episode_status,
                "episode_success": (episode_status == "completed"),
                "cup_order": cup_order,
                "bowl_assignment": bowl_assignment,
                "seed": {"base": args.seed, "episode": ep_seed},
                "recorded_at": datetime.utcnow().isoformat() + "Z",
            }
            write_json_atomic(os.path.join(ep_dir, "task_sequence.json"), task_seq)

            # meta.json (compact robot params)
            meta = {
                "task": TASK_NAME, "subset": args.subset, "episode": ep,
                "assignment_policy": task_seq["assignment_policy"],
                "record_depth": args.record_depth,
                "hover_z": hover_z, "pick_z_cup": pick_z_cup, "pick_z_bowl": pick_z_bowl,
                "pick_z_cup_in_bowl": pick_z_cup_in_bowl,
                "release_z_cup_in_bowl": release_z_in_bowl, "lift_z": lift_z,
                "speed": speed, "accel": accel,
                "descend_speed": descend_speed, "descend_accel": descend_accel,
                "grip_pos": grip_pos, "transit_grip_pos": transit_grip_pos,
                "home": v["home"], "reset_pause": args.reset_pause,
                "ep_seed": ep_seed, "base_seed": args.seed,
                "plan_path": args.plan,
            }
            write_json_atomic(os.path.join(ep_dir, "meta.json"), meta)

            # positions.jsonl (derived index)
            with open(positions_log, "a") as f:
                f.write(json.dumps({
                    "episode": ep, "policy": policy,
                    "cup_order": cup_order, "bowl_assignment": bowl_assignment,
                    "start_positions": {oid: list(xy) for oid, xy in current.items()},
                    "clutter": clutter_ids,
                    "timestamp": time.time(),
                }) + "\n")

            print(f"  -> {ep_dir} ({episode_status})")

            # ----- RESET (not recorded): reshuffle 6 objects for next ep -----
            if ep < end_ep or args.reset_after_last:
                print(f"\n--- RESET (not recorded) ---")
                # Plan against the objects that are physically present during
                # every move. Any of the 90 valid orders may be selected; the
                # only dependency is cup-before-the-bowl-that-contains-it.
                reset_move_order = sample_reset_move_order(
                    layout_rng, bowl_assignment,
                )
                print(f"  reset order: {reset_move_order}")
                new_layout = plan_sequential_reset(
                    layout_rng, reset_move_order, bowl_assignment,
                    current, catalog, polygon,
                    min_gap=min_gap,
                    max_attempts=max_attempts,
                    max_restarts=max_restarts,
                    min_shift=min_reset_shift,
                    transition_gap=transition_gap,
                )
                # Fail before any physical reset motion if a future change ever
                # violates the transition-aware clearance model.
                validate_sequential_reset(
                    new_layout, reset_move_order, bowl_assignment,
                    current, catalog, polygon, min_gap, transition_gap,
                )
                # Execute the exact order that was planned and preflighted.
                for obj_id in reset_move_order:
                    if obj_id.startswith("cup_"):
                        cup_color = obj_id.removeprefix("cup_")
                        source_bowl_id = f"bowl_{bowl_assignment[cup_color]}"
                        from_xy = current[source_bowl_id]
                        pick_z = pick_z_cup_in_bowl
                        kind = "cup"
                    else:
                        from_xy = current[obj_id]
                        pick_z = pick_z_bowl
                        kind = "bowl"
                    to_xy = new_layout[obj_id]
                    print(f"  reset {kind} {obj_id}: "
                          f"({from_xy[0]:+.3f},{from_xy[1]:+.3f}) "
                          f"-> ({to_xy[0]:+.3f},{to_xy[1]:+.3f})")
                    unrecorded_pick_place(from_xy, pick_z, to_xy)
                    current[obj_id] = to_xy
                go(hx, hy, hz_home)
                state = {oid: list(xy) for oid, xy in current.items()}
                state_pose_source = "robot_placed_from_plan"
                save_state(state_path, state)

                next_ep = ep + 1
                if next_ep <= total:
                    next_clutter = clutter_schedule[next_ep - 1]
                    next_names = " + ".join(
                        str(catalog[obj_id].get("name", obj_id))
                        for obj_id in next_clutter
                    ) or "none"
                    print(f"  next episode {next_ep} clutter: {next_names}")
                    if args.reset_pause > 0:
                        print(f"  [reset-pause {args.reset_pause}s] "
                              f"place / swap clutter now")
                        time.sleep(args.reset_pause)

        print(f"\nDONE: episodes {args.start_episode}..{end_ep} collected.")

    except KeyboardInterrupt:
        print("\nInterrupted. State saved.")
        save_state(state_path, {oid: list(xy) for oid, xy in current.items()})
    except Exception as e:
        print(f"\nERROR: {e}. State saved.")
        try:
            save_state(state_path, {oid: list(xy) for oid, xy in current.items()})
        except Exception:
            pass
        raise
    finally:
        try:
            recorder.stop(); recorder.close()
        except Exception:
            pass
        rtde_c.stopScript()


if __name__ == "__main__":
    main()
