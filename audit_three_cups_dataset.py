#!/usr/bin/env python3
"""Audit and, optionally, rewrite place_three_cups_in_bowls metadata.

The audit treats task_sequence.json as authoritative and cross-checks it
against meta.json, positions.jsonl, state.csv, RGB/depth frame inventories,
camera intrinsics, the configured clutter roster, and geometric clearances.

Instruction rewriting is deterministic:
  * every unrestricted episode receives its exact three-cup assignment;
  * of the 61 forced episodes, alternating episodes in global dataset order
    retain the generic same-color instruction (31 generic / 30 exact).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


TASK = "place_three_cups_in_bowls"
SUBSETS = ("clean", "d1", "d2", "d3", "d4")
EXPECTED_COUNTS = {"clean": 100, "d1": 25, "d2": 25, "d3": 25, "d4": 25}
COLORS = ("blue", "pink", "yellow")
TASK_IDS = {f"{kind}_{color}" for kind in ("cup", "bowl") for color in COLORS}
CAMERAS = ("camera1", "camera2", "camera3")
FRAME_DIRS = CAMERAS + tuple(f"{camera}_depth" for camera in CAMERAS)
GENERIC_FORCED = "place each cup in the same color bowl"
LEGACY_UNRESTRICTED = "place each cup in any bowl"
REQUIRED_CSV_COLUMNS = {
    "frame_idx", "timestamp", "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "tcp_x", "tcp_y", "tcp_z", "tcp_qx",
    "tcp_qy", "tcp_qz", "tcp_qw", "gripper_pos", "camera1_file",
    "camera2_file", "camera3_file", "chunk_index", "phase",
    "active_target_id", "active_destination_id",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def exact_instruction(assignment: dict[str, str]) -> str:
    parts = [f"the {color} cup in the {assignment[color]} bowl" for color in COLORS]
    return f"place {parts[0]}, {parts[1]}, and {parts[2]}"


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = len(polygon) - 1
    for index, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[previous]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        previous = index
    return inside


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist(point, start)
    amount = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return math.dist(point, (ax + amount * dx, ay + amount * dy))


def footprint_inside(
    point: tuple[float, float], polygon: list[tuple[float, float]], radius: float
) -> bool:
    if not point_in_polygon(point, polygon):
        return False
    margin = min(
        point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    return margin + 1e-9 >= radius


def expanded_clutter_schedule(plan: dict[str, Any], subset: str) -> list[list[str]]:
    schedule: list[list[str]] = []
    for block in plan["clutter_schedules"][subset]:
        schedule.extend([list(block["objects"])] * int(block["count"]))
    return schedule


def numeric_png_names(directory: Path) -> list[str]:
    names = [item.name for item in directory.iterdir() if item.is_file() and item.suffix == ".png"]
    return sorted(names, key=lambda name: float(Path(name).stem))


def inspect_image(path: Path, depth: bool) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return "cannot be decoded"
    if image.shape[:2] != (480, 640):
        return f"unexpected shape {image.shape}"
    if depth:
        if image.ndim != 2 or image.dtype != np.uint16:
            return f"expected uint16 depth, got shape={image.shape} dtype={image.dtype}"
    elif image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return f"expected uint8 color, got shape={image.shape} dtype={image.dtype}"
    if not np.any(image):
        return "image is entirely zero"
    return None


def episode_ref(path: Path) -> str:
    return f"{path.parent.parent.parent.name}/{path.parent.name}"


def forced_instruction_split(episodes: list[tuple[str, int, Path, dict[str, Any]]]) -> set[tuple[str, int]]:
    forced = [
        (subset, episode)
        for subset, episode, _path, task_sequence in episodes
        if task_sequence.get("assignment_policy") == "forced_same_color"
    ]
    return set(forced[::2])


def audit(root: Path, plan_path: Path, expect_rewritten: bool) -> dict[str, Any]:
    plan = yaml.safe_load(plan_path.read_text())
    polygon = [tuple(map(float, point)) for point in plan["workspace"]["corners"]]
    minimum_gap = float(plan["sampling"]["min_object_gap_m"])
    catalog = {
        item["id"]: item
        for group in ("task_objects", "clutter_pool")
        for item in plan["object_catalog"][group]
    }
    radii = {item_id: float(item["footprint_radius_m"]) for item_id, item in catalog.items() if item_id in TASK_IDS}

    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    episodes: list[tuple[str, int, Path, dict[str, Any]]] = []
    frame_total = 0
    png_total = 0
    decoded_samples = 0
    minimum_observed_gap = float("inf")

    position_indexes: dict[str, dict[int, dict[str, Any]]] = {}
    for subset in SUBSETS:
        index_path = root / subset / "positions.jsonl"
        index: dict[int, dict[str, Any]] = {}
        if not index_path.is_file():
            errors.append(f"{subset}: missing positions.jsonl")
        else:
            for line_number, line in enumerate(index_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    episode = int(record["episode"])
                except Exception as exc:
                    errors.append(f"{subset}:positions.jsonl:{line_number}: {exc}")
                    continue
                if episode in index:
                    errors.append(f"{subset}: duplicate positions entry for episode {episode}")
                index[episode] = record
        position_indexes[subset] = index

    for subset in SUBSETS:
        data_dir = root / subset / "data"
        expected_numbers = list(range(1, EXPECTED_COUNTS[subset] + 1))
        if not data_dir.is_dir():
            errors.append(f"{subset}: missing data directory")
            continue
        actual_numbers = sorted(
            int(item.name)
            for item in data_dir.iterdir()
            if item.is_dir() and item.name.isdigit()
        )
        if actual_numbers != expected_numbers:
            errors.append(f"{subset}: episode directories are {actual_numbers}, expected {expected_numbers}")
        index_numbers = sorted(position_indexes[subset])
        if index_numbers != expected_numbers:
            errors.append(f"{subset}: positions episodes are {index_numbers}, expected {expected_numbers}")

        clutter_schedule = expanded_clutter_schedule(plan, subset)
        if len(clutter_schedule) != EXPECTED_COUNTS[subset]:
            errors.append(f"{subset}: configured clutter schedule has {len(clutter_schedule)} entries")

        for episode in actual_numbers:
            ep_dir = data_dir / str(episode)
            reference = f"{subset}/{episode}"
            required_files = ("task_sequence.json", "meta.json", "intrinsics.json", "state.csv")
            missing = [name for name in required_files if not (ep_dir / name).is_file()]
            missing += [name for name in FRAME_DIRS if not (ep_dir / name).is_dir()]
            if missing:
                errors.append(f"{reference}: missing {missing}")
                continue
            try:
                task_sequence = load_json(ep_dir / "task_sequence.json")
                meta = load_json(ep_dir / "meta.json")
                intrinsics = load_json(ep_dir / "intrinsics.json")
            except Exception as exc:
                errors.append(f"{reference}: JSON read failed: {exc}")
                continue
            episodes.append((subset, episode, ep_dir / "task_sequence.json", task_sequence))

            policy = task_sequence.get("assignment_policy")
            counts[subset]["episodes"] += 1
            counts[subset][str(policy)] += 1
            if task_sequence.get("task") != TASK or task_sequence.get("condition") != subset:
                errors.append(f"{reference}: task/condition mismatch")
            if task_sequence.get("episode_status") != "completed" or task_sequence.get("episode_success") is not True:
                errors.append(f"{reference}: episode not marked completed/successful")
            if policy not in {"forced_same_color", "unrestricted"}:
                errors.append(f"{reference}: invalid assignment policy {policy!r}")

            assignment = task_sequence.get("bowl_assignment", {})
            if set(assignment) != set(COLORS) or sorted(assignment.values()) != sorted(COLORS):
                errors.append(f"{reference}: invalid bowl_assignment {assignment}")
                assignment = {}
            if policy == "forced_same_color" and assignment and any(assignment[color] != color for color in COLORS):
                errors.append(f"{reference}: forced policy has non-identity assignment {assignment}")

            cup_order = task_sequence.get("cup_order", [])
            if sorted(cup_order) != sorted(COLORS):
                errors.append(f"{reference}: invalid cup_order {cup_order}")
            steps = task_sequence.get("steps", [])
            if len(steps) != 3:
                errors.append(f"{reference}: expected 3 steps, found {len(steps)}")
            else:
                step_assignment: dict[str, str] = {}
                previous_completion = -1
                for index, step in enumerate(steps):
                    target = step.get("target_id", "")
                    destination = step.get("destination_id", "")
                    cup_color = target.removeprefix("cup_")
                    bowl_color = destination.removeprefix("bowl_")
                    step_assignment[cup_color] = bowl_color
                    if step.get("chunk_index") != index or (index < len(cup_order) and cup_color != cup_order[index]):
                        errors.append(f"{reference}: step {index} order/chunk mismatch")
                    if step.get("success") is not True:
                        errors.append(f"{reference}: step {index} not successful")
                    frames = [step.get(name) for name in ("start_frame", "grasp_frame", "release_frame", "completion_frame")]
                    if not all(isinstance(value, int) for value in frames) or frames != sorted(frames):
                        errors.append(f"{reference}: step {index} frame markers invalid: {frames}")
                    elif frames[0] < previous_completion:
                        errors.append(f"{reference}: step {index} begins before prior completion")
                    previous_completion = frames[-1] if isinstance(frames[-1], int) else previous_completion
                    if step.get("color_match") is not (cup_color == bowl_color):
                        errors.append(f"{reference}: step {index} color_match incorrect")
                if assignment and step_assignment != assignment:
                    errors.append(f"{reference}: step assignment {step_assignment} != {assignment}")

            object_records = task_sequence.get("objects", [])
            by_id = {item.get("id"): item for item in object_records}
            if set(by_id) != TASK_IDS or len(object_records) != 6:
                errors.append(f"{reference}: task object IDs are invalid")
            object_positions: dict[str, tuple[float, float]] = {}
            for item_id in TASK_IDS & set(by_id):
                try:
                    xyz = by_id[item_id]["planned_pose"]["position"]
                    point = (float(xyz[0]), float(xyz[1]))
                    object_positions[item_id] = point
                    if not footprint_inside(point, polygon, radii[item_id]):
                        warnings.append(
                            f"{reference}: {item_id} footprint crosses the conservative workspace "
                            f"boundary at {point}; episode is otherwise complete"
                        )
                except Exception as exc:
                    errors.append(f"{reference}: invalid pose for {item_id}: {exc}")
            positioned = sorted(object_positions)
            for left_index, left in enumerate(positioned):
                for right in positioned[left_index + 1 :]:
                    gap = math.dist(object_positions[left], object_positions[right]) - radii[left] - radii[right]
                    minimum_observed_gap = min(minimum_observed_gap, gap)
                    if gap + 1e-9 < minimum_gap:
                        warnings.append(
                            f"{reference}: {left}/{right} edge gap {gap:.4f} < configured "
                            f"{minimum_gap:.4f}; episode is otherwise complete"
                        )

            expected_clutter = clutter_schedule[episode - 1] if episode <= len(clutter_schedule) else []
            actual_clutter = [item.get("id") for item in task_sequence.get("clutter_objects", [])]
            if actual_clutter != expected_clutter:
                errors.append(f"{reference}: clutter {actual_clutter} != {expected_clutter}")
            if task_sequence.get("clutter_level") != len(expected_clutter):
                errors.append(f"{reference}: clutter level mismatch")

            if (
                meta.get("task") != TASK
                or meta.get("subset") != subset
                or meta.get("episode") != episode
                or meta.get("assignment_policy") != policy
                or meta.get("ep_seed") != task_sequence.get("seed", {}).get("episode")
                or meta.get("base_seed") != task_sequence.get("seed", {}).get("base")
            ):
                errors.append(f"{reference}: meta.json disagrees with task_sequence.json")

            position_record = position_indexes[subset].get(episode)
            if position_record is not None:
                position_policy = "forced_same_color" if position_record.get("policy") == "forced" else position_record.get("policy")
                if (
                    position_policy != policy
                    or position_record.get("cup_order") != cup_order
                    or position_record.get("bowl_assignment") != assignment
                    or position_record.get("clutter") != expected_clutter
                ):
                    errors.append(f"{reference}: positions.jsonl disagrees with task_sequence.json")
                indexed_positions = position_record.get("start_positions", {})
                for item_id, point in object_positions.items():
                    if item_id not in indexed_positions or any(
                        abs(float(indexed_positions[item_id][axis]) - point[axis]) > 1e-9 for axis in (0, 1)
                    ):
                        errors.append(f"{reference}: indexed start position differs for {item_id}")

            if set(intrinsics) != set(CAMERAS):
                errors.append(f"{reference}: intrinsics camera keys are {sorted(intrinsics)}")
            for camera in CAMERAS:
                calibration = intrinsics.get(camera, {})
                if calibration.get("width") != 640 or calibration.get("height") != 480:
                    errors.append(f"{reference}: invalid {camera} dimensions in intrinsics")
                if not calibration.get("serial") or float(calibration.get("depth_scale", 0)) <= 0:
                    errors.append(f"{reference}: incomplete {camera} intrinsics")

            frame_names: dict[str, list[str]] = {}
            for directory_name in FRAME_DIRS:
                try:
                    names = numeric_png_names(ep_dir / directory_name)
                except Exception as exc:
                    errors.append(f"{reference}: cannot list {directory_name}: {exc}")
                    names = []
                frame_names[directory_name] = names
            baseline = frame_names["camera1"]
            for directory_name in FRAME_DIRS[1:]:
                if frame_names[directory_name] != baseline:
                    errors.append(
                        f"{reference}: {directory_name} frame names/count differ from camera1 "
                        f"({len(frame_names[directory_name])} vs {len(baseline)})"
                    )
            if not baseline:
                errors.append(f"{reference}: no frames")
            png_total += sum(len(names) for names in frame_names.values())

            csv_path = ep_dir / "state.csv"
            csv_names = {camera: [] for camera in CAMERAS}
            chunks: dict[int, set[tuple[str, str]]] = defaultdict(set)
            row_count = 0
            previous_timestamp = -float("inf")
            try:
                with csv_path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not REQUIRED_CSV_COLUMNS.issubset(set(reader.fieldnames or [])):
                        errors.append(f"{reference}: state.csv missing required columns")
                    for row_count, row in enumerate(reader, 1):
                        if int(row["frame_idx"]) != row_count - 1:
                            errors.append(f"{reference}: non-sequential frame_idx at row {row_count}")
                            break
                        timestamp = float(row["timestamp"])
                        if timestamp <= previous_timestamp:
                            errors.append(f"{reference}: non-increasing timestamp at row {row_count}")
                            break
                        previous_timestamp = timestamp
                        for camera in CAMERAS:
                            csv_names[camera].append(row[f"{camera}_file"])
                        chunk_index = int(row["chunk_index"])
                        if chunk_index >= 0:
                            chunks[chunk_index].add((row["active_target_id"], row["active_destination_id"]))
            except Exception as exc:
                errors.append(f"{reference}: state.csv read failed: {exc}")
            frame_total += row_count
            if row_count != len(baseline):
                errors.append(f"{reference}: state.csv has {row_count} rows but cameras have {len(baseline)} frames")
            for camera in CAMERAS:
                if csv_names[camera] != baseline:
                    errors.append(f"{reference}: {camera}_file column differs from frame inventory")
            if len(steps) == 3:
                for index, step in enumerate(steps):
                    expected_pair = {(step["target_id"], step["destination_id"])}
                    if chunks.get(index) != expected_pair:
                        errors.append(f"{reference}: CSV chunk {index} target/destination mismatch: {chunks.get(index)}")
                    if int(step["completion_frame"]) > row_count:
                        errors.append(f"{reference}: step {index} completion beyond recorded frames")

            for directory_name, names in frame_names.items():
                if not names:
                    continue
                sample_indices = sorted({0, len(names) // 2, len(names) - 1})
                for sample_index in sample_indices:
                    problem = inspect_image(
                        ep_dir / directory_name / names[sample_index],
                        depth=directory_name.endswith("_depth"),
                    )
                    decoded_samples += 1
                    if problem:
                        errors.append(f"{reference}:{directory_name}/{names[sample_index]}: {problem}")

    generic_forced = forced_instruction_split(episodes)
    if expect_rewritten:
        for subset, episode, _path, task_sequence in episodes:
            assignment = task_sequence.get("bowl_assignment", {})
            if set(assignment) != set(COLORS):
                continue
            policy = task_sequence.get("assignment_policy")
            expected = (
                GENERIC_FORCED
                if policy == "forced_same_color" and (subset, episode) in generic_forced
                else exact_instruction(assignment)
            )
            if task_sequence.get("language_instruction") != expected:
                errors.append(
                    f"{subset}/{episode}: instruction {task_sequence.get('language_instruction')!r} != {expected!r}"
                )

    total_counts = Counter()
    for subset_counts in counts.values():
        total_counts.update(subset_counts)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts_by_subset": {subset: dict(counts[subset]) for subset in SUBSETS},
        "total_episodes": sum(counts[subset]["episodes"] for subset in SUBSETS),
        "forced_generic_target": len(generic_forced),
        "forced_exact_target": total_counts["forced_same_color"] - len(generic_forced),
        "unrestricted_exact_target": total_counts["unrestricted"],
        "state_rows": frame_total,
        "png_files": png_total,
        "decoded_image_samples": decoded_samples,
        "minimum_observed_edge_gap_m": minimum_observed_gap,
        "episodes": episodes,
        "generic_forced": generic_forced,
    }


def rewrite_instructions(root: Path, audit_result: dict[str, Any]) -> dict[str, Any]:
    if not audit_result["ok"]:
        raise RuntimeError("refusing to rewrite a dataset that failed structural audit")
    generic_forced: set[tuple[str, int]] = audit_result["generic_forced"]
    records: list[dict[str, Any]] = []
    changed = 0
    style_counts: Counter[str] = Counter()
    for subset, episode, path, task_sequence in audit_result["episodes"]:
        assignment = task_sequence["bowl_assignment"]
        policy = task_sequence["assignment_policy"]
        if policy == "forced_same_color" and (subset, episode) in generic_forced:
            instruction = GENERIC_FORCED
            style = "generic_same_color"
        else:
            instruction = exact_instruction(assignment)
            style = "exact_assignment"
        old_instruction = task_sequence.get("language_instruction")
        if old_instruction != instruction:
            task_sequence["language_instruction"] = instruction
            write_json_atomic(path, task_sequence)
            changed += 1
        style_counts[style] += 1
        records.append(
            {
                "subset": subset,
                "episode": episode,
                "assignment_policy": policy,
                "bowl_assignment": assignment,
                "instruction_style": style,
                "old_instruction": old_instruction,
                "new_instruction": instruction,
                "task_sequence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "task": TASK,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": (
            "all unrestricted episodes use exact assignments; alternating forced episodes "
            "in clean,d1,d2,d3,d4 episode order retain the generic same-color instruction"
        ),
        "counts": dict(style_counts),
        "changed_files": changed,
        "episodes": records,
    }
    write_json_atomic(root / "instruction_rewrite_manifest.json", manifest)
    return manifest


def printable_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"episodes", "generic_forced", "errors", "warnings"}
    } | {"error_count": len(result["errors"]), "warning_count": len(result["warnings"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / TASK,
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parent / "collect" / "plan_place_three_cups_in_bowls.yaml",
    )
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--expect-rewritten", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = audit(args.root, args.plan, expect_rewritten=args.expect_rewritten)
    print(json.dumps(printable_summary(result), indent=2))
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    for error in result["errors"]:
        print(f"ERROR: {error}")

    if args.report:
        report = printable_summary(result) | {
            "errors": result["errors"],
            "warnings": result["warnings"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(args.report, report)

    if not result["ok"]:
        return 1
    if args.rewrite:
        manifest = rewrite_instructions(args.root, result)
        print(json.dumps({"rewrite_counts": manifest["counts"], "changed_files": manifest["changed_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
