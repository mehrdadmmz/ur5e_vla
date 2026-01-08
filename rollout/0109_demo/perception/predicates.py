"""
Predicate Utilities for Block World Planning.

Converts geometric object observations into symbolic predicates for PDDL planning.

Object format: [id, class, x, y, z, width, length, height]
    class 0 = table
    class 1 = robot gripper
    class 2 = cube block
    class 3 = plank/surface (for bridges)
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np


# =============================================================================
# Default Predicate Thresholds
# =============================================================================

DEFAULT_THRESHOLDS = {
    'beside': {
        'dist_min': 0.8,       # Min horizontal distance as fraction of expected (allows slight overlap)
        'dist_max': 1.5,       # Max horizontal distance as fraction of expected (allows gap)
        'z_tolerance': 2.0,    # Z alignment tolerance (multiplier of min height)
    },
    'above': {
        'z_diff_min': 0.5,     # Min Z difference as fraction of top block height (loosened from 0.75)
        'z_diff_max': 1.5,     # Max Z difference as fraction of top block height (loosened from 1.2)
        'xy_tolerance': 1.0,   # XY overlap tolerance (multiplier of max dimension) (loosened from 0.7)
    },
    'on_table': {
        'z_threshold': 1.4,    # Max Z above table as fraction of block height
    },
    'alignment': {
        'yaw_tolerance': 0.15,  # Tolerance in radians (~8.6 degrees) for aligned detection
    },
}

# Global thresholds (can be set by set_thresholds)
_thresholds = DEFAULT_THRESHOLDS.copy()


def set_thresholds(thresholds: Dict):
    """Set predicate detection thresholds from config."""
    global _thresholds
    for key in ['beside', 'above', 'on_table', 'alignment']:
        if key in thresholds:
            if key not in _thresholds:
                _thresholds[key] = {}
            _thresholds[key].update(thresholds[key])


# =============================================================================
# Alignment Utilities
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_long_axis_offset(width: float, length: float) -> float:
    """
    Get the angular offset of a block's long axis from its X-axis.

    Args:
        width: Block width (X dimension)
        length: Block length (Y dimension)

    Returns:
        0.0 if X is longer (long axis along X), π/2 if Y is longer (long axis along Y)
    """
    if length > width:
        return np.pi / 2
    return 0.0


def get_valid_alignments(class1: int, class2: int) -> List[float]:
    """
    Get valid relative yaw angles for alignment between block types.

    Args:
        class1: Block class (2=cube, 3=plank)
        class2: Block class (2=cube, 3=plank)

    Returns:
        List of valid relative angles in radians.
    """
    if class1 == 2 and class2 == 2:
        # Cube + Cube: 4 symmetric orientations (90-degree increments)
        return [0.0, np.pi/2, np.pi, -np.pi/2]
    elif class1 == 3 and class2 == 3:
        # Plank + Plank: 2 orientations (parallel long axes)
        # Note: actual check now uses long axis comparison, not raw yaw
        return [0.0, np.pi]
    else:
        # Cube + Plank (either order): 4 orientations
        return [0.0, np.pi/2, np.pi, -np.pi/2]


def is_aligned(yaw1: float, yaw2: float, class1: int, class2: int,
               yaw_tolerance: float = 0.15,
               dims1: Tuple[float, float] = None,
               dims2: Tuple[float, float] = None) -> bool:
    """
    Check if two blocks are aligned in orientation.

    For planks (class 3), this checks if long axes are parallel, not raw yaw angles.
    This handles planks with different aspect ratios (e.g., one with X long, one with Y long).

    Args:
        yaw1: Yaw angle of block 1 (radians)
        yaw2: Yaw angle of block 2 (radians)
        class1: Block class (2=cube, 3=plank)
        class2: Block class (2=cube, 3=plank)
        yaw_tolerance: Tolerance in radians (default ~8.6 degrees)
        dims1: Optional (width, length) of block 1 for plank long-axis calculation
        dims2: Optional (width, length) of block 2 for plank long-axis calculation

    Returns:
        True if blocks are aligned within tolerance
    """
    # For plank + plank, compare long axis angles (not raw yaw)
    if class1 == 3 and class2 == 3 and dims1 is not None and dims2 is not None:
        # Compute long axis angle for each plank
        # Long axis angle = yaw + offset (0 if X is long, π/2 if Y is long)
        long_axis1 = yaw1 + get_long_axis_offset(dims1[0], dims1[1])
        long_axis2 = yaw2 + get_long_axis_offset(dims2[0], dims2[1])

        # Long axes are parallel if they differ by 0° or 180°
        relative_angle = normalize_angle(long_axis1 - long_axis2)
        if abs(relative_angle) <= yaw_tolerance or abs(abs(relative_angle) - np.pi) <= yaw_tolerance:
            return True
        return False

    # For other combinations, use raw yaw comparison
    relative_yaw = normalize_angle(yaw1 - yaw2)

    # Get valid alignment angles based on block type combination
    valid_angles = get_valid_alignments(class1, class2)

    # Check if relative_yaw matches any valid angle within tolerance
    for angle in valid_angles:
        if abs(normalize_angle(relative_yaw - angle)) <= yaw_tolerance:
            return True
    return False


# =============================================================================
# Geometric Predicate Detection
# =============================================================================

def get_beside(obj1: List, obj2: List, table_z: float = None) -> Optional[List[Tuple]]:
    """
    Check if obj1 is beside obj2 (horizontally adjacent).

    Only applies to cube blocks (class 2).
    Both blocks must be on the table (not stacked) for beside to apply.
    Blocks in an above/below relationship cannot be beside each other.
    Returns BOTH ('beside', id1, id2) AND ('beside', id2, id1) since beside is symmetric.
    Only returns predicates when obj1's id < obj2's id to avoid duplicates.
    """
    if obj1[1] != 2 or obj2[1] != 2:
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1, *_ = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2, *_ = obj2

    # Only process each pair once (when id1 < id2)
    if id1 >= id2:
        return None

    # Get thresholds
    t = _thresholds['beside']
    t_on_table = _thresholds['on_table']

    # Check both blocks are on the table (not stacked on other blocks)
    if table_z is not None:
        z_diff1 = z1 - table_z
        z_diff2 = z2 - table_z
        # Both blocks must be within on_table threshold of table surface
        if z_diff1 >= h1 * t_on_table['z_threshold'] or z_diff2 >= h2 * t_on_table['z_threshold']:
            return None

    # Note: We no longer exclude beside based on above z_diff_min here.
    # The filter_beside_by_above() function handles removing beside predicates
    # when an actual 'above' relationship is detected, which is more accurate.
    z_diff = abs(z1 - z2)
    min_height = min(h1, h2)

    # Check: Horizontal distance is approximately the sum of half-lengths (blocks touching)
    # This is direction-agnostic - works for any orientation
    horizontal_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    expected_dist = (l1 + l2) / 2  # center-to-center distance when blocks are touching

    if (expected_dist * t['dist_min'] < horizontal_dist < expected_dist * t['dist_max'] and
        z_diff < min_height * t['z_tolerance']):

        # Check yaw alignment (blocks must be oriented similarly)
        yaw1 = obj1[8] if len(obj1) > 8 else 0.0
        yaw2 = obj2[8] if len(obj2) > 8 else 0.0
        yaw_tolerance = _thresholds.get('alignment', {}).get('yaw_tolerance', 0.15)

        if not is_aligned(yaw1, yaw2, cls1, cls2, yaw_tolerance, (w1, l1), (w2, l2)):
            return None  # Geometrically beside but not aligned in orientation

        # Return both directions since beside is symmetric
        return [('beside', str(id1), str(id2)), ('beside', str(id2), str(id1))]

    return None


def get_above(obj1: List, obj2: List, debug: bool = False) -> Optional[Tuple]:
    """
    Check if obj1 is directly above obj2 (stacked).

    Applies to cubes (class 2) and planks (class 3).
    Returns ('above', id1, id2) if obj1 is on top of obj2.
    """
    if (obj1[1] not in [2, 3]) or (obj2[1] not in [2, 3]):
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1, *_ = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2, *_ = obj2

    # Get thresholds
    t = _thresholds['above']

    # Check: Z difference is about one block height, XY overlap
    z_diff = z1 - z2

    # For planks (class 3), use the longer dimension for XY tolerance
    # This handles planks spanning multiple supports
    if cls1 == 3:  # Top object is a plank
        # Plank can span - use max of width/length for tolerance
        max_dim1 = max(w1, l1)
        xy_tol_x = max(max_dim1, w2) * t['xy_tolerance']
        xy_tol_y = max(max_dim1, l2) * t['xy_tolerance']
    else:
        xy_tol_x = max(w1, w2) * t['xy_tolerance']
        xy_tol_y = max(l1, l2) * t['xy_tolerance']

    # Debug logging
    if debug and z_diff > 0:
        z_min = h1 * t['z_diff_min']
        z_max = h1 * t['z_diff_max']
        print(f"  [above check] {id1} over {id2}:")
        print(f"    z_diff={z_diff:.4f} (need {z_min:.4f} <= z < {z_max:.4f})")
        print(f"    y_diff={abs(y1-y2):.4f} (need < {xy_tol_y:.4f})")
        print(f"    x_diff={abs(x1-x2):.4f} (need < {xy_tol_x:.4f})")

    if (h1 * t['z_diff_min'] <= z_diff < h1 * t['z_diff_max'] and
        abs(y1 - y2) < xy_tol_y and
        abs(x1 - x2) < xy_tol_x):

        # Check yaw alignment (stacked blocks must be oriented similarly)
        # For planks, this checks long axis alignment (handles different aspect ratios)
        yaw1 = obj1[8] if len(obj1) > 8 else 0.0
        yaw2 = obj2[8] if len(obj2) > 8 else 0.0
        yaw_tolerance = _thresholds.get('alignment', {}).get('yaw_tolerance', 0.15)

        if not is_aligned(yaw1, yaw2, cls1, cls2, yaw_tolerance, (w1, l1), (w2, l2)):
            return None  # Geometrically above but not aligned in orientation

        return ('above', str(id1), str(id2))

    return None


def print_block_positions(observations: List[List], compact: bool = True):
    """Print all block positions in robot frame for debugging."""
    if compact:
        # Single-line format: ID0:[x,y,z] ID1:[x,y,z] ...
        blocks = []
        for obs in observations:
            if obs[1] in [2, 3]:  # cube, plank only
                blocks.append(f"{obs[0]}:[{obs[2]:.3f},{obs[3]:.3f},{obs[4]:.3f}]")
        if blocks:
            print(f"Blocks: {' '.join(blocks)}")
    else:
        print("\n=== Block Positions (Robot Frame) ===")
        print(f"{'ID':>3} {'Class':>5} {'X':>8} {'Y':>8} {'Z':>8} {'W':>6} {'L':>6} {'H':>6}")
        print("-" * 60)
        for obs in observations:
            if obs[1] in [0, 2, 3]:  # Table, cube, plank
                obj_id = obs[0]
                cls = obs[1]
                x, y, z = obs[2], obs[3], obs[4]
                if len(obs) > 7:
                    w, l, h = obs[5], obs[6], obs[7]
                    print(f"{obj_id:>3} {cls:>5} {x:>8.4f} {y:>8.4f} {z:>8.4f} {w:>6.3f} {l:>6.3f} {h:>6.3f}")
                else:
                    print(f"{obj_id:>3} {cls:>5} {x:>8.4f} {y:>8.4f} {z:>8.4f}")
        print("=" * 60 + "\n")


def get_on_table(obj1: List, obj2: List) -> Optional[Tuple]:
    """
    Check if obj1 is on the table (obj2).

    obj1 must be a block (class 2 or 3), obj2 must be table (class 0).
    """
    if (obj1[1] not in [2, 3]) or obj2[1] != 0:
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1, *_ = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2, *_ = obj2

    # Get thresholds
    t = _thresholds['on_table']

    # Check: Z is close to table surface + block height
    z_diff = z1 - z2
    if z_diff < h1 * t['z_threshold']:
        return ('on-table', str(id1), str(id2))

    return None


def get_holding(obj1: List, obj2: List) -> Optional[Tuple]:
    """
    Check if robot (obj2) is holding block (obj1).

    obj1 must be a block, obj2 must be robot (class 1).
    Robot format: [id, class, x, y, z, gripper_open]
    """
    if (obj1[1] not in [2, 3]) or obj2[1] != 1:
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1, *_ = obj1
    id2, cls2, x2, y2, z2, gripper_open, *_ = obj2

    # If gripper is open, hand is free
    if gripper_open == 1:
        return ('hand_free', str(id2))

    # Check if block is near gripper position
    if (abs(z1 - z2) < h1 / 2 and
        abs(y1 - y2) < l1 / 2 and
        abs(x1 - x2) < w1 / 2 and
        gripper_open == 0):
        return ('holding', str(id1), str(id2))

    return None


def get_class(obj: List) -> Optional[Tuple]:
    """Get the class predicate for an object."""
    cls = obj[1]
    obj_id = str(obj[0])

    if cls == 0:
        return ('table', obj_id)
    elif cls == 1:
        return ('robot', obj_id)
    elif cls == 2:
        return ('box', obj_id)
    elif cls == 3:
        return ('box', obj_id)  # Planks are also 'box' in PDDL

    return None


# =============================================================================
# Derived Predicates
# =============================================================================

def get_top_predicates(predicates: List[Tuple]) -> List[Tuple]:
    """
    Derive 'top' predicates for blocks that have nothing above them.
    """
    below_set = set()  # Objects that have something above them
    all_boxes = set()

    for pred in predicates:
        if pred[0] == 'above':
            below_set.add(pred[2])  # The bottom object
        if pred[0] == 'box':
            all_boxes.add(pred[1])

    tops = []
    for obj_id in all_boxes:
        if obj_id not in below_set:
            tops.append(('top', obj_id))

    return tops


def get_nothing_beside_predicates(predicates: List[Tuple]) -> List[Tuple]:
    """
    Derive 'nothing_beside' predicates for blocks with no neighbor.
    """
    has_neighbor = set()
    all_boxes = set()

    for pred in predicates:
        if pred[0] == 'beside':
            # Both objects in a beside relation have neighbors
            has_neighbor.add(pred[1])
            has_neighbor.add(pred[2])
        if pred[0] == 'box':
            all_boxes.add(pred[1])

    alone = []
    for obj_id in all_boxes:
        if obj_id not in has_neighbor:
            alone.append(('nothing_beside', obj_id))

    return alone


def filter_beside_by_above(predicates: List[Tuple]) -> List[Tuple]:
    """
    Remove 'beside' predicates for pairs that have an 'above' relationship.

    If above(A, B) or above(B, A) exists, then BOTH beside(A, B) and beside(B, A) must be removed.
    """
    # Collect all pairs with above relationship (as unordered pairs)
    above_pairs = set()
    for pred in predicates:
        if pred[0] == 'above':
            a, b = pred[1], pred[2]
            # Store as frozenset so order doesn't matter
            above_pairs.add(frozenset([a, b]))

    # Filter out beside predicates for those pairs
    filtered = []
    for pred in predicates:
        if pred[0] == 'beside':
            a, b = pred[1], pred[2]
            pair = frozenset([a, b])
            if pair in above_pairs:
                continue  # Skip - these objects have above relationship
        filtered.append(pred)

    return filtered


def filter_above_by_stack(predicates: List[Tuple]) -> List[Tuple]:
    """
    Filter 'above' predicates to only keep direct (adjacent) stacking.

    If A is above B and B is above C, we should NOT have A above C.
    """
    below_to_above = {}
    all_objects = set()

    for pred in predicates:
        if pred[0] == 'above':
            above, below = pred[1], pred[2]
            below_to_above[below] = above
            all_objects.update([above, below])
        elif pred[0] == 'on-table':
            all_objects.add(pred[1])

    # Build stacks from bottom to top
    base_objects = all_objects - set(below_to_above.values())
    valid_above = set()

    for base in base_objects:
        current = base
        while current in below_to_above:
            top = below_to_above[current]
            valid_above.add(('above', top, current))
            current = top

    # Filter predicates
    filtered = []
    for pred in predicates:
        if pred[0] == 'above':
            if pred in valid_above:
                filtered.append(pred)
        else:
            filtered.append(pred)

    return filtered


def clean_table_predicates(predicates: List[Tuple]) -> List[Tuple]:
    """
    Remove 'on-table' for objects that are above other objects.
    """
    above_objects = set()
    for pred in predicates:
        if pred[0] == 'above':
            above_objects.add(pred[1])

    cleaned = []
    for pred in predicates:
        if pred[0] == 'on-table' and pred[1] in above_objects:
            continue  # Skip - this object is stacked, not on table
        cleaned.append(pred)

    return cleaned


def filter_predicates_for_held_blocks(predicates: List[Tuple]) -> List[Tuple]:
    """
    Remove position-based predicates for blocks that are currently being held.

    When a block is held, its ArUco marker is occluded by the gripper, so
    perception uses stale cached position data. This causes incorrect
    position predicates (on-table, beside) to be generated.

    Args:
        predicates: Current predicates list

    Returns:
        Filtered predicates with position predicates removed for held blocks
    """
    # Collect held block IDs
    held_ids = set()
    for pred in predicates:
        if pred[0] == 'holding':
            held_ids.add(pred[1])

    if not held_ids:
        return predicates

    # Filter out position predicates for held blocks
    filtered = []
    for pred in predicates:
        if pred[0] == 'on-table' and pred[1] in held_ids:
            continue  # Skip on-table for held blocks
        if pred[0] == 'beside':
            if pred[1] in held_ids or pred[2] in held_ids:
                continue  # Skip beside if either block is held
        if pred[0] == 'above' and pred[1] in held_ids:
            continue  # Skip above for held blocks (held block can't be on top of anything)
        filtered.append(pred)

    return filtered


def get_floating_predicates(predicates: List[Tuple]) -> List[Tuple]:
    """
    Detect floating blocks - blocks that are neither on-table nor above anything.

    These are likely perception errors that need handling. The planner can use
    pick_floating to recover by placing them back on the table.

    Args:
        predicates: Current predicates list

    Returns:
        List of ('floating', block_id) tuples for floating blocks
    """
    floating = []

    # Get all box IDs
    box_ids = set()
    for pred in predicates:
        if pred[0] == 'box':
            box_ids.add(pred[1])

    # Get blocks that are on-table
    on_table_ids = set()
    for pred in predicates:
        if pred[0] == 'on-table':
            on_table_ids.add(pred[1])

    # Get blocks that are above something
    above_ids = set()
    for pred in predicates:
        if pred[0] == 'above':
            above_ids.add(pred[1])

    # Get blocks that are being held
    held_ids = set()
    for pred in predicates:
        if pred[0] == 'holding':
            held_ids.add(pred[1])

    # Floating = box but not on-table and not above anything and not held
    for box_id in box_ids:
        if box_id not in on_table_ids and box_id not in above_ids and box_id not in held_ids:
            floating.append(('floating', box_id))

    return floating


def get_above_both(predicates: List[Tuple], observations: List[List]) -> List[Tuple]:
    """
    Convert double 'above' relations to 'above_both' for bridge planks.

    Only applies to class 3 (plank) objects that span two supports.
    """
    above_map = defaultdict(list)
    other_preds = []

    for pred in predicates:
        if pred[0] == 'above':
            above_map[pred[1]].append(pred[2])
        else:
            other_preds.append(pred)

    result = other_preds.copy()

    for top_obj, under_objs in above_map.items():
        # Check if this is a plank (class 3)
        obj_class = None
        for obs in observations:
            if str(obs[0]) == top_obj:
                obj_class = obs[1]
                break

        if obj_class == 3 and len(under_objs) == 2:
            # This is a bridge plank spanning two supports
            # Add both orderings: above_both(A, B, C) and above_both(A, C, B)
            result.append(('above_both', top_obj, under_objs[0], under_objs[1]))
            result.append(('above_both', top_obj, under_objs[1], under_objs[0]))
            # Also keep individual above relations
            for under_obj in under_objs:
                result.append(('above', top_obj, under_obj))
        else:
            # Regular stacking
            for under_obj in under_objs:
                result.append(('above', top_obj, under_obj))

    return result


# =============================================================================
# Main Function
# =============================================================================

def get_logical_state(observations: List[List], debug: bool = False) -> List[Tuple]:
    """
    Convert geometric observations to symbolic predicates.

    Args:
        observations: List of objects, each as [id, class, x, y, z, w, l, h]
                     Robot is [id, class, x, y, z, gripper_open]
        debug: If True, print debug info for above detection

    Returns:
        List of predicates like [('box', '0'), ('on-table', '0', '9'), ...]
    """
    predicates = []

    # Find table_z from table object (class 0)
    table_z = None
    for obj in observations:
        if obj[1] == 0:  # Table class
            table_z = obj[4]  # z coordinate
            break

    # Debug: print block positions
    if debug:
        print_block_positions(observations)
        print("[DEBUG] Checking 'above' predicates:")

    # Get class predicates and pairwise relations
    for obj1 in observations:
        # Class predicate
        cls_pred = get_class(obj1)
        if cls_pred and cls_pred not in predicates:
            predicates.append(cls_pred)

        # Pairwise relations
        for obj2 in observations:
            if obj1[0] == obj2[0]:
                continue

            # Check above with debug
            result = get_above(obj1, obj2, debug=debug)
            if result and result not in predicates:
                predicates.append(result)

            # Check beside (requires table_z to verify both blocks are on table)
            result = get_beside(obj1, obj2, table_z=table_z)
            if result:
                for pred in result:
                    if pred not in predicates:
                        predicates.append(pred)

            # Check other predicates
            for get_fn in [get_on_table, get_holding]:
                result = get_fn(obj1, obj2)
                if result:
                    # Handle both single predicates and lists of predicates
                    if isinstance(result, list):
                        for pred in result:
                            if pred not in predicates:
                                predicates.append(pred)
                    elif result not in predicates:
                        predicates.append(result)

    # Ensure hand_free or holding is set for robot
    # If gripper is closed but no holding detected, assume hand_free (anomalous but allows operation)
    robot_obs = None
    for obj in observations:
        if obj[1] == 1:  # Robot class
            robot_obs = obj
            break

    if robot_obs is not None:
        robot_id = str(robot_obs[0])
        has_hand_free = any(p[0] == 'hand_free' and p[1] == robot_id for p in predicates)
        has_holding = any(p[0] == 'holding' and p[2] == robot_id for p in predicates)

        if not has_hand_free and not has_holding:
            # Gripper closed but not holding anything - add hand_free to allow operation
            predicates.append(('hand_free', robot_id))

    # Derived predicates
    predicates.extend(get_top_predicates(predicates))
    predicates.extend(get_nothing_beside_predicates(predicates))

    # Clean up
    predicates = clean_table_predicates(predicates)
    predicates = filter_above_by_stack(predicates)
    predicates = filter_beside_by_above(predicates)  # Remove beside if above exists for same pair
    predicates = get_above_both(predicates, observations)

    # Filter out stale position predicates for held blocks
    predicates = filter_predicates_for_held_blocks(predicates)

    # Detect floating blocks (perception error recovery)
    predicates.extend(get_floating_predicates(predicates))

    return predicates


# =============================================================================
# Goal Checking
# =============================================================================

def is_goal_satisfied(goal: List[Tuple], state: List[Tuple]) -> bool:
    """Check if all goal predicates are satisfied in current state."""
    for g in goal:
        if g not in state:
            return False
    return True


def get_unsatisfied_goals(goal: List[Tuple], state: List[Tuple]) -> List[Tuple]:
    """Get list of goal predicates not yet satisfied."""
    return [g for g in goal if g not in state]


# =============================================================================
# Action Effect Application (for symbolic state updates)
# =============================================================================

def apply_action_effects(state: List[Tuple], action_name: str, args: Tuple) -> List[Tuple]:
    """
    Apply PDDL action effects to update logical state symbolically.

    Used after pick-up/unstack actions when perception is blocked.

    Args:
        state: Current logical state as list of predicates
        action_name: Name of the action executed
        args: Action arguments as tuple of string IDs

    Returns:
        Updated logical state with action effects applied
    """
    new_state = list(state)  # Copy state

    def add_pred(pred):
        if pred not in new_state:
            new_state.append(pred)

    def remove_pred(pred):
        if pred in new_state:
            new_state.remove(pred)

    if action_name == 'pick-up':
        # pick-up(?b1, ?t1, ?r1)
        b1, t1, r1 = args[0], args[1], args[2]
        # Add effects
        add_pred(('holding', b1, r1))
        # Remove effects
        remove_pred(('hand_free', r1))
        remove_pred(('top', b1))
        remove_pred(('on-table', b1, t1))

    elif action_name == 'unstack':
        # unstack(?b1, ?b2, ?r1)
        b1, b2, r1 = args[0], args[1], args[2]
        # Add effects
        add_pred(('holding', b1, r1))
        add_pred(('top', b2))
        # Remove effects
        remove_pred(('hand_free', r1))
        remove_pred(('top', b1))
        remove_pred(('above', b1, b2))

    elif action_name == 'remove-beside':
        # remove-beside(?b1, ?b2, ?t1, ?r1)
        b1, b2, t1, r1 = args[0], args[1], args[2], args[3]
        # Add effects
        add_pred(('holding', b1, r1))
        add_pred(('nothing_beside', b2))
        # Remove effects
        remove_pred(('hand_free', r1))
        remove_pred(('top', b1))
        remove_pred(('on-table', b1, t1))
        remove_pred(('beside', b1, b2))
        remove_pred(('beside', b2, b1))

    elif action_name == 'pick_floating':
        # pick_floating(?b1, ?r1)
        b1, r1 = args[0], args[1]
        # Add effects
        add_pred(('holding', b1, r1))
        # Remove effects
        remove_pred(('hand_free', r1))
        remove_pred(('top', b1))
        remove_pred(('floating', b1))

    # Note: place actions (align, put-down, cover, release) don't need
    # symbolic updates since we re-observe after them

    return new_state
