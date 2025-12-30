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


# =============================================================================
# Default Predicate Thresholds
# =============================================================================

DEFAULT_THRESHOLDS = {
    'beside': {
        'y_sep_min': 0.5,      # Min Y separation as fraction of combined length
        'y_sep_max': 1.25,     # Max Y separation as fraction of combined length
        'x_tolerance': 1.2,    # X alignment tolerance (multiplier of min width)
        'z_tolerance': 1.2,    # Z alignment tolerance (multiplier of min height)
    },
    'above': {
        'z_diff_min': 0.75,    # Min Z difference as fraction of top block height
        'z_diff_max': 1.2,     # Max Z difference as fraction of top block height
        'xy_tolerance': 0.7,   # XY overlap tolerance (multiplier of max dimension)
    },
    'on_table': {
        'z_threshold': 1.4,    # Max Z above table as fraction of block height
    },
}

# Global thresholds (can be set by set_thresholds)
_thresholds = DEFAULT_THRESHOLDS.copy()


def set_thresholds(thresholds: Dict):
    """Set predicate detection thresholds from config."""
    global _thresholds
    for key in ['beside', 'above', 'on_table']:
        if key in thresholds:
            _thresholds[key].update(thresholds[key])


# =============================================================================
# Geometric Predicate Detection
# =============================================================================

def get_beside(obj1: List, obj2: List) -> Optional[Tuple]:
    """
    Check if obj1 is beside obj2 (horizontally adjacent).

    Only applies to cube blocks (class 2).
    Returns ('beside', id2, id1) if obj1 is in front of obj2 in Y direction.
    """
    if obj1[1] != 2 or obj2[1] != 2:
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1 = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2 = obj2

    # Get thresholds
    t = _thresholds['beside']

    # Check: Y separation is about one block width, X aligned, same height
    y_sep = y1 - y2
    if ((l1 + l2) * t['y_sep_min'] < y_sep < t['y_sep_max'] * (l1 + l2) and
        abs(x1 - x2) < min(w1, w2) * t['x_tolerance'] and
        abs(z1 - z2) < min(h1, h2) * t['z_tolerance']):
        return ('beside', str(id2), str(id1))

    return None


def get_above(obj1: List, obj2: List) -> Optional[Tuple]:
    """
    Check if obj1 is directly above obj2 (stacked).

    Applies to cubes (class 2) and planks (class 3).
    Returns ('above', id1, id2) if obj1 is on top of obj2.
    """
    if (obj1[1] not in [2, 3]) or (obj2[1] not in [2, 3]):
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1 = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2 = obj2

    # Get thresholds
    t = _thresholds['above']

    # Check: Z difference is about one block height, XY overlap
    z_diff = z1 - z2
    if (h1 * t['z_diff_min'] <= z_diff < h1 * t['z_diff_max'] and
        abs(y1 - y2) < max(l1, l2) * t['xy_tolerance'] and
        abs(x1 - x2) < max(w1, w2) * t['xy_tolerance']):
        return ('above', str(id1), str(id2))

    return None


def get_on_table(obj1: List, obj2: List) -> Optional[Tuple]:
    """
    Check if obj1 is on the table (obj2).

    obj1 must be a block (class 2 or 3), obj2 must be table (class 0).
    """
    if (obj1[1] not in [2, 3]) or obj2[1] != 0:
        return None

    id1, cls1, x1, y1, z1, w1, l1, h1 = obj1
    id2, cls2, x2, y2, z2, w2, l2, h2 = obj2

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

    id1, cls1, x1, y1, z1, w1, l1, h1 = obj1
    id2, cls2, x2, y2, z2, gripper_open = obj2

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
            has_neighbor.add(pred[2])  # The object that has something beside it
        if pred[0] == 'box':
            all_boxes.add(pred[1])

    alone = []
    for obj_id in all_boxes:
        if obj_id not in has_neighbor:
            alone.append(('nothing_beside', obj_id))

    return alone


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
            sorted_under = sorted(under_objs[:2], key=int)
            result.append(('above_both', top_obj, *sorted_under))
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

def get_logical_state(observations: List[List]) -> List[Tuple]:
    """
    Convert geometric observations to symbolic predicates.

    Args:
        observations: List of objects, each as [id, class, x, y, z, w, l, h]
                     Robot is [id, class, x, y, z, gripper_open]

    Returns:
        List of predicates like [('box', '0'), ('on-table', '0', '9'), ...]
    """
    predicates = []

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

            for get_fn in [get_above, get_beside, get_on_table, get_holding]:
                pred = get_fn(obj1, obj2)
                if pred and pred not in predicates:
                    predicates.append(pred)

    # Derived predicates
    predicates.extend(get_top_predicates(predicates))
    predicates.extend(get_nothing_beside_predicates(predicates))

    # Clean up
    predicates = clean_table_predicates(predicates)
    predicates = filter_above_by_stack(predicates)
    predicates = get_above_both(predicates, observations)

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
