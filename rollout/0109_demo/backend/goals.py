"""
Goal Library - Single source of truth for available goals.

All goal definitions should be imported from this file.
"""
from typing import Dict, List, Tuple, Optional


# Goal predicates use the format: (predicate_name, arg1, arg2, ...)
# Available predicates:
#   - on-table(block, table) - block is on the table
#   - above(block1, block2) - block1 is directly above block2
#   - beside(block1, block2) - block1 is beside block2
#   - above_both(block, block1, block2) - block spans above both block1 and block2

GOAL_LIBRARY: Dict[str, Dict] = {
    'tower': {
        'description': 'Build a 3-block tower',
        'predicates': [
            ('above', '1', '2'),
            ('above', '0', '1'),
        ]
    },
    'bridge': {
        'description': 'Build a bridge with a plank on two pillars',
        'predicates': [
            ('on-table', '3', '9'), ('on-table', '7', '9'),
            ('beside', '7', '3'),
            ('above_both', '6', '3', '7'),
        ]
    },
    'two_towers': {
        'description': 'Build two separate towers',
        'predicates': [
            ('above', '1', '2'),
            ('above', '0', '1'),
            ('above', '5', '3'),
        ]
    },
    'pyramid': {
        'description': 'Build a pyramid structure',
        'predicates': [
            ('on-table', '2', '9'), ('on-table', '1', '9'),
            ('beside', '1', '2'),
            ('above_both', '0', '2', '1'),
        ]
    },
    'double_bridge': {
        'description': 'Build a tall double bridge',
        'predicates': [
            ('on-table', '3', '9'), ('on-table', '7', '9'),
            ('beside', '7', '3'),
            ('above', '2', '3'), ('above', '1', '7'),
            ('above', '5', '1'), ('above', '0', '2'),
            ('above_both', '6', '0', '5'),
        ]
    },
}


def get_goal_predicates(goal_name: str) -> Optional[List[Tuple]]:
    """Get predicates for a goal name.

    Args:
        goal_name: Name of the goal

    Returns:
        List of predicate tuples, or None if goal not found
    """
    if goal_name not in GOAL_LIBRARY:
        return None
    return GOAL_LIBRARY[goal_name]['predicates']


def get_goal_description(goal_name: str) -> Optional[str]:
    """Get description for a goal name.

    Args:
        goal_name: Name of the goal

    Returns:
        Description string, or None if goal not found
    """
    if goal_name not in GOAL_LIBRARY:
        return None
    return GOAL_LIBRARY[goal_name]['description']


def get_available_goals() -> Dict[str, Dict]:
    """Get all available goals.

    Returns:
        Dictionary of goal_name -> {description, predicates}
    """
    return GOAL_LIBRARY


def get_goal_names() -> List[str]:
    """Get list of available goal names.

    Returns:
        List of goal names
    """
    return list(GOAL_LIBRARY.keys())
