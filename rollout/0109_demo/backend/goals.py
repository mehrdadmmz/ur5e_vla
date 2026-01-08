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
#   - top(block) - block is at the top of a stack (nothing above it)

GOAL_LIBRARY: Dict[str, Dict] = {
    # Bridge: two pillars with blocks stacked, then plank on top
    'bridge': {
        'description': 'Build a bridge with a plank on two pillars',
        'predicates': [
            ('on-table', '1', '9'), ('on-table', '2', '9'),
            ('beside', '2', '1'),
            ('above', '0', '1'), ('above', '3', '2'),
            ('above_both', '6', '0', '3'),
            ('top', '6'),  # Plank 6 is on top
        ]
    },
    # Tall bridge: double-height pillars
    'tall_bridge': {
        'description': 'Build a tall bridge with double-height pillars',
        'predicates': [
            ('on-table', '1', '9'), ('on-table', '2', '9'),
            ('beside', '2', '1'),
            ('above', '0', '1'), ('above', '3', '2'),
            ('above', '7', '0'), ('above', '5', '3'),
            ('above_both', '6', '5', '7'),
            ('top', '6'),  # Plank 6 is on top
        ]
    },
    # Simple tower: stack of blocks
    'tower': {
        'description': 'Build a 4-block tower',
        'predicates': [
            ('on-table', '0', '9'),
            ('above', '1', '0'),
            ('above', '2', '1'),
            ('above', '3', '2'),
            ('top', '3'),  # Block 3 is on top
        ]
    },
    # Totem pole: tall stack
    'totem_pole': {
        'description': 'Build a totem pole (5-block tower)',
        'predicates': [
            ('on-table', '7', '9'),
            ('above', '1', '7'), ('above', '3', '1'),
            ('above', '6', '3'), ('above', '0', '6'),
            ('top', '0'),  # Block 0 is on top
        ]
    },
    # House shape
    'house': {
        'description': 'Build a house structure',
        'predicates': [
            ('on-table', '3', '9'), ('on-table', '7', '9'),
            ('beside', '7', '3'),
            ('above', '2', '3'), ('above', '1', '7'),
            ('above', '5', '1'), ('above', '0', '2'),
            ('above_both', '6', '0', '5'),
            ('top', '6'),  # Plank 6 is on top
        ]
    },
    # Chinese character: 土 (tu - earth/soil)
    'tu': {
        'description': 'Build Chinese character 土 (tǔ - earth/soil)',
        'predicates': [
            ('on-table', '19', '9'),
            ('above', '7', '19'), ('above', '4', '7'),
            ('above', '5', '4'),
            ('top', '5'),  # Block 5 is on top
        ]
    },
    # Chinese character: 王 (wang - king)
    'wang': {
        'description': 'Build Chinese character 王 (wáng - king)',
        'predicates': [
            ('on-table', '19', '9'),
            ('above', '7', '19'), ('above', '4', '7'),
            ('above', '5', '4'), ('above', '6', '5'),
            ('top', '6'),  # Block 19 is on top
        ]
    },
    # Chinese character: 干 (gan - dry/stem)
    'gan': {
        'description': 'Build Chinese character 干 (gān - dry/stem)',
        'predicates': [
            ('on-table', '7', '9'),
            ('above', '4', '7'), ('above', '3', '4'),
            ('above', '6', '3'),
            ('top', '6'),  # Plank 6 is on top
        ]
    },
    # Chinese character: 十 (shi - ten)
    'shi': {
        'description': 'Build Chinese character 十 (shí - ten)',
        'predicates': [
            ('on-table', '7', '9'),
            ('above', '4', '7'), ('above', '3', '4'),
            ('top', '3'),  # Block 3 is on top - distinguishes from gan!
        ]
    },
    # Boat shape
    'boat': {
        'description': 'Build a boat structure',
        'predicates': [
            ('on-table', '6', '9'),
            ('above', '4', '6'), ('above', '3', '4'),
            ('top', '3'),  # Block 3 is on top
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
