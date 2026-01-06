"""
PDDL Solver for Block World Planning.

Simplified interface to pddlstream library for task planning.
"""

import os
import sys
import re
from typing import List, Tuple, Optional

# Add pddlstream to path
current_dir = os.path.dirname(os.path.abspath(__file__))
pddlstream_path = os.path.join(current_dir, 'pddlstream')
sys.path.insert(0, pddlstream_path)

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.utils import read
from pddlstream.language.constants import PDDLProblem, And


# =============================================================================
# PDDL File Utilities
# =============================================================================

def read_pddl_file(filename: str) -> str:
    """Read PDDL file content."""
    if not os.path.isabs(filename):
        filename = os.path.join(current_dir, filename)
    return read(filename)


def modify_pddl_for_negative_goals(pddl_content: str) -> str:
    """
    Modify PDDL domain to support negative goal tracking.

    Adds 'not-X' predicates for each negated effect in actions.
    """
    negated_pattern = re.compile(r'\(not \(([^\)]+)\)\)')

    lines = pddl_content.split("\n")
    modified_lines = []
    all_negated = set()

    for line in lines:
        matches = negated_pattern.findall(line)
        if matches:
            all_negated.update(matches)
            for match in matches:
                negated_pred = match.strip()
                new_pred = f"(not-{negated_pred})"
                line = line.replace(
                    f"(not ({negated_pred}))",
                    f"(not ({negated_pred})) {new_pred}"
                )
        modified_lines.append(line)

    # Insert new predicates in :predicates section
    predicates_found = False
    for i, line in enumerate(modified_lines):
        if ":predicates" in line:
            predicates_found = True
        elif predicates_found and line.strip() == ")":
            new_preds = "\n        " + "\n        ".join(
                [f"(not-{p.strip()})" for p in sorted(all_negated)]
            )
            modified_lines[i] = new_preds + "\n" + modified_lines[i]
            break

    return "\n".join(modified_lines)


def process_domain_file(input_file: str, use_negative: bool = False) -> str:
    """
    Process PDDL domain file, optionally adding negative goal support.

    Returns path to the (possibly modified) domain file.
    """
    if not use_negative:
        return input_file

    # Read and modify
    with open(input_file, 'r') as f:
        content = f.read()

    modified = modify_pddl_for_negative_goals(content)

    # Write to temp file
    output_file = os.path.join(current_dir, 'temp', 'neg_domain.pddl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(modified)

    return output_file


# =============================================================================
# Problem Construction
# =============================================================================

def create_pddl_problem(
    domain_file: str,
    init_state: List[Tuple],
    goal_state: List[Tuple]
) -> PDDLProblem:
    """
    Create a PDDLProblem from domain file and state specifications.

    Args:
        domain_file: Path to PDDL domain file
        init_state: List of initial state predicates
        goal_state: List of goal predicates

    Returns:
        PDDLProblem instance
    """
    domain_pddl = read_pddl_file(domain_file)
    constant_map = {}
    stream_pddl = None
    stream_map = {}
    goal = And(*goal_state)

    return PDDLProblem(
        domain_pddl,
        constant_map,
        stream_pddl,
        stream_map,
        init_state,
        goal
    )


# =============================================================================
# Main Solver Interface
# =============================================================================

def solve_pddl(
    init_state: List[Tuple],
    goal_state: List[Tuple],
    domain_file: str = "domain.pddl",
    use_negative_goals: bool = True,
    planner: str = "ff-astar",
    debug: bool = False
) -> Optional[List[Tuple]]:
    """
    Solve a PDDL planning problem.

    Args:
        init_state: List of initial state predicates
        goal_state: List of goal predicates
        domain_file: Path to PDDL domain file
        use_negative_goals: Whether to process domain for negative goals
        planner: Planner to use ('ff-astar', 'ff-eager', 'dijkstra')
        debug: Print debug information

    Returns:
        List of actions as (name, args) tuples, or None if no plan found

    Example:
        >>> init = [('box', '0'), ('on-table', '0', '9'), ('hand_free', '8')]
        >>> goal = [('above', '0', '1')]
        >>> plan = solve_pddl(init, goal)
        >>> # plan = [('pick-up', ('0', '9', '8')), ('put-down', ('0', '1', '8'))]
    """
    # Resolve domain file path
    if not os.path.isabs(domain_file):
        domain_file = os.path.join(current_dir, domain_file)

    # Process domain for negative goals if needed
    processed_domain = process_domain_file(domain_file, use_negative_goals)

    # Create problem
    problem = create_pddl_problem(processed_domain, init_state, goal_state)

    # Create parser for default args
    parser = create_parser()
    args = parser.parse_args([])

    # Solve (suppress PDDLStream verbose output)
    import io
    import sys
    try:
        # Redirect stdout to suppress PDDLStream's verbose output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            solution = solve(
                problem,
                algorithm='adaptive',
                unit_costs=False,
                debug=debug,
                max_failures=5,
                planner=planner,
                reorder=False,
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        if solution is None or solution.plan is None:
            return None

        # Extract plan as list of (action_name, args) tuples
        plan = []
        for action in solution.plan:
            plan.append((action.name, action.args))

        return plan

    except Exception as e:
        print(f"PDDL solver error: {e}")
        return None


def plan_to_string(plan: List[Tuple]) -> str:
    """Convert a plan to a readable string."""
    if plan is None:
        return "No plan found"

    lines = []
    for i, (action, args) in enumerate(plan):
        args_str = ', '.join(str(a) for a in args)
        lines.append(f"{i+1}. {action}({args_str})")

    return '\n'.join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test with a simple bridge goal
    init_state = [
        ('box', '0'), ('box', '1'), ('box', '2'), ('box', '3'), ('box', '4'),
        ('table', '9'), ('robot', '8'),
        ('on-table', '0', '9'), ('on-table', '1', '9'),
        ('on-table', '2', '9'), ('on-table', '3', '9'), ('on-table', '4', '9'),
        ('top', '0'), ('top', '1'), ('top', '2'), ('top', '3'), ('top', '4'),
        ('nothing_beside', '0'), ('nothing_beside', '1'),
        ('nothing_beside', '2'), ('nothing_beside', '3'), ('nothing_beside', '4'),
        ('hand_free', '8'),
    ]

    goal_state = [
        ('on-table', '1', '9'), ('on-table', '2', '9'),
        ('beside', '2', '1'),
        ('above', '0', '1'), ('above', '3', '2'),
        ('above_both', '4', '0', '3'),
    ]

    print("Solving PDDL problem...")
    plan = solve_pddl(init_state, goal_state, debug=False)
    print("\nPlan:")
    print(plan_to_string(plan))
