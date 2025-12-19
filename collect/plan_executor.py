"""
Plan executor for UR5e robot.
Executes action sequences defined in YAML plan files.
"""

import time
import yaml
import numpy as np
from typing import Dict, List, Optional


class PlanExecutor:
    """Executes action plans defined in YAML."""

    def __init__(self, robot, plan_path: str = "plan.yaml", gripper_speed: int = 100, gripper_force: int = 100):
        """
        Args:
            robot: UR5E_CONTROLLER instance
            plan_path: Path to the plan YAML file
            gripper_speed: Gripper speed (0-255)
            gripper_force: Gripper force (0-255)
        """
        self.robot = robot
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.rng = np.random.default_rng()

        # Load plan
        with open(plan_path, 'r') as f:
            self.plan = yaml.safe_load(f)

        # Parse plan components
        self.task = self.plan.get('task', {})
        self.poses = self.plan.get('poses', {})
        self.heights = self.plan.get('heights', {})
        self.actions = self.plan.get('actions', {})
        self.sequence = self.plan.get('sequence', [])
        self.reset_sequence = self.plan.get('reset', [])

        # Track dynamic pose updates (for jitter)
        self.dynamic_poses = dict(self.poses)

    def get_task_config(self) -> Dict:
        """Return task configuration."""
        return self.task

    def resolve_pose(self, pose_ref) -> List[float]:
        """Resolve a pose reference to actual coordinates.

        Args:
            pose_ref: Either a pose name (string) or direct coordinates (list)

        Returns:
            [x, y, z, qx, qy, qz, qw]
        """
        if isinstance(pose_ref, str):
            if pose_ref in self.dynamic_poses:
                return list(self.dynamic_poses[pose_ref])
            elif pose_ref in self.poses:
                return list(self.poses[pose_ref])
            else:
                raise ValueError(f"Unknown pose: {pose_ref}")
        elif isinstance(pose_ref, list):
            return list(pose_ref)
        else:
            raise ValueError(f"Invalid pose reference: {pose_ref}")

    def apply_height(self, pose: List[float], height: Optional[float]) -> List[float]:
        """Apply height override to pose."""
        if height is not None:
            pose = list(pose)
            pose[2] = height
        return pose

    def apply_jitter(self, pose: List[float], jitter: List[float]) -> List[float]:
        """Apply random jitter to pose.

        Args:
            pose: [x, y, z, qx, qy, qz, qw]
            jitter: [max_x, max_y, max_z] - max jitter in each dimension
        """
        pose = list(pose)
        for i, max_jitter in enumerate(jitter[:3]):
            if max_jitter > 0:
                offset = self.rng.uniform(-max_jitter, max_jitter)
                pose[i] += offset
        return pose

    def apply_random_offset(self, pose: List[float], offset: List[float]) -> List[float]:
        """Apply random positive offset to pose.

        Args:
            pose: [x, y, z, qx, qy, qz, qw]
            offset: [max_x, max_y, max_z] - max offset in each dimension (positive only)
        """
        pose = list(pose)
        for i, max_offset in enumerate(offset[:3]):
            if max_offset > 0:
                pose[i] += self.rng.uniform(0, max_offset)
        return pose

    # ==================== Primitive Actions ====================

    def execute_move_to(self, params: Dict) -> None:
        """Execute move_to primitive.

        params:
            pose: pose name or [x,y,z,qx,qy,qz,qw]
            height: optional height override
            jitter: optional [max_x, max_y, max_z]
            random_offset: optional [max_x, max_y, max_z]
        """
        pose = self.resolve_pose(params['pose'])

        if 'height' in params:
            pose = self.apply_height(pose, params['height'])

        if 'random_offset' in params:
            pose = self.apply_random_offset(pose, params['random_offset'])

        if 'jitter' in params:
            pose = self.apply_jitter(pose, params['jitter'])

        self.robot.move_to_target(pose)

    def execute_move_delta(self, delta: List[float]) -> None:
        """Execute move_delta primitive.

        delta: [dx, dy, dz, dqx, dqy, dqz, dqw]
        """
        self.robot.move_delta(delta)

    def execute_gripper(self, command) -> None:
        """Execute gripper primitive.

        command: 'open', 'close', or int 0-255
        """
        if command == 'open':
            self.robot.gripper.move_and_wait_for_pos(0, self.gripper_speed, self.gripper_force)
        elif command == 'close':
            self.robot.gripper.move_and_wait_for_pos(255, self.gripper_speed, self.gripper_force)
        elif isinstance(command, int):
            self.robot.gripper.move_and_wait_for_pos(command, self.gripper_speed, self.gripper_force)
        else:
            raise ValueError(f"Invalid gripper command: {command}")

    def execute_wait(self, seconds: float) -> None:
        """Execute wait primitive."""
        time.sleep(seconds)

    # ==================== Action Execution ====================

    def substitute_variables(self, step: Dict, variables: Dict) -> Dict:
        """Substitute $variables in a step with actual values."""
        result = {}
        for key, value in step.items():
            if isinstance(value, str) and value.startswith('$'):
                var_name = value[1:]  # Remove $
                if var_name in variables:
                    result[key] = variables[var_name]
                else:
                    raise ValueError(f"Unknown variable: {value}")
            elif isinstance(value, dict):
                result[key] = self.substitute_variables(value, variables)
            else:
                result[key] = value
        return result

    def execute_primitive(self, step: Dict) -> None:
        """Execute a single primitive step."""
        if 'move_to' in step:
            self.execute_move_to(step['move_to'])
        elif 'move_delta' in step:
            self.execute_move_delta(step['move_delta'])
        elif 'gripper' in step:
            self.execute_gripper(step['gripper'])
        elif 'wait' in step:
            self.execute_wait(step['wait'])
        else:
            raise ValueError(f"Unknown primitive: {step}")

    def execute_action(self, action_item) -> None:
        """Execute an action (primitive or compound).

        action_item can be:
            - str: name of compound action with no params (e.g., "go_home")
            - dict: {action_name: {params}} (e.g., {"pick": {"pose": "chip_start", "height": 0.2}})
        """
        if isinstance(action_item, str):
            # Compound action with no parameters
            action_name = action_item
            params = {}
        elif isinstance(action_item, dict):
            # Could be primitive or compound with params
            action_name = list(action_item.keys())[0]
            params = action_item[action_name]
            if params is None:
                params = {}

            # Check if it's a primitive
            if action_name in ['move_to', 'move_delta', 'gripper', 'wait']:
                self.execute_primitive(action_item)
                return
        else:
            raise ValueError(f"Invalid action item: {action_item}")

        # Execute compound action
        if action_name not in self.actions:
            raise ValueError(f"Unknown action: {action_name}")

        action_steps = self.actions[action_name]
        for step in action_steps:
            # Substitute variables
            resolved_step = self.substitute_variables(step, params)
            self.execute_primitive(resolved_step)

    def execute_sequence(self, sequence: List, update_poses: bool = False) -> None:
        """Execute a sequence of actions.

        Args:
            sequence: List of action items
            update_poses: If True, update dynamic_poses after place with jitter
        """
        for action_item in sequence:
            self.execute_action(action_item)

            # Track pose updates from jitter (for reset sequence)
            if update_poses and isinstance(action_item, dict):
                action_name = list(action_item.keys())[0]
                params = action_item.get(action_name, {})
                if params and 'jitter' in params and 'pose' in params:
                    pose_name = params['pose']
                    if isinstance(pose_name, str) and pose_name in self.poses:
                        # Update the dynamic pose with jittered value
                        original = self.resolve_pose(pose_name)
                        jittered = self.apply_jitter(original, params['jitter'])
                        if 'height' in params:
                            jittered = self.apply_height(jittered, params['height'])
                        self.dynamic_poses[pose_name] = jittered

    def run_main_sequence(self) -> None:
        """Run the main recorded sequence."""
        self.execute_sequence(self.sequence)

    def run_reset_sequence(self) -> None:
        """Run the reset sequence (updates poses for next round)."""
        self.execute_sequence(self.reset_sequence, update_poses=True)

    def reset_dynamic_poses(self) -> None:
        """Reset dynamic poses to original values."""
        self.dynamic_poses = dict(self.poses)
