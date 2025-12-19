"""
Data collection script using plan-based execution.
Reads task definition from plan.yaml and records robot data.
"""

import time
import yaml

from utils import find_datacount
from ur5e_controller import UR5E_CONTROLLER
from recorder import DataRecorder
from plan_executor import PlanExecutor


def main(plan_path: str = "plan.yaml", config_path: str = "config.yaml"):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize robot with config values
    ur5e = UR5E_CONTROLLER(
        ip=config['robot']['ip'],
        gripper_port=config['gripper']['port'],
        speed=config['robot']['speed'],
        acceleration=config['robot']['acceleration']
    )

    # Initialize plan executor with gripper settings
    executor = PlanExecutor(
        ur5e,
        plan_path,
        gripper_speed=config['gripper']['speed'],
        gripper_force=config['gripper']['force']
    )
    task_config = executor.get_task_config()

    # Initialize recorder with camera settings
    recorder = DataRecorder(
        rtde_r=ur5e.rtde_r,
        gripper=ur5e.gripper,
        camera_serials=config['recording']['camera_serials'],
        hz=config['recording']['hz'],
        camera_resolution=tuple(config['recording']['camera_resolution'])
    )

    # Test all recording components before starting
    recorder.test()

    # Get task settings
    task_name = task_config.get('name', 'unnamed_task')
    num_rounds = task_config.get('num_rounds', 1)
    should_record = task_config.get('record', True)

    print(f"Task: {task_name}")
    print(f"Rounds: {num_rounds}")
    print(f"Recording: {should_record}")

    try:
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")

            # Start recording
            if should_record:
                episode_num = find_datacount(task_name)
                output_dir = f"{config['data']['output_dir']}{task_name}/{episode_num}"
                recorder.start(output_dir)

            # Execute main sequence
            print("Executing main sequence...")
            executor.run_main_sequence()

            # Stop recording
            if should_record:
                recorder.stop()

            # Execute reset sequence (not recorded)
            print("Executing reset sequence...")
            executor.run_reset_sequence()

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        recorder.stop()
        recorder.close()
        print("Cleanup complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data collection from plan")
    parser.add_argument("--plan", default="plan.yaml", help="Path to plan YAML file")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    main(plan_path=args.plan, config_path=args.config)
