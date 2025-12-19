"""
Replay a recorded episode using joint control.
Reads state.csv and executes joint positions sequentially.
"""

import argparse
import csv
import time
import sys
import os

import rtde_control
import rtde_receive

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collect'))
import robotiq_gripper


class EpisodeReplayer:
    """Replays recorded episodes using joint control."""

    def __init__(self, robot_ip='192.168.56.101', gripper_port=63352, speed=0.4, acceleration=0.4):
        self.speed = speed
        self.acceleration = acceleration

        print(f"Connecting to robot at {robot_ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        print(f"Connecting to gripper on port {gripper_port}...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(robot_ip, gripper_port)
        self.gripper.activate()

        print("Robot and gripper ready.")

    def load_episode(self, episode_path):
        """Load state.csv from episode directory."""
        csv_path = os.path.join(episode_path, 'state.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"state.csv not found in {episode_path}")

        states = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = {
                    'frame_idx': int(row['frame_idx']),
                    'timestamp': float(row['timestamp']),
                    'joints': [
                        float(row['joint_0']),
                        float(row['joint_1']),
                        float(row['joint_2']),
                        float(row['joint_3']),
                        float(row['joint_4']),
                        float(row['joint_5']),
                    ],
                    'gripper_pos': int(float(row['gripper_pos'])),
                }
                states.append(state)

        print(f"Loaded {len(states)} frames from {csv_path}")
        return states

    def replay(self, states, rate=None, replay_gripper=True, dry_run=False):
        """Replay the episode.

        Args:
            states: List of state dicts from load_episode()
            rate: Playback rate multiplier (None = use original timing, 0.5 = half speed, 2.0 = double speed)
            replay_gripper: Whether to replay gripper positions
            dry_run: If True, only print actions without executing
        """
        if len(states) < 2:
            print("Not enough states to replay")
            return

        print(f"\nReplaying {len(states)} frames...")
        print(f"  Rate: {rate if rate else 'original timing'}")
        print(f"  Gripper: {'enabled' if replay_gripper else 'disabled'}")
        print(f"  Dry run: {dry_run}")
        print()

        if not dry_run:
            input("Press Enter to start replay (Ctrl+C to abort)...")

        start_time = time.time()
        first_timestamp = states[0]['timestamp']

        for i, state in enumerate(states):
            frame_idx = state['frame_idx']
            joints = state['joints']
            gripper_pos = state['gripper_pos']

            # Calculate timing
            if rate is not None and i > 0:
                # Fixed rate playback
                target_time = start_time + (i / (rate * 20))  # Assuming 20 Hz original
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            elif i > 0:
                # Original timing playback
                time_delta = state['timestamp'] - states[i-1]['timestamp']
                time.sleep(max(0, time_delta))

            # Execute
            if dry_run:
                print(f"[{frame_idx:4d}] joints={[f'{j:.3f}' for j in joints]}, gripper={gripper_pos}")
            else:
                # Move joints
                self.rtde_c.moveJ(joints, self.speed, self.acceleration)

                # Move gripper
                if replay_gripper:
                    self.gripper.move(gripper_pos, 255, 255)  # Max speed/force for responsiveness

                # Progress indicator
                if i % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"Frame {frame_idx}/{len(states)-1} ({elapsed:.1f}s)")

        elapsed = time.time() - start_time
        print(f"\nReplay complete. Total time: {elapsed:.1f}s")

    def go_to_start(self, states):
        """Move robot to the starting position of the episode."""
        if not states:
            return

        print("Moving to start position...")
        start_joints = states[0]['joints']
        start_gripper = states[0]['gripper_pos']

        self.rtde_c.moveJ(start_joints, self.speed * 0.5, self.acceleration * 0.5)  # Slower for safety
        self.gripper.move_and_wait_for_pos(start_gripper, 100, 100)
        print("At start position.")


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode")
    parser.add_argument("episode", help="Path to episode directory (containing state.csv)")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="Robot IP address")
    parser.add_argument("--gripper-port", type=int, default=63352, help="Gripper port")
    parser.add_argument("--speed", type=float, default=0.4, help="Joint speed (rad/s)")
    parser.add_argument("--acceleration", type=float, default=0.4, help="Joint acceleration (rad/s^2)")
    parser.add_argument("--rate", type=float, default=None, help="Playback rate multiplier (default: original timing)")
    parser.add_argument("--no-gripper", action="store_true", help="Don't replay gripper")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--skip-start", action="store_true", help="Skip moving to start position")
    args = parser.parse_args()

    # Initialize replayer
    if not args.dry_run:
        replayer = EpisodeReplayer(
            robot_ip=args.robot_ip,
            gripper_port=args.gripper_port,
            speed=args.speed,
            acceleration=args.acceleration
        )
    else:
        replayer = type('DummyReplayer', (), {
            'load_episode': lambda self, p: EpisodeReplayer.load_episode(None, p),
            'go_to_start': lambda self, s: print("(dry run) Would move to start"),
            'replay': lambda self, s, **kw: EpisodeReplayer.replay(None, s, dry_run=True, **kw)
        })()

    # Load episode
    states = replayer.load_episode(args.episode)

    # Move to start
    if not args.skip_start and not args.dry_run:
        replayer.go_to_start(states)

    # Replay
    try:
        replayer.replay(
            states,
            rate=args.rate,
            replay_gripper=not args.no_gripper,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")


if __name__ == "__main__":
    main()
