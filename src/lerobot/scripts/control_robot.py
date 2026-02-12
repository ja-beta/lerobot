#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to test individual motors on a robot, optionally with teleoperation.

Examples:
    Test wrist_roll with teleoperation (follower follows leader):
    python -m lerobot.scripts.control_robot \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.wchusbserial5AAF2702761 \
        --robot.id=None \
        --teleop.type=so101_leader \
        --teleop.port=/dev/tty.wchusbserial5A7A0157241 \
        --teleop.id=None \
        --motors="wrist_roll"
    
    Test with verbose debugging (shows detailed action values):
    python -m lerobot.scripts.control_robot \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.wchusbserial5AAF2702761 \
        --robot.id=None \
        --teleop.type=so101_leader \
        --teleop.port=/dev/tty.wchusbserial5A7A0157241 \
        --teleop.id=None \
        --motors="wrist_roll" \
        --verbose=true
    
    Test direct motor control (moves motor back and forth):
    python -m lerobot.scripts.control_robot \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.wchusbserial5AAF2702761 \
        --robot.id=None \
        --motors="wrist_roll" \
        --test_direct=true
"""

import logging
import time
from dataclasses import dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up

logger = logging.getLogger(__name__)


@dataclass
class ControlRobotConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None
    motors: str | None = None  # Comma-separated or space-separated motor names, e.g. "wrist_roll" or "wrist_roll,wrist_flex"
    test_direct: bool = False  # Test direct motor control (move motor back and forth)
    verbose: bool = False  # Show detailed debugging information
    fps: int = 60


def test_direct_motor_control(robot, motor_name, fps):
    """Test direct motor control by moving the motor back and forth."""
    motor_key = f"{motor_name}.pos"
    if motor_key not in robot.action_features:
        raise ValueError(f"Motor {motor_name} not found. Available: {list(robot.action_features.keys())}")
    
    print("\n" + "=" * 50)
    print(f"Direct Motor Test: {motor_name}")
    print("=" * 50)
    print("This will move the motor back and forth to test if it responds.")
    print("Press Ctrl+C to exit.")
    print("=" * 50 + "\n")
    
    # Get initial position
    obs = robot.get_observation()
    initial_pos = obs.get(motor_key, 0.0)
    print(f"Initial position: {initial_pos:.2f}")
    
    # Calculate movement range (small movement, ±10% of normalized range)
    movement_range = 0.1  # 10% of normalized range
    target_high = initial_pos + movement_range
    target_low = initial_pos - movement_range
    
    print(f"Will move between {target_low:.2f} and {target_high:.2f}")
    time.sleep(2)
    
    direction = 1
    cycle_count = 0
    last_switch = time.perf_counter()
    switch_interval = 3.0  # Switch direction every 3 seconds
    
    while True:
        loop_start = time.perf_counter()
        
        # Get current position
        obs = robot.get_observation()
        current_pos = obs.get(motor_key, 0.0)
        
        # Switch direction periodically
        if time.perf_counter() - last_switch > switch_interval:
            direction *= -1
            last_switch = time.perf_counter()
            cycle_count += 1
        
        # Calculate target position
        if direction > 0:
            target_pos = target_high
        else:
            target_pos = target_low
        
        # Send command
        action = {motor_key: target_pos}
        robot.send_action(action)
        
        # Display
        print(f"\rCycle {cycle_count} | Current: {current_pos:>7.2f} | Target: {target_pos:>7.2f} | Diff: {abs(current_pos - target_pos):>7.2f}", end="")
        
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)


def control_loop(robot, teleop, motors_to_control, verbose, fps):
    """Control loop that reads current positions and displays them, optionally with teleoperation."""
    # Determine which motors to control
    all_motor_keys = [m.replace(".pos", "") for m in robot.action_features.keys() if m.endswith(".pos")]
    
    if motors_to_control:
        # Filter to only the specified motors
        motors_to_control = [m for m in motors_to_control if m in all_motor_keys]
        if not motors_to_control:
            raise ValueError(
                f"None of the specified motors are available. "
                f"Available motors: {all_motor_keys}"
            )
        print(f"Testing motors: {motors_to_control}")
        display_motors = True  # Always display when testing specific motors
    else:
        motors_to_control = all_motor_keys
        print(f"Controlling all motors: {motors_to_control}")
        display_motors = False  # Don't spam output for all motors

    display_len = max(len(key) for key in motors_to_control) if motors_to_control else 20

    # Initialize processors if using teleoperation
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    print("\n" + "=" * 50)
    print("Motor Control Test")
    print("=" * 50)
    if teleop:
        print(f"Teleoperation mode: follower will follow leader for motors: {motors_to_control}")
        print("Move the leader's wrist_roll to see if the follower follows.")
        if verbose:
            print("Verbose mode: showing detailed action values")
    else:
        print("This script will display the current positions of the specified motors.")
        print("The motors will maintain their current positions.")
    print("Press Ctrl+C to exit.")
    print("=" * 50 + "\n")

    teleop_action = None  # Initialize for display section
    
    while True:
        loop_start = time.perf_counter()

        # Get robot observation (current positions)
        obs = robot.get_observation()

        if teleop:
            # Get teleop action
            raw_action = teleop.get_action()
            
            if verbose:
                # Show raw leader values
                leader_wrist_roll = raw_action.get("wrist_roll.pos", "NOT FOUND")
                print(f"\n[DEBUG] Leader raw wrist_roll.pos: {leader_wrist_roll}")
            
            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))
            
            if verbose:
                # Show processed teleop values
                processed_wrist_roll = teleop_action.get("wrist_roll.pos", "NOT FOUND")
                print(f"[DEBUG] Teleop processed wrist_roll.pos: {processed_wrist_roll}")
            
            # Filter to only the motors we want to control
            filtered_action = {}
            for motor in motors_to_control:
                motor_key = f"{motor}.pos"
                if motor_key in teleop_action:
                    filtered_action[motor_key] = teleop_action[motor_key]
                    if verbose:
                        print(f"[DEBUG] Using teleop value for {motor_key}: {teleop_action[motor_key]}")
                else:
                    # If motor not in teleop action, maintain current position
                    motor_key_obs = f"{motor}.pos"
                    if motor_key_obs in obs:
                        filtered_action[motor_key] = obs[motor_key_obs]
                        if verbose:
                            print(f"[DEBUG] Motor {motor_key} not in teleop_action, maintaining position: {obs[motor_key_obs]}")
                    else:
                        if verbose:
                            print(f"[WARNING] Motor {motor_key} not found in obs or teleop_action!")
            
            if verbose:
                print(f"[DEBUG] Filtered action (before robot processor): {filtered_action}")
            
            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((filtered_action, obs))
            
            if verbose:
                print(f"[DEBUG] Final robot action to send: {robot_action_to_send}")
                print()  # Blank line for readability
        else:
            # Filter observations to only the motors we care about
            filtered_obs = {}
            for motor in motors_to_control:
                motor_key = f"{motor}.pos"
                if motor_key in obs:
                    filtered_obs[motor] = obs[motor_key]

            # Send the current positions back to maintain position
            robot_action_to_send = {f"{motor}.pos": filtered_obs[motor] for motor in filtered_obs.keys()}
            teleop_action = None  # Reset when not using teleop

        # Send action to robot
        robot.send_action(robot_action_to_send)

        if display_motors:
            # Display the motor positions
            print("\n" + "-" * (display_len + 50))
            header = f"{'MOTOR':<{display_len}} | {'POSITION':>10} | {'TARGET':>10}"
            if teleop:
                header += f" | {'LEADER':>10}"
            print(header)
            print("-" * (display_len + 50))
            for motor in motors_to_control:
                motor_key = f"{motor}.pos"
                current_pos = obs.get(motor_key, 0.0)
                target_pos = robot_action_to_send.get(motor_key, current_pos)
                line = f"{motor:<{display_len}} | {current_pos:>10.2f} | {target_pos:>10.2f}"
                if teleop:
                    # Show leader value if available
                    leader_val = "N/A"
                    if teleop_action and motor_key in teleop_action:
                        leader_val = f"{teleop_action[motor_key]:>10.2f}"
                    line += f" | {leader_val:>10}"
                print(line)
            move_cursor_up(len(motors_to_control) + 4)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        if display_motors:
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")


@parser.wrap()
def control_robot(cfg: ControlRobotConfig):
    init_logging()
    logging.info(pformat(cfg.__dict__))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    teleop = None
    if cfg.teleop:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

    # Parse motors string into list
    motors_list = None
    if cfg.motors:
        # Support both comma-separated and space-separated
        motors_list = [m.strip() for m in cfg.motors.replace(",", " ").split() if m.strip()]

    try:
        # Test direct motor control if requested
        if cfg.test_direct and motors_list and len(motors_list) == 1:
            test_direct_motor_control(robot, motors_list[0], cfg.fps)
        else:
            control_loop(
                robot=robot,
                teleop=teleop,
                motors_to_control=motors_list,
                verbose=cfg.verbose,
                fps=cfg.fps,
            )
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        if teleop:
            teleop.disconnect()
        robot.disconnect()
        print("Robot disconnected.")


def main():
    register_third_party_devices()
    control_robot()


if __name__ == "__main__":
    main()
