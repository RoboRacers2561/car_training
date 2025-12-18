# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with custom action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Custom action agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import car_training.tasks  # noqa: F401
import carb
import omni




class WasdTeleop:
    def __init__(self, throttle_speed=10.0, steer_angle=0.5):
        self.throttle_speed = float(throttle_speed)
        self.steer_angle = float(steer_angle)

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, e: carb.input.KeyboardEvent):
        # KEY_PRESS / KEY_REPEAT => pressed, KEY_RELEASE => released
        print(f"Key event: {e.input}, type: {e.type}")
        # pressed = e.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT)

        if e.input == carb.input.KeyboardInput.W:
            print("Increasing throttle")
            self.throttle_speed += 1 
        elif e.input == carb.input.KeyboardInput.S:
            print("Decreasing throttle")
            self.throttle_speed -= 1
        elif e.input == carb.input.KeyboardInput.A:
            print("Increasing steering angle to the left")
            self.steer_angle += 0.1
        elif e.input == carb.input.KeyboardInput.D:
            print("Decreasing steering angle to the right")
            self.steer_angle -= 0.1
        return True  # keep receiving events
    
    def get(self):
        return self.throttle_speed, self.steer_angle

def main():
    """Custom action agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    print("Joint names:", env.unwrapped.scene["robot"].data.joint_names)
    teleop = WasdTeleop(throttle_speed=10.0, steer_angle=0.5)
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute custom actions
            # 0 = Rear left wheel effort, 1 = Rear right wheel effort,
            # 2 = Front right steering angle, 3 = Front left steering angle
            # Use small values first; scales are defined in the env cfg (100 for effort, 1 for steering)
            # throttle_cmd = 10 # wheels velocity rad/s
            # steer_cmd = 0.5      # front angle radians

            # get keyboard input
            throttle_cmd, steer_cmd = teleop.get()

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions = set_throttle(actions, throttle_cmd)
            actions = set_steering(actions, steer_cmd)
            
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

def set_throttle(action, throttle_cmd):
    """Set throttle commands in the action tensor."""
    action[:, 0] = throttle_cmd
    action[:, 1] = throttle_cmd
    return action

def set_steering(action, steer_cmd):
    """Set steering commands in the action tensor."""
    action[:, 2] = steer_cmd
    action[:, 3] = steer_cmd
    return action

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
