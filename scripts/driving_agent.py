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

import logging

import car_training.tasks  # noqa: F401
import carb
import omni
from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger

class WasdTeleop:
    def __init__(self, throttle_speed=10.0, steer_angle=0.5, number_of_envs=1):
        self.throttle_speed = float(throttle_speed)
        self.steer_angle = float(steer_angle)
        self.number_of_envs = number_of_envs
        self.current_env_index = 0

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, e: carb.input.KeyboardEvent):
        # Only handle press (ignore release/char)
        if e.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

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
        elif e.input == carb.input.KeyboardInput.TAB:
            print("Switching environment")
            self.current_env_index = (self.current_env_index + 1) % self.number_of_envs
            print(f"Current environment index: {self.current_env_index}")
            
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

    # setup logger
    logger = setup_logger(task_name=args_cli.task, name="driving_agent_logger", console_level=logging.INFO, file_level=logging.DEBUG)

    env.unwrapped.logger = logger

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # initialize static track image
    init_static_track_image(env.unwrapped, camera_name="overhead_camera")

    print("Joint names:", env.unwrapped.scene["robot"].data.joint_names)

    number_of_envs = env.unwrapped.num_envs
    print(f"Number of environments: {number_of_envs}")
    
    teleop = WasdTeleop(throttle_speed=1.0, steer_angle=0.5, number_of_envs=number_of_envs)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # get keyboard input
            throttle_cmd, steer_cmd = teleop.get()
            # throttle is in rad/s, steering is steering angle of the front wheels in radians

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            
            actions = set_drive(actions, throttle_cmd, steer_cmd, teleop.current_env_index)
            
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


def set_drive(action, throttle_cmd, steer_cmd, env_index):
    """Set [throttle, steer] for one env."""
    action[env_index, 0] = throttle_cmd   # throttle
    action[env_index, 1] = steer_cmd      # steer (center/bicycle steering)
    return action

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
