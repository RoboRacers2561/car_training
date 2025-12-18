# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils import convert_dict_to_backend
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import *
from car_training.tasks.manager_based.car_training.mdp.observation import robot_pos, robot_vel


def get_total_reward(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, completed_laps: int, total_laps: int, max_speed: float) -> torch.Tensor:
    """
    Calculate the total reward based on multiple factors.

    :param completed_laps: Number of laps completed by the car.
    :type completed_laps: int
    :param total_laps: Total number of laps in the race.
    :type total_laps: int
    :param max_speed: Maximum speed of the car.
    :type max_speed: float

    :return: Total reward calculated from various factors.
    :rtype: float
    """

    camera: Articulation = env.scene[camera_cfg.name]
    
    cam_data = convert_dict_to_backend(
                {k: v for k, v in camera.data.output.items()}, backend="numpy")

    # compute the reward

    H, W, _ = cam_data['rgb'][0].shape

    # Here I think we can have access to the sensors and images for each copy of the simulation.
    # We can compute the reward for each copy separately and then combine them.

    all_rewards = []
    for frames in cam_data['rgb']:
        # print(f"frames.shape:{frames.shape, type(frames)}")
        cv2.imwrite("debug_image.png", frames)  # Save the image for debugging
        # print("Saved debug_image.png")
        # for each copy of the simulation find the track points and compute the reward
        track_points_list = find_track_points_from_image(frames)
        car_center = robot_pos(env, robot_cfg).cpu().numpy()[0] 
        car_velocity = robot_vel(env, robot_cfg).cpu().numpy()[0]  
        car_speed = np.linalg.norm(car_velocity)

        print(f"Car center: {car_center}, Car velocity: {car_velocity}, Car speed: {car_speed}")

        # Now compute the reward for each simulation copy
        distance_reward = distance_from_center_reward(car_center, track_points_list, weight=2.0)
        velocity_reward_value = velocity_reward(car_speed, max_speed, weight=6.5)
        alive_reward = remains_alive_reward(weight=0.2)
        track_penalty = leaves_track_penalty(car_center, track_points_list, weight=2.0)

        total_reward = (distance_reward + velocity_reward_value + alive_reward - track_penalty)

        all_rewards.append(total_reward)

    total_reward = torch.tensor(all_rewards, dtype=torch.float32, device=env.device)
    return total_reward

