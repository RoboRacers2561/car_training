# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils import convert_dict_to_backend
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2
import logging

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import *
from car_training.tasks.manager_based.car_training.mdp.observation import robot_pos, robot_vel
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import GpuSafeLogger

_LOGGER: Optional[GpuSafeLogger] = None


def get_logger(env: ManagerBasedRLEnv) -> GpuSafeLogger:
    """
    Get a child logger of the project logger for the car_training.mdp.observation module.

    :param env: The environment instance containing the base logger.
    :type env: ManagerBasedRLEnv
    
    :return: Child logger for the observation module.
    :rtype: GpuSafeLogger
    """
    global _LOGGER # creating a cache to avoid repeated calls
    if _LOGGER is None:
        base = env.unwrapped.logger
        short_name = ".".join(__name__.split(".")[-2:]) # car_training.mdp.observation
        _LOGGER = base.getChild(short_name)
    return _LOGGER


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
    logger = get_logger(env)

    logger.debug("Calculating total reward...")

    camera: Articulation = env.scene[camera_cfg.name]
    
    cam_data = convert_dict_to_backend(
                {k: v for k, v in camera.data.output.items()}, backend="numpy")

    # compute the reward

    H, W, _ = cam_data['rgb'][0].shape
    logger.debug(f"Camera image shape: H={H}, W={W}")
    

    # Here I think we can have access to the sensors and images for each copy of the simulation.
    # We can compute the reward for each copy separately and then combine them.

    all_rewards = []
    index = 0
    # TODO: I added index here because the car_center is expected to be a tuple or list not a list[list], but I think 
    # there is a better way to do this, especially since we are looping through all images, which is O(num_envs), maybe there
    # is a way to vectorize this.
    for frame in cam_data['rgb']:
        # print(f"frames.shape:{frames.shape, type(frames)}")
        # cv2.imwrite("debug_image.png", frame)  # Save the image for debugging
        # print("Saved debug_image.png")
        # for each copy of the simulation find the track points and compute the reward
        track_points_list = find_track_points_from_image(frame)
        car_center = robot_pos(env, robot_cfg, do_convert_coords=True).cpu().numpy()[index]
        car_velocity = robot_vel(env, robot_cfg).cpu().numpy()[index]
        car_speed = np.linalg.norm(car_velocity)
        logger.debug(f"car_speed type: {type(car_speed)}")
        logger.debug_lazy("car speed %s", lambda: car_speed)

        # cv2.circle(frame, (int(car_center[0]), int(car_center[1])), 5, (0, 0, 255), -1)  # Draw car center in red
        # cv2.imwrite(f"debug_image_with_car_center_{index}.png", frame)  # Save the image with car center for debugging
        # print(f"type of car_center: {type(car_center)}, shape: {car_center.shape}")
        # print(f"Car center: {car_center}, Car velocity: {car_velocity}, Car speed: {car_speed}")

        # Now compute the reward for each simulation copy
        distance_reward = distance_from_center_reward(car_center, track_points_list, weight=3.0)
        # print(f"Distance reward: {distance_reward}")
        velocity_reward_value = velocity_reward(car_speed, max_speed, weight=1.5)
        alive_reward = remains_alive_reward(weight=0.2)
        track_penalty = leaves_track_penalty(car_center, track_points_list, weight=0.5)

        total_reward = (distance_reward + velocity_reward_value + alive_reward - track_penalty)

        all_rewards.append(total_reward)
        logger.debug(f"Simulation {index}: Distance Reward: {distance_reward}, Velocity Reward: {velocity_reward_value}, Alive Reward: {alive_reward}, Track Penalty: {track_penalty}, Total Reward: {total_reward}")
        index += 1

    total_reward = torch.tensor(all_rewards, dtype=torch.float32, device=env.device)
    return total_reward

