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
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2
import logging

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import *
from car_training.tasks.manager_based.car_training.mdp.observation import robot_pos, robot_vel, robot_heading
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

    rgb = camera.data.output['rgb']

    rgb_cpu = rgb.detach().cpu().numpy()

    car_centers = robot_pos(env, robot_cfg, do_convert_coords=True).detach().cpu().numpy()
    car_vels = robot_vel(env, robot_cfg).detach().cpu().numpy()
    car_speeds = np.linalg.norm(car_vels, axis=1)

    num_envs = car_centers.shape[0]
    rewards = np.empty((num_envs,), dtype=np.float32)


    for i in range(num_envs):
        # for each copy of the simulation find the track points and compute the reward
        frame = rgb_cpu[i]
        track_points_list = find_track_points_from_image(frame)
        
        car_center = car_centers[i]
        car_speed = float(car_speeds[i])
        # car_heading = robot_heading(env, robot_cfg).cpu().numpy()[index]
        logger.debug(f"car_speed type: {type(car_speed)}")
        logger.debug_lazy("car speed %s", lambda: car_speed)

        # draw the velocity vector on the image for debugging
        # start_point = (int(car_center[0]), int(car_center[1]))
        # end_point = (int(car_center[0] + car_velocity[0]*100), int(car_center[1] + car_velocity[1]*100))
        # cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Draw velocity vector in blue
        # end_point = (int(car_center[0] + 50 * car_heading[0]), int(car_center[1] + 50 * car_heading[1]))
        # cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Draw heading vector in blue
        # cv2.circle(frame, (int(car_center[0]), int(car_center[1])), 5, (0, 255, 0), -1)  # Draw car center in green
        # cv2.imwrite(f"debug_image_with_velocity_{index}.png", frame)  # Save the image with velocity vector for debugging

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

        rewards[i] = total_reward
        logger.debug(f"Simulation {i}: Distance Reward: {distance_reward}, Velocity Reward: {velocity_reward_value}, Alive Reward: {alive_reward}, Track Penalty: {track_penalty}, Total Reward: {total_reward}")

    return torch.from_numpy(rewards).to(device=env.device)

