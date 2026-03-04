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
import torch.nn.functional as F

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_closest_track_points
from car_training.tasks.manager_based.car_training.mdp.observation import (
    robot_pos,
    aligned_robot_vel,
    robot_vel,
)
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import (
    GpuSafeLogger,
)

_LOGGER: Optional[GpuSafeLogger] = None

MAX_DISTANCE_FROM_LINE = 40



def get_logger(env: ManagerBasedRLEnv) -> GpuSafeLogger:
    """
    Get a child logger of the project logger for the car_training.mdp.observation module.

    :param env: The environment instance containing the base logger.
    :type env: ManagerBasedRLEnv

    :return: Child logger for the observation module.
    :rtype: GpuSafeLogger
    """
    global _LOGGER  # creating a cache to avoid repeated calls
    if _LOGGER is None:
        base = env.unwrapped.logger
        # car_training.mdp.observation
        short_name = ".".join(__name__.split(".")[-2:])
        _LOGGER = base.getChild(short_name)
    return _LOGGER

def distance_from_center_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for being close to the center of the track.

    :param env: The environment instance containing the track and robot information.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot to extract its position.
    :type robot_cfg: SceneEntityCfg

    :return: Distance from center reward.
    :rtype: torch.Tensor
    """
    car_centers = robot_pos(env, robot_cfg, do_convert_coords=True)
    track_points = env.track_points_tensor  # (num_envs, num_track_points, 2)
    track_points_lengths = env.track_points_lengths  # (num_envs,)

    closest_track_points = find_closest_track_points(car_centers, track_points, track_points_lengths)  # (num_envs, 2)
    distances = torch.norm(car_centers - closest_track_points, dim=1)  #

    max_distance = MAX_DISTANCE_FROM_LINE
    normalized_distances = distances / max_distance
    # Reward is higher when the distance is smaller
    reward = 1.0 - normalized_distances
    return reward

def velocity_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, max_speed: float) -> torch.Tensor:
    """
    Reward for maintaining a high speed.

    :param env: The environment instance containing the robot information.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot to extract its velocity.
    :type robot_cfg: SceneEntityCfg
    :param max_speed: Maximum speed of the car for normalization.
    :type max_speed: float

    :return: Velocity reward.
    :rtype: torch.Tensor
    """
    car_vels = aligned_robot_vel(env, robot_cfg)
    car_speeds = torch.norm(car_vels, dim=1)
    normalized_speeds = car_speeds / max_speed
    return normalized_speeds

def remains_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns a constant reward for remaining alive.

    :return: Constant reward value.
    :rtype: torch.Tensor
    """
    return torch.ones((env.num_envs,), device=env.device)

def steering_jerk_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for smooth steering by penalizing large changes in steering angle.

    :return: Smooth steering reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    logger.debug("Calculating steering jerk penalty...")

    drive = env.action_manager._terms["drive"]

    # Need to create a buffer to average second derivitive over multiple points
    # Convolution kernal is [1,-2,1]
    def d2_filter(buffer):
        # Kernel: [-1, 2, -1] 
        kernel = torch.tensor([1.0, -2.0, 1.0], device=buffer.device)
        kernel = kernel.view(1, 1, -1)  
        buffer_input = buffer.unsqueeze(1)
        result = F.conv1d(buffer_input, kernel, padding=0)  # (num_envs, 1, buffer_size-2)
        return result.squeeze(1)  # (num_envs, buffer_size-2) 

    steer_buffer = drive.action_buffer[:, :, 1]
    
    min_buffer_size = 3
    valid_envs = drive.buffer_count >= min_buffer_size
    
    if not valid_envs.any():
        # Not enough history yet, return zero penalty
        logger.debug("Insufficient buffer history for steering jerk penalty.")
        return torch.zeros((env.num_envs,), device=env.device)
    # dt is optional; keep if you want rate-invariance
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    #Updated j with filter 
    j = d2_filter(steer_buffer)/(dt*dt)

    mask = torch.zeros_like(j, dtype=torch.bool)
    for env_idx in range(env.num_envs):
        count = drive.buffer_count[env_idx].item()
        if count >= 3:
            num_valid = count - 2
            mask[env_idx, -num_valid:] = True

    j_sqr_mask = (j ** 2) * mask.float()
    num_valid = mask.sum(dim=1).float().clamp(min=1.0)
    j_rms = torch.sqrt(j_sqr_mask.sum(dim=1) / num_valid)

    penalty = -1 * (j_rms)

    logger.debug_lazy("Steering angles prev2: %s", lambda: steer_buffer)
    logger.debug_lazy("Decimation: %s", lambda: env.cfg.decimation)
    logger.debug_lazy("Sim dt: %s", lambda: env.cfg.sim.dt)
    logger.debug_lazy("Control timestep dt: %s", lambda: dt)
    logger.debug_lazy("Steering jerk: %s", lambda: j)
    logger.debug_lazy("Smooth steering reward: %s", lambda: penalty)

    return penalty


def turning_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalty for large steering angles, encouraging the reset agent to take smoother turns.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param weight: The weight of the turning penalty.
    :type weight: float

    :return: Turning penalty for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating turning penalty...")

    drive = env.action_manager._terms["drive"]
    steer = drive._processed_actions[:, 1]  # (N,)

    penalty = -1 * torch.abs(steer) / drive.max_steer # divide by max_steer to normalize

    logger.debug_lazy("Steering angles for turning penalty: %s", lambda: steer)
    logger.debug_lazy("Turning penalty: %s", lambda: penalty)

    return penalty
