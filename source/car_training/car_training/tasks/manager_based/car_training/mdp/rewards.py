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

MAX_DISTANCE_FROM_LINE = 45



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


def get_total_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, max_speed: float) -> torch.Tensor:
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

    # camera: Articulation = env.scene["overhead_camera"]

    # rgb = camera.data.output['rgb']

    # rgb_cpu = rgb.detach().cpu().numpy()

    car_centers = (
        robot_pos(env, robot_cfg, do_convert_coords=True).detach().cpu().numpy()
    )
    # car_vels = robot_vel(env, robot_cfg).detach().cpu().numpy()
    car_vels = aligned_robot_vel(env, robot_cfg).detach().cpu().numpy()
    car_speeds = np.linalg.norm(car_vels, axis=1)

    num_envs = car_centers.shape[0]
    rewards = np.empty((num_envs,), dtype=np.float32)

    for i in range(num_envs):
        # for each copy of the simulation find the track points and compute the reward
        # frame = rgb_cpu[i]
        track_points_list = env.unwrapped.track_points[i]

        # draw track points for debugging
        # for point in track_points_list:
        #     cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw track points in green
        # cv2.imwrite(f"debug_image_with_track_points_{i}.png", frame)  # Save the image with track points for debugging

        car_center = car_centers[i]
        car_speed = float(car_speeds[i])
        # car_heading = robot_heading(env, robot_cfg).cpu().numpy()[index]
        logger.debug(f"car_speed type: {type(car_speed)}")
        logger.debug_lazy("car speed %s", lambda: car_speed)

        # draw the velocity vector on the image for debugging
        # start_point = (int(car_center[0]), int(car_center[1]))
        # end_point = (int(car_center[0] + car_vels[i][0]*100), int(car_center[1] + car_vels[i][1]*100))
        # cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Draw velocity vector in blue
        # end_point = (int(car_center[0] + 50 * car_heading[0]), int(car_center[1] + 50 * car_heading[1]))
        # cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Draw heading vector in blue
        # cv2.circle(frame, (int(car_center[0]), int(car_center[1])), 5, (0, 255, 0), -1)  # Draw car center in green
        # cv2.imwrite(f"debug_image_with_velocity_{i}.png", frame)  # Save the image with velocity vector for debugging

        # cropped_image = env.observation_manager.compute_group("policy")["image"]
        # debug_cropped_image = cropped_image[0, 0].cpu().numpy() * 255
        # debug_cropped_image = cv2.cvtColor(debug_cropped_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # cv2.arrowedLine(debug_cropped_image, (24, 24), (24 + int(car_vels[i][0]*30), 24 + int(car_vels[i][1]*30)), (255, 0, 0), 2)  # Draw velocity vector in blue
        # cv2.imwrite(f"debug_cropped_image_with_velocity_{i}.png", debug_cropped_image)  # Save the cropped image with velocity vector for debugging

        # cv2.circle(frame, (int(car_center[0]), int(car_center[1])), 5, (0, 0, 255), -1)  # Draw car center in red
        # cv2.imwrite(f"debug_image_with_car_center_{i}.png", frame)  # Save the image with car center for debugging
        # print(f"type of car_center: {type(car_center)}, shape: {car_center.shape}")
        # print(f"Car center: {car_center}, Car velocity: {car_velocity}, Car speed: {car_speed}")

        # Now compute the reward for each simulation copy
        distance_reward = distance_from_center_reward(
            car_center, track_points_list, weight=3.0
        )
        # print(f"Distance reward: {distance_reward}")
        velocity_reward_value = velocity_reward(
            car_speed, max_speed, weight=1.5)
        alive_reward = remains_alive_reward(weight=0.2)
        track_penalty = leaves_track_penalty(
            car_center, track_points_list, weight=0.5)

        total_reward = (
            distance_reward + velocity_reward_value + alive_reward - track_penalty
        )

        rewards[i] = total_reward
        logger.debug(
            f"Simulation {i}: Distance Reward: {distance_reward}, Velocity Reward: {velocity_reward_value}, Alive Reward: {alive_reward}, Track Penalty: {track_penalty}, Total Reward: {total_reward}"
        )

    total_reward_no_smoothing = torch.from_numpy(rewards).to(device=env.device)
    smooth_steering = steering_jerk_penalty(env, weight=0.003)
    total_reward = total_reward_no_smoothing + smooth_steering
    logger.debug_lazy("Total reward with smooth steering: %s",
                      lambda: total_reward)

    return total_reward

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

def leaves_track_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalty for leaving the track.

    :param env: The environment instance containing the track and robot information.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot to extract its position.
    :type robot_cfg: SceneEntityCfg

    :return: Track leaving penalty.
    :rtype: torch.Tensor
    """
    car_centers = robot_pos(env, robot_cfg, do_convert_coords=True)
    track_points = env.track_points_tensor  # (num_envs, num_track_points, 2)
    track_points_lengths = env.track_points_lengths  # (num_envs,)

    closest_track_points = find_closest_track_points(car_centers, track_points, track_points_lengths)  # (num_envs, 2)
    distances = torch.norm(car_centers - closest_track_points, dim=1)  #

    threshold_distance = 35
    penalty = torch.where(distances > threshold_distance, 1.0, 0.0)
    return penalty * -1

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
