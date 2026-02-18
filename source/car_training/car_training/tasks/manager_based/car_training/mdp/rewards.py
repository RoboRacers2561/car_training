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

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import *
from car_training.tasks.manager_based.car_training.mdp.observation import (
    robot_pos,
    aligned_robot_vel,
    robot_vel,
)
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import (
    GpuSafeLogger,
)
from car_training.tasks.manager_based.car_training.mdp.events import reset_destination_pos

_LOGGER: Optional[GpuSafeLogger] = None


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


def steering_jerk_penalty(env: ManagerBasedRLEnv, weight: float = 0.1) -> torch.Tensor:
    """
    Reward for smooth steering by penalizing large changes in steering angle.

    :param weight: Weight of the smooth steering reward.
    :type weight: float

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

    # action_now = drive.action_curr
    # if torch.isnan(action_now).any():
    #     raise ValueError(
    #         "Current action contains NaN in steering_jerk_penalty.")

    # Initialize history on first call
    # if torch.isnan(drive.action_prev).all() and torch.isnan(drive.action_prev2).all():
    #     logger.debug("Initializing action history for steering jerk penalty.")
    #     drive.action_prev = action_now.clone()
    #     drive.action_prev2 = action_now.clone()
    #     return torch.zeros((env.num_envs,), device=env.device)
    # elif torch.isnan(drive.action_prev).any() or torch.isnan(drive.action_prev2).any():
    #     # Not all NaN, but some NaN values exist which is because of reset of some envs
    #     logger.debug(
    #         "Partial NaN detected in action history; initializing missing values for steering jerk penalty."
    #     )
    #     nan_prev = torch.isnan(drive.action_prev).any(dim=1)
    #     nan_prev2 = torch.isnan(drive.action_prev2).any(dim=1)
    #     drive.action_prev[nan_prev] = action_now[nan_prev].clone()
    #     drive.action_prev2[nan_prev2] = action_now[nan_prev2].clone()

    # steer_now = action_now[:, 1]
    # steer_prev = drive.action_prev[:, 1]
    # steer_prev2 = drive.action_prev2[:, 1]
    steer_buffer = drive.action_buffer[:, :, 1]
    
    min_buffer_size = 3
    valid_envs = drive.buffer_count >= min_buffer_size
    
    if not valid_envs.any():
        # Not enough history yet, return zero penalty
        logger.debug("Insufficient buffer history for steering jerk penalty.")
        return torch.zeros((env.num_envs,), device=env.device)
    # dt is optional; keep if you want rate-invariance
    dt = env.cfg.sim.dt * env.cfg.decimation
    #j = (steer_now - 2.0 * steer_prev + steer_prev2) / (dt * dt)
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

    penalty = -weight * (j_rms)

    # logger.debug_lazy("Steering angles now: %s", lambda: steer_now)
    # logger.debug_lazy("Steering angles prev: %s", lambda: steer_prev)
    # logger.debug_lazy("Steering angles prev2: %s", lambda: steer_prev2)
    logger.debug_lazy("Steering angles prev2: %s", lambda: steer_buffer)
    logger.debug_lazy("Decimation: %s", lambda: env.cfg.decimation)
    logger.debug_lazy("Sim dt: %s", lambda: env.cfg.sim.dt)
    logger.debug_lazy("Control timestep dt: %s", lambda: dt)
    logger.debug_lazy("Steering jerk: %s", lambda: j)
    logger.debug_lazy("Smooth steering reward: %s", lambda: penalty)

    # IMPORTANT: shift history ONCE per RL step (reward is called once per RL step)
    # drive.action_prev2 = drive.action_prev.clone()
    # drive.action_prev = action_now.clone()

    return penalty

############ FOR RESET AGENT ############

def reset_agent_distance_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Total reward for the reset agent, which can be different from the main agent's reward.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot.
    :type robot_cfg: SceneEntityCfg

    :return: Total reward for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating reset agent total reward...")
    cam_width, cam_height = env.unwrapped.get_camera_dimensions()

    # max distance in the image is the diagonal
    max_distance = np.sqrt(cam_width**2 + cam_height**2)

    car_pos: torch.Tensor = env.unwrapped.current_observation[:, :2]  # (N, 2)
    destination_pos: torch.Tensor = env.unwrapped.destination_pos  # (N, 2)

    # Compute pairwise distances on GPU: (N, N)
    distances = torch.cdist(car_pos, destination_pos, p=2)
    # Distance from each car to its own destination (diagonal)
    per_env_distance = distances.diagonal()
    distance_reward =  1 - per_env_distance / max_distance  # Normalize and invert so that closer to destination is higher reward

    logger.debug_lazy("Pairwise distances shape: %s", lambda: distances.shape)
    logger.debug_lazy("Per-env distance to destination: %s", lambda: per_env_distance)
    logger.debug_lazy("Reset agent reward: %s", lambda: distance_reward)

    return distance_reward

def get_terminal_bonus(env: ManagerBasedRLEnv, bonus: float) -> torch.Tensor:
    """
    Bonus reward given at the end of an episode, for example when the agent reaches the goal.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param bonus: The bonus reward to give.
    :type bonus: float

    :return: Terminal bonus reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating terminal bonus...")

    # get robot position and destination position and check if they are close enough
    robot_pos = env.unwrapped.current_observation[:, :2]  # (N, 2), we only care about x and y
    destination_pos = env.destination_pos  # (N, 2)
    
    # Check if agent reached the goal (within threshold distance)
    distance_to_goal = torch.norm(robot_pos - destination_pos, dim=1)  # (N,)
    goal_threshold = env.unwrapped.goal_distance_threshold
    reached_goal_flags = distance_to_goal < goal_threshold

    env_ids_reached_goal = torch.where(reached_goal_flags)[0]
    logger.debug_lazy("Environments that reached the goal: %s", lambda: env_ids_reached_goal)

    reset_destination_pos(env, env_ids_reached_goal) # reset the destination for the envs that reached the goal so that they can continue training towards the next destination in the same episode

    env.unwrapped.reached_goal_flags = reached_goal_flags # store this in the env for use in rewards and logging

    # Give bonus to envs that are done
    rewards = torch.where(reached_goal_flags, torch.tensor(bonus, device=env.device), torch.zeros_like(reached_goal_flags, dtype=torch.float))

    logger.debug_lazy("Reached goal flags: %s", lambda: reached_goal_flags)
    logger.debug_lazy("Terminal bonus rewards: %s", lambda: rewards)

    return rewards


def heading_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for aligning the car's heading with the direction to the goal.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv

    :return: Heading alignment reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating heading alignment reward...")

    current_observation = env.unwrapped.current_observation

    angle_between_heading_goal = current_observation[:, -1]  # (N,) range [0 , pi]

    # compute reward based on this direction angle
    # reward is 1 when angle is 0 (perfect alignment), 0 when angle is pi (opposite direction)
    reward = 1 - angle_between_heading_goal / torch.pi  # (N,)

    logger.debug_lazy("Heading alignment rewards: %s", lambda: reward)

    return reward


