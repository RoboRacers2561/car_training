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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
import cv2

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def get_total_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, completed_laps: int, total_laps: int, max_speed: float) -> torch.Tensor:
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
    ###### Here I am blindly copying the code from the youtube video.
    asset: Articulation = env.scene[asset_cfg.name]
    
    cam_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="numpy")

    # compute the reward
    #print(f"cam_data['rgb'].shape:{cam_data['rgb'].shape, type(cam_data['rgb'])}")
    H, W, _ = cam_data['rgb'][0].shape
    #print(f"H:{H}")
    #########################################################

    # Here I think we can have access to the sensors and images for each copy of the simulation.
    # We can compute the reward for each copy separately and then combine them.

    all_rewards = []
    for frames in cam_data['rgb']:
        # for each copy of the simulation find the track points and compute the reward
        track_points_list = find_track_points_from_image(frames)
        car_center = None # TODO: Need to understand how to get the car's center position from the asset or env (potentially using a gps)
        car_speed = None # TODO: Need to understand how to get the car's speed from the asset or env (potentially using a speedometer)

        # Now compute the reward for each simulation copy
        distance_reward = distance_from_center_reward(car_center, track_points_list, weight=1.0)
        velocity_reward_value = velocity_reward(car_speed, max_speed, weight=1.0)
        alive_reward = remains_alive_reward()
        track_penalty = leaves_track_penalty(car_center, track_points_list, weight=1.0)

        total_reward = (distance_reward + velocity_reward_value + alive_reward - track_penalty)

        all_rewards.append(total_reward)

    total_reward = torch.tensor(all_rewards, dtype=torch.float32, device=env.device)
    return total_reward



#### For all of the functions below, we compute reward for only a single copy of the simulation. If we have multiple copies running in parallel, we can call the functions below repeatedly, and that is what we do above. ####

def find_track_points_from_image(image: np.ndarray):
    """
    Process the image to find track points.

    :param image: Input image from the car's camera.
    :type image: np.ndarray
    :return: List of track points as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    track_points = []
    for contour in contours:
        for point in contour:
            track_points.append((point[0][0], point[0][1]))  # (x, y) format

    return track_points


# TODO: This reward is not being used right now!
def track_completion_reward(completed_laps: int, total_laps: int, weight=1.0):
    """
    Calculate reward based on the completion of laps.

    :param completed_laps: Number of laps completed by the car.
    :type completed_laps: int
    :param total_laps: Total number of laps in the race.
    :type total_laps: int
    :param weight: Weight for the reward.
    :type weight: float
    :return: Reward based on the completion of laps.
    :rtype: float
    """
    if total_laps <= 0:
        return 0.0  # Avoid division by zero
    normalized_completion = completed_laps / total_laps
    reward = normalized_completion * weight
    return reward

def distance_from_center_reward(car_center: tuple[float, float], track_points_list: list[tuple[float, float]]):
    """
    Calculate reward based on the car's distance from the center of the track.
    
    """
    closest_track_point = __find_closest_track_point(car_center, track_points_list)
    distance = np.linalg.norm(np.array(car_center) - np.array(closest_track_point))
    # Assuming a maximum distance for normalization, e.g., the width of the track
    max_distance = 10.0  # This should be defined based on the track's characteristics
    normalized_distance = distance / max_distance
    # Reward is higher when the distance is smaller
    reward = 1.0 - normalized_distance
    return reward

def __find_closest_track_point(car_center: tuple[float, float], track_points_list: list[tuple[float, float]], weight=1.0):
    """
    Find the closest track point to the car's center.
    
    :param car_center: Tuple representing the car's center coordinates (x, y).
    :type car_center: tuple[float, float]
    :param track_points_list: List of tuples representing the track points (x, y).
    :type track_points_list: list[tuple[float, float]]
    :param weight: Weight for the reward.
    :type weight: float

    :return: The closest track point to the car's center.
    :rtype: tuple[float, float]
    """
    distances = [np.linalg.norm(np.array(car_center) - np.array(point)) for point in track_points_list]
    closest_index = np.argmin(distances)
    return track_points_list[closest_index] * weight

def velocity_reward(car_speed: float, max_speed: float, weight=1.0):
    """
    Calculates reward based on the car's speed relative to the maximum speed.

    :param car_speed: Current speed of the car.
    :type car_speed: float
    :param max_speed: Maximum speed of the car.
    :type max_speed: float
    :param weight: Weight for the reward.
    :type weight: float
    """
    normalized_speed = car_speed / max_speed
    # Reward is higher when the speed is closer to the maximum speed
    reward = normalized_speed * weight
    return reward

# TODO: Need to complete this, still do not know how 
def remains_alive_reward(weight=1.0):
    """
    Returns a constant reward for remaining alive.

    :param weight: Weight for the reward.
    :type weight: float

    Returns:
        float: Constant reward value.
    """
    return 1.0

def leaves_track_penalty(car_center: tuple[float, float], track_points_list: list[tuple[float, float]], weight=1.0):
    """
    Calculate penalty if the car leaves the track.

    :param car_center: Tuple representing the car's center coordinates (x, y).
    :type car_center: tuple[float, float]
    :param track_points_list: List of tuples representing the track points (x, y).
    :type track_points_list: list[tuple[float, float]]
    :param weight: Weight for the penalty.
    :type weight: float

    :return: Penalty value if the car leaves the track, otherwise 0.
    :rtype: float
    """
    closest_track_point = __find_closest_track_point(car_center, track_points_list, weight)
    distance = np.linalg.norm(np.array(car_center) - np.array(closest_track_point))
    # Assuming a threshold distance to determine if the car is off track
    threshold_distance = 5.0  # This should be defined based on the track's characteristics
    penalty = 0.0
    if distance > threshold_distance:
        penalty = 1.0 - (distance / threshold_distance) * weight  # Normalize the penalty
        penalty = max(penalty, 0.0)  # Ensure penalty is not negative
    else:
        penalty = 0.0  # No penalty if the car is within the track limits
    return penalty


