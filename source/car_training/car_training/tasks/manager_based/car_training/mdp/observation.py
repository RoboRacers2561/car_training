from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
import copy
from math import sqrt
import numpy as np
from isaaclab.utils import convert_dict_to_backend
from isaaclab.assets import RigidObject
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# TODO: Vectorize these functions to avoid for loops over num_envs

# For CNN approach: Grayscale image observation
def gray_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Convert the RGB image from the camera sensor to grayscale. This observation space is used for the case where we want to use CNNs for processing the image input.

    :return: Grayscale image tensor. Shape: (num_envs, 1, H, W), pixel values normalized to [0, 1].
    :rtype: torch.Tensor
    """

    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        # use the static track image saved in the env
        cam_data = env.unwrapped.static_track_image.cpu().numpy() # type of cam_data is numpy array (num_envs, H, W, 3)
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        H, W = 512, 512  # TODO: fix hardcoded values
        print("static track image not found, returning dummy gray image")
        return torch.zeros((num_envs, 1, H, W), device=env.unwrapped.device, dtype=torch.float32)

    gray_img_list = []
    for img in cam_data:
        # Convert to grayscale using the luminosity method
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #cv2.imwrite("debug_gray_image.png", gray_img)  # Save the grayscale image for debugging
        gray_img_list.append([copy.copy(gray_img)])

    gray_imgs = torch.tensor(np.array(gray_img_list), device=env.unwrapped.device, dtype=torch.float32) / 255.0 # here I am normalizing the pixel values to [0, 1], which is something the youtube video does not do.

    return gray_imgs


def track_neighborhood_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get a cropped image around the car from the overhead camera.

    :return: Cropped image tensor around the car. Shape: (num_envs, 3, crop_H, crop_W), pixel values normalized to [0, 1].
    :rtype: torch.Tensor
    """

    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        # use the static track image saved in the env
        cam_data = env.unwrapped.static_track_image.cpu().numpy() # type of cam_data is numpy array (num_envs, H, W, 3)
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        H, W = 512, 512  # TODO: fix hardcoded values
        print("static track image not found, returning dummy gray image")
        return torch.zeros((num_envs, 1, H, W), device=env.unwrapped.device, dtype=torch.float32)

    # Get car positions in image coordinates
    car_positions = robot_pos(env, asset_cfg, do_convert_coords=True).cpu().numpy()
    cropped_img_list = []
    crop_size = 64  # Define the size of the cropped image around the car
    for i, img in enumerate(cam_data):
        car_x, car_y = int(car_positions[i][0]), int(car_positions[i][1])
        # Define crop boundaries
        x_min = max(car_x - crop_size // 2, 0)
        x_max = min(car_x + crop_size // 2, img.shape[1])
        y_min = max(car_y - crop_size // 2, 0)
        y_max = min(car_y + crop_size // 2, img.shape[0])
        # Crop the image around the car
        cropped_img = img[y_min:y_max, x_min:x_max, :]
        # If the cropped image is smaller than the desired size (due to boundary conditions), pad it with white pixels
        padded_img = np.ones((crop_size, crop_size, 3), dtype=cropped_img.dtype) * 255
        padded_img[:cropped_img.shape[0], :cropped_img.shape[1], :] = cropped_img
        cropped_img_list.append(padded_img)
    cropped_imgs = torch.tensor(np.array(cropped_img_list), device=env.unwrapped.device, dtype=torch.float32) / 255.0
    # Save one of the cropped images for debugging
    # cv2.imwrite("debug_cropped_image.png", (cropped_imgs[0].cpu().numpy() * 255).astype(np.uint8))
    return cropped_imgs


def robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the (x, y) velocity of the car.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The velocity vector of the car.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    relative_vel = asset.data.root_lin_vel_w[:, :2] # Return x, y velocity
    return relative_vel

def robot_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, do_convert_coords: bool = False) -> torch.Tensor:
    """
    Get the position of the car relative to the environment or convert to image coordinates.

    :param do_convert_coords: Whether to convert the coordinates to image space.
    :type do_convert_coords: bool
    :return: The (x, y) position of the car.
    :rtype: torch.Tensor
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins  # Relative to environment origin
    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, (10.0, 10.0), (512.0, 512.0))  # TODO: fix hardcoded values
    return relative_pos[:, :2]  # Return x, y position
