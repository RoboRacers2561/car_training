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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# For CNN approach: Grayscale image observation
def gray_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Convert the RGB image from the camera sensor to grayscale. This observation space is used for the case where we want to use CNNs for processing the image input.

    :return: Grayscale image tensor. Shape: (num_envs, 1, H, W), pixel values normalized to [0, 1].
    :rtype: torch.Tensor
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    cam_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="numpy")

    gray_img_list = []
    for img in cam_data['rgb']:
        # Convert to grayscale using the luminosity method
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img_list.append([copy.copy(gray_img)])

    gray_imgs = torch.tensor(np.array(gray_img_list), device=env.unwrapped.device, dtype=torch.float32) / 255.0 # here I am normalizing the pixel values to [0, 1], which is something the youtube video does not do.

    return gray_imgs

def get_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the linear velocity of the car.

    :return: Linear velocity tensor. Shape: (num_envs, 3).
    :rtype: torch.Tensor
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # vel_data = convert_dict_to_backend(
    #             {k: v for k, v in asset.data.output.items()}, backend="torch")

    # linear_vel_list = []
    # for vel in vel_data['linear_velocity']:
    #     linear_vel_list.append(copy.copy(vel))

    # linear_vels = torch.tensor(np.array(linear_vel_list), device=env.unwrapped.device, dtype=torch.float32)

    # dummy for now
    linear_vels = torch.zeros((env.num_envs, 2), device=env.unwrapped.device, dtype=torch.float32)

    return linear_vels

def get_car_center(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the position of the car's center.

    :return: Position tensor. Shape: (num_envs, 3).
    :rtype: torch.Tensor
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    pos_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="torch")

    position_list = []
    for pos in pos_data['position']:
        position_list.append(copy.copy(pos))

    positions = torch.tensor(np.array(position_list), device=env.unwrapped.device, dtype=torch.float32)

    return positions