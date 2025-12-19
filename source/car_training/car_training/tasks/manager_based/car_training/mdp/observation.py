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
    Get the position of the car relative to the environment.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The position of the car.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins  # Relative to environment origin
    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, (10.0, 10.0), (512.0, 512.0))  # TODO: fix hardcoded values
    return relative_pos[:, :2]  # Return x, y position
