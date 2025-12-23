from __future__ import annotations


from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import logging

import cv2
import copy
import torch
import torch.nn.functional as F
import numpy as np
from isaaclab.assets import RigidObject
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CROPPED_IMAGE_SIZE = 64

_LOGGER: Optional[logging.Logger] = None

def get_logger(env: ManagerBasedRLEnv) -> logging.Logger:
    """
    Get a child logger of the project logger for the car_training.mdp.observation module.

    :param env: The environment instance containing the base logger.
    :type env: ManagerBasedRLEnv
    
    :return: Child logger for the observation module.
    :rtype: logging.Logger
    """
    global _LOGGER # creating a cache to avoid repeated calls
    if _LOGGER is None:
        base = env.unwrapped.logger
        short_name = ".".join(__name__.split(".")[-2:]) # car_training.mdp.observation
        _LOGGER = base.getChild(short_name)
    return _LOGGER

def track_neighborhood_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Vectorized GPU crop around the car using grid_sample.

    Returns:
        crops: (N, 1, crop, crop) grayscale images
    """
    logger = get_logger(env)

    # TODO: remove attribute check - fix in env setup since this is not on gpu
    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        # use the static track image saved in the env
        cam_data = env.unwrapped.static_track_image.cpu().numpy() # type of cam_data is numpy array (num_envs, H, W, 3)
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        print("static track image not found, returning dummy gray image")
        return torch.zeros((num_envs, 1, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), device=env.unwrapped.device, dtype=torch.float32)

    device = env.unwrapped.device

    imgs = env.unwrapped.static_track_image
    N, H, W, _ = imgs.shape
    logger.debug(f"Static track image shape: number of envs={N}, H={H}, W={W}")

    # Convert to float in [0, 1] and NCHW
    if imgs.dtype != torch.float32:
        imgs = imgs.float() / 255.0
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    car_xy = robot_pos(env, asset_cfg, do_convert_coords=True)

    # Normalize to [-1, 1] for grid_sample
    cx = 2.0 * car_xy[:, 0] / (W - 1) - 1.0
    cy = 2.0 * car_xy[:, 1] / (H - 1) - 1.0
    logger.debug(f"Car positions (image coords): cx={cx}, cy={cy}")

    lin = torch.linspace(-1, 1, CROPPED_IMAGE_SIZE, device=device)
    dy, dx = torch.meshgrid(lin, lin, indexing="ij")

    # Scale offsets to image size
    dx = dx[None] * (CROPPED_IMAGE_SIZE / W)
    dy = dy[None] * (CROPPED_IMAGE_SIZE / H)
    logger.debug(f"Offset grid shape: dx={dx.shape}, dy={dy.shape}")

    # Shift grid to each car center
    grid = torch.stack(
        (
            cx[:, None, None] + dx,
            cy[:, None, None] + dy,
        ),
        dim=-1,
    )  # (N, crop, crop, 2)

    # Sample with border padding
    crops = F.grid_sample(
        imgs,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (N, 3, crop, crop)

    # Convert to grayscale
    gray_crops = 0.2989 * crops[:, 0:1, :, :] + 0.5870 * crops[:, 1:2, :, :] + 0.1140 * crops[:, 2:3, :, :]  # (N, 1, crop, crop)

    # print image shape for debugging
    # print(f"track neighborhood image shape: {gray_crops.shape}")

    # save debug image
    # debug_img = gray_crops[0, 0].cpu().numpy() * 255.0
    # cv2.imwrite("debug_track_neighborhood_image.png", debug_img)

    return gray_crops


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

    track_size = env.cfg.scene.ground_plane.ground_plane_params.each_track_size
    camera_width = env.cfg.scene.overhead_camera.camera_params.image.width
    camera_height = env.cfg.scene.overhead_camera.camera_params.image.height

    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, track_size, (camera_width, camera_height))
    return relative_pos[:, :2]  # Return x, y position
