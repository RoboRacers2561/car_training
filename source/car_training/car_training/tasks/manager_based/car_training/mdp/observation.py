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
import torchvision.transforms.functional as TF
import numpy as np
from isaaclab.assets import RigidObject
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords, rotate_torch_image
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import GpuSafeLogger
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CROPPED_IMAGE_SIZE = 64
FRONT_STEER_NAMES = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]


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


def robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the (x, y) velocity of the car.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The velocity vector of the car.
    """
    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]

    relative_vel = asset.data.root_com_lin_vel_w[:, :2] # Return x, y velocity

    logger.debug_lazy("Car velocities: %s", lambda: relative_vel)
    return relative_vel

def robot_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:

    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_heading = asset.data.heading_w

    logger.debug_lazy("Car heading: %s", lambda: robot_heading)

    # make a unit vector
    robot_heading = torch.stack((torch.cos(robot_heading), -torch.sin(robot_heading)), dim=-1) # negative y to match image coords
    return robot_heading

def robot_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, do_convert_coords: bool = False) -> torch.Tensor:
    """
    Get the position of the car relative to the environment or convert to image coordinates.

    :param do_convert_coords: Whether to convert the coordinates to image space.
    :type do_convert_coords: bool
    :return: The (x, y) position of the car.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    asset: RigidObject = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins  # Relative to environment origin
    logger.debug_lazy("Car world positions: %s", lambda: relative_pos)

    track_size = env.cfg.scene.ground_plane.ground_plane_params.each_track_size
    camera_width = env.cfg.scene.overhead_camera.camera_params.image.width
    camera_height = env.cfg.scene.overhead_camera.camera_params.image.height
    logger.debug(f"Track size: {track_size}, Camera width: {camera_width}, Camera height: {camera_height}")

    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, track_size, (camera_width, camera_height))
        logger.debug_lazy("Car image positions: %s", lambda: relative_pos)
    return relative_pos[:, :2]  # Return x, y position

def forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the forward speed of the car.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The forward speed of the car.
    """
    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]

    heading = asset.data.heading_w  # (num_envs,)
    lin_vel = asset.data.root_com_lin_vel_w  # (num_envs, 3)

    heading_vector = torch.stack((torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=-1)  # (num_envs, 3)

    forward_speed = torch.sum(lin_vel * heading_vector, dim=-1)  # (num_envs,)
    logger.debug_lazy("Car forward speeds: %s", lambda: forward_speed)
    # print speed of env 0 for debugging
    print(f"Car forward speed env 0: {forward_speed[0].item()}")
    return forward_speed



def aligned_track_neighborhood_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the track neighborhood image rotated and aligned with the car's heading.
    The car's heading will always point towards the top of the image.

    Returns:
        torch.Tensor: Aligned grayscale images of shape (N, 1, crop, crop).
    """
    logger = get_logger(env)

    # TODO: remove attribute check - fix in env setup since this is not on gpu
    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        pass
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        logger.warning("static track image not found, returning dummy gray image")
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
    logger.debug_lazy("Car positions (image coords): %s", lambda: car_xy)

    # Normalize to [-1, 1] for grid_sample
    cx = 2.0 * car_xy[:, 0] / (W - 1) - 1.0
    cy = 2.0 * car_xy[:, 1] / (H - 1) - 1.0

    lin = torch.linspace(-1, 1, CROPPED_IMAGE_SIZE, device=device)
    dy, dx = torch.meshgrid(lin, lin, indexing="ij")

    # Scale offsets to image size
    dx = dx[None] * (CROPPED_IMAGE_SIZE / W)
    dy = dy[None] * (CROPPED_IMAGE_SIZE / H)
    logger.debug(f"Crop offsets dx shape: {dx.shape}, dy shape: {dy.shape}")

    # Shift grid to each car center
    grid = torch.stack(
        (
            cx[:, None, None] + dx,
            cy[:, None, None] + dy,
        ),
        dim=-1,
    )  # (N, crop, crop, 2)

    # Get car headings and compute rotation angles
    headings = asset_heading = robot_heading(env, asset_cfg)  # (N, 2)
    angles = torch.atan2(-headings[:, 0], headings[:, 1]) + torch.pi  # (N,) Negative x to match image coords

    cos = torch.cos(angles).view(-1, 1, 1, 1)  # (N,1,1,1)
    sin = torch.sin(angles).view(-1, 1, 1, 1)  # (N,1,1,1)

    center = torch.stack((cx, cy), dim=-1).view(-1, 1, 1, 2)  # (N,1,1,2)
    rel = grid - center  # (N,H,W,2)
    rel_x = rel[..., 0:1]
    rel_y = rel[..., 1:2]

    rot_x = cos * rel_x - sin * rel_y
    rot_y = sin * rel_x + cos * rel_y

    grid = torch.cat((rot_x, rot_y), dim=-1) + center  # (N,H,W,2)

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

    gray_crops = (gray_crops > 0.5).float()
    # log the shape of gray crops

    logger.debug("Obtained track neighborhood images.")
    logger.debug_lazy("Track neighborhood image shape: %s", lambda: list(gray_crops.shape))

    logger.debug_lazy("Track neighborhood image min/max: %s", lambda: list((gray_crops.min().item(), gray_crops.max().item())))

    # print image shape for debugging
    # print(f"track neighborhood image shape: {gray_crops.shape}")

    #save debug image
    # debug_img = gray_crops[0, 0].cpu().numpy() * 255
    # cv2.imwrite("debug_track_neighborhood_image.png", debug_img)

    return gray_crops

def aligned_robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the car's velocity in the car's local frame, aligned with its heading.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The velocity vector of the car in its local frame.
    """
    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]

    relative_vel = asset.data.root_com_lin_vel_w[:, :2]  # (N, 2)

    heading = asset.data.heading_w  # (N,)

    cos_heading = torch.cos(-heading)  # (N,)
    sin_heading = torch.sin(-heading)  # (N,)

    rot_matrix = torch.stack(
        [
            torch.stack([cos_heading, -sin_heading], dim=-1),
            torch.stack([sin_heading, cos_heading], dim=-1),
        ],
        dim=-2,
    )  # (N, 2, 2)

    local_vel = torch.bmm(rot_matrix, relative_vel.unsqueeze(-1)).squeeze(-1)  # (N, 2)

    logger.debug_lazy("Car local frame velocities: %s", lambda: local_vel)
    return local_vel

def front_wheels_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get the front wheels steering angle of the car.

    
    """
    logger = get_logger(env)
    # robot: Articulation = env.scene[asset_cfg.asset_name]
    robot = env.unwrapped.scene["robot"]

    front_ids, _ = robot.find_joints(FRONT_STEER_NAMES)

    front_right_angle = robot.data.joint_pos[:, front_ids][:, 0] # Num_envs x 1
    front_left_angle = robot.data.joint_pos[:, front_ids][:, 1] # Num_envs x 1

    logger.debug_lazy("Front right wheel angle: %s", lambda: front_right_angle)
    logger.debug_lazy("Front left wheel angle: %s", lambda: front_left_angle)

    return front_right_angle, front_left_angle
