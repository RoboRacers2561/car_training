from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from car_training.tasks.manager_based.car_training.mdp.observation import aligned_track_neighborhood_image, robot_pos, robot_heading
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

def off_track_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, middle_only = False) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car goes off track.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """
    logger = get_logger(env)

    # Get track neigborhborhood image & if not black pixels, terminate
    track_imgs = aligned_track_neighborhood_image(env, asset_cfg)  # (N, 1, H, W)
    gray_imgs = track_imgs.squeeze(1)  # (N, H, W)
    # Check for black pixels
    black_pixel_threshold = 0.12  # Define a threshold to consider a pixel as black

    # If middle_only flag is set, we can focus on a smaller region in the center of the image to determine if the car is off track, which can reduce false positives from shadows or other non-track elements at the edges of the image.
    if middle_only:
        h, w = gray_imgs.shape[1], gray_imgs.shape[2]
        center_h, center_w = h // 2, w // 2
        crop_size = min(h, w) // 4  # Focus on the central quarter of the image
        gray_imgs = gray_imgs[:, center_h - crop_size:center_h + crop_size, center_w - crop_size:center_w + crop_size]
    has_track_pixel = torch.any(
        gray_imgs < black_pixel_threshold,
        dim=(1, 2),
    )

    off_track_flags = ~has_track_pixel
    logger.debug_lazy("Off track flags: %s", lambda: off_track_flags)
    return off_track_flags

def collision_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, obstacle_cfg: SceneEntityCfg, collision_threshold: float = 20.0) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car 'collides' (is too close) with another object.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """
    logger = get_logger(env)

    # Placeholder implementation
    robot_positions = robot_pos(env, asset_cfg, do_convert_coords=True)  # (N, 2)
    obstacle_positions = robot_pos(env, obstacle_cfg, do_convert_coords=True)  # (N, 2)

    # Compute distance between robot and obstacle
    distances = torch.norm(robot_positions - obstacle_positions, dim=1)  # (N,)

    collision_flags = distances < collision_threshold
    logger.debug_lazy("Collision flags: %s", lambda: collision_flags)
    return collision_flags