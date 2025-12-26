from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from car_training.tasks.manager_based.car_training.mdp.observation import track_neighborhood_image
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

def off_track_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    track_imgs = track_neighborhood_image(env, asset_cfg)  # (N, 1, H, W)
    gray_imgs = track_imgs.squeeze(1)  # (N, H, W)
    # Check for black pixels
    black_pixel_threshold = 0.12  # Define a threshold to consider a pixel as black
    has_track_pixel = torch.any(
        gray_imgs < black_pixel_threshold,
        dim=(1, 2),
    )

    off_track_flags = ~has_track_pixel
    logger.debug_lazy("Off track flags: %s", lambda: off_track_flags)
    return off_track_flags
