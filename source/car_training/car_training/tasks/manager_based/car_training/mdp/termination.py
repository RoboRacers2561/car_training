from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import is_off_track, find_track_points_from_image
from car_training.tasks.manager_based.car_training.mdp.observation import robot_pos, track_neighborhood_image

def off_track_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car goes off track.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """

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
    return off_track_flags
