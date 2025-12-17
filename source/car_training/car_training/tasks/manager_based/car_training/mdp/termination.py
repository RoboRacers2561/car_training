from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import cv2
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from rewards import is_off_track, find_track_points_from_image

def off_track_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Termination function that ends an episode if the car goes off track.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:

        bool: True if the episode should terminate, False otherwise.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    
    cam_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="torch")

    off_track_flags = []
    for img in cam_data['rgb']:
        track_points_list = find_track_points_from_image(img)
        car_center = None  # TODO: Need to get the car's center position
        off_track = is_off_track(car_center, track_points_list)
        off_track_flags.append(off_track)

    return torch.tensor(off_track_flags, device=env.unwrapped.device, dtype=torch.bool)
    