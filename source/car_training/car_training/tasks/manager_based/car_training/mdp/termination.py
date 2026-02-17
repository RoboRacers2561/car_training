from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from car_training.tasks.manager_based.car_training.mdp.observation import aligned_track_neighborhood_image
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
    track_imgs = aligned_track_neighborhood_image(env, asset_cfg)  # (N, 1, H, W)
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


def reached_goal_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car reaches the goal.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """
    logger = get_logger(env)
    cam_width, cam_height = env.unwrapped.get_camera_dimensions()

    # get robot position and destination position and check if they are close enough
    robot_pos = env.unwrapped.current_observation[:, :2]  # (N, 2), we only care about x and y
    destination_pos = env.destination_pos  # (N, 2)
    
    # Check if agent is outside of camera view
    outside_view = (robot_pos[:, 0] < 0) | (robot_pos[:, 0] > cam_width) | (robot_pos[:, 1] < 0) | (robot_pos[:, 1] > cam_height)
    
    # Check if agent reached the goal (within threshold distance)
    distance_to_goal = torch.norm(robot_pos - destination_pos, dim=1)  # (N,)
    goal_threshold = env.unwrapped.goal_distance_threshold
    reached_goal_flags = distance_to_goal < goal_threshold

    env.unwrapped.reached_goal_flags = reached_goal_flags # store this in the env for use in rewards and logging
    
    # Combine both conditions: terminate if either out of view OR reached goal
    terminate_flags = outside_view | reached_goal_flags
    
    logger.debug_lazy("Outside view flags: %s", lambda: outside_view)
    logger.debug_lazy("Reached goal flags: %s", lambda: reached_goal_flags)
    logger.debug_lazy("Combined terminate flags: %s", lambda: terminate_flags)
    
    return terminate_flags
