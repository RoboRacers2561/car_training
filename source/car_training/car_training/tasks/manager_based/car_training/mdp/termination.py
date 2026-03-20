from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from car_training.tasks.manager_based.car_training.mdp.events import reset_destination_pos
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from car_training.tasks.manager_based.car_training.mdp.observation import aligned_track_neighborhood_image, robot_pos, robot_heading, aligned_obstacle_image
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

def off_track_termination_with_obstacle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode track is not visible or obstacle is not visible.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """
    # Get track neigborhborhood image & if not black pixels, terminate
    track_imgs = aligned_track_neighborhood_image(env, asset_cfg)  # (N, 1, H, W)
    track_imgs = track_imgs.squeeze(1)  # (N, H, W)
    # Check for black pixels
    black_pixel_threshold = 0.12  # Define a threshold to consider a pixel as black
    has_track_pixel = torch.any(
        track_imgs < black_pixel_threshold,
        dim=(1, 2),
    )

    # get obstacle image
    obstacle_imgs = aligned_obstacle_image(env, asset_cfg, obstacle_cfg)  # (N, 1, H, W)
    obstacle_imgs = obstacle_imgs.squeeze(1)  # (N, H, W)
    has_obstacle_pixel = torch.any(
        obstacle_imgs < black_pixel_threshold,  # assuming obstacle pixels are bright in the image
        dim=(1, 2),
    )

    no_track_no_obstacle = ~(has_track_pixel | has_obstacle_pixel)
    return no_track_no_obstacle




def reached_goal_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car reaches the goal.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """

    robot_pos = env.unwrapped.current_observation["pos"]  # (N, 2)
    destination_pos = env.destination_pos  # (N, 2)
    
    # Check if agent reached the goal (within threshold distance)
    distance_to_goal = torch.norm(robot_pos - destination_pos, dim=1)  # (N,)
    goal_threshold = env.unwrapped.goal_distance_threshold
    inside_goal = distance_to_goal < goal_threshold

    env.unwrapped.inside_goal_flags = inside_goal.clone() # store this in the env for use in rewards and logging

    # increment/reset consecutive hold counter
    env.unwrapped.goal_hold_steps = torch.where(
        inside_goal,
        env.unwrapped.goal_hold_steps + 1,
        torch.zeros_like(env.unwrapped.goal_hold_steps),
    )

    reached_goal_flags = (
        env.unwrapped.goal_hold_steps >= env.unwrapped.goal_hold_steps_required
    )

    env.unwrapped.reached_goal_flags = reached_goal_flags # store this in the env for use in rewards and logging

    env_ids_reached_goal = torch.where(reached_goal_flags)[0]

    
    reset_destination_pos(env, env_ids_reached_goal) # reset the destination for the envs that reached the goal so that they can continue training towards the next destination in the same episode
    
    return reached_goal_flags # this is modified inside rewards

def out_of_view_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car goes out of the camera view.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """
    logger = get_logger(env)
    cam_width, cam_height = env.unwrapped.get_camera_dimensions()

    # get robot position and check if it is outside of camera view
    robot_pos = env.unwrapped.current_observation["pos"]  # (N, 2), we only care about x and y
    
    outside_view = (robot_pos[:, 0] < 0) | (robot_pos[:, 0] > cam_width) | (robot_pos[:, 1] < 0) | (robot_pos[:, 1] > cam_height)
    
    logger.debug_lazy("Outside view flags: %s", lambda: outside_view)

    env.unwrapped.out_of_bounds_flags = outside_view # store this in the env for use in rewards and logging

    return outside_view


def reached_goal_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Vectorized termination function that ends an episode if the car reaches the goal.

    Args:
        env: The environment instance.
        state: The current state of the environment.
    Returns:
        torch.Tensor: A tensor indicating which episodes should terminate.
    """

    robot_pos = env.unwrapped.current_observation["pos"]  # (N, 2)
    destination_pos = env.destination_pos  # (N, 2)
    
    # Check if agent reached the goal (within threshold distance)
    distance_to_goal = torch.norm(robot_pos - destination_pos, dim=1)  # (N,)
    goal_threshold = env.unwrapped.goal_distance_threshold
    inside_goal = distance_to_goal < goal_threshold

    env.unwrapped.inside_goal_flags = inside_goal.clone() # store this in the env for use in rewards and logging

    # increment/reset consecutive hold counter
    env.unwrapped.goal_hold_steps = torch.where(
        inside_goal,
        env.unwrapped.goal_hold_steps + 1,
        torch.zeros_like(env.unwrapped.goal_hold_steps),
    )

    reached_goal_flags = (
        env.unwrapped.goal_hold_steps >= env.unwrapped.goal_hold_steps_required
    )

    env.unwrapped.reached_goal_flags = reached_goal_flags # store this in the env for use in rewards and logging

    env_ids_reached_goal = torch.where(reached_goal_flags)[0]

    
    reset_destination_pos(env, env_ids_reached_goal) # reset the destination for the envs that reached the goal so that they can continue training towards the next destination in the same episode
    
    return reached_goal_flags # this is modified inside rewards

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

def wheel_contact_with_other_car(env, sensor_name: str, threshold: float = 1.0):
    sensor = env.scene[sensor_name]

    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"{sensor_name}: force_matrix_w is None. "
            "Make sure filter_prim_paths_expr is set."
        )

    # shape: (num_envs, num_sensor_bodies, num_filtered_bodies, 3)
    force_mag = torch.norm(force_matrix_w, dim=-1)
    # print(force_mag)

    # any contact above threshold
    collision = torch.any(force_mag > threshold, dim=(1, 2))
    return collision