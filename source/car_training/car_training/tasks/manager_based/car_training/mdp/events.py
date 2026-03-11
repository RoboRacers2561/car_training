from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import logging
import torch
import math

from PIL import Image
import omni.usd
from pxr import UsdGeom
from isaacsim.core.utils.prims import get_prim_at_path

from isaaclab.envs.mdp import reset_scene_to_default


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

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


def set_robot_visibility(env: ManagerBasedRLEnv, visible: bool):
    """
    Set the visibility of the robot in the simulation environment.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param visible: Whether to make the robot visible or invisible.
    :type visible: bool
    """
    logger = get_logger(env)
    robot_paths = [f"{p}/Robot" for p in env.scene.env_prim_paths]

    for p in robot_paths:
        logger.info(f"{'Showing' if visible else 'Hiding'} robot prim at path: {p}")
        prim = get_prim_at_path(p)
        if visible:
            UsdGeom.Imageable(prim).MakeVisible()
        else:
            UsdGeom.Imageable(prim).MakeInvisible()


def set_obstacle_visibility(env: ManagerBasedRLEnv, visible: bool):
    """
    Set the visibility of the obstacle in the simulation environment.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param visible: Whether to make the obstacle visible or invisible.
    :type visible: bool
    """
    logger = get_logger(env)
    obstacle_paths = [f"{p}/ObstacleCar" for p in env.scene.env_prim_paths]

    for p in obstacle_paths:
        logger.info(f"{'Showing' if visible else 'Hiding'} obstacle prim at path: {p}")
        prim = get_prim_at_path(p)
        if visible:
            UsdGeom.Imageable(prim).MakeVisible()
        else:
            UsdGeom.Imageable(prim).MakeInvisible()


def init_static_track_image(env: ManagerBasedRLEnv, camera_name: str = "overhead_camera", num_warmup_steps: int = 10):
    """
    Initialize and capture a static track image from a specified camera in the environment. This method should be called once for all. 
    The function temporarily hides the robot(s) in the environment to ensure they do not appear in the captured image.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param camera_name: The name of the camera from which to capture the static track image
    :type camera_name: str
    :param num_warmup_steps: The number of warmup steps to render before capturing
    :type num_warmup_steps: int
    :return: The captured static track image as a tensor of shape (num_envs, H, W, 3).
    :rtype: torch.Tensor
    """


    logger = get_logger(env)
    logger.info("Initializing static track image from camera: %s", camera_name)

    # hide robots so they don't appear
    set_robot_visibility(env, visible=False)
    set_obstacle_visibility(env, visible=False)

    logger.debug("Rendering simulation for %d warmup steps to capture static track image.", num_warmup_steps)
    # Render some new frames (avoiding using step)
    for _ in range(num_warmup_steps):
        # env.step(actions)
        # update scene buffers (including sensors) if available
        if hasattr(env.scene, "update"):
            env.scene.update(dt=0.0)

        # render a frame so RTX sensors/cameras refresh
        if hasattr(env.sim, "render"):
            env.sim.render()

    rgb = env.scene[camera_name].data.output["rgb"]  # (num_envs, H, W, 3)

    # for debugging: save images
    # for i in range(rgb.shape[0]):
    #     img = rgb[i].cpu().numpy()
    #     Image.fromarray(img).save(f"{camera_name}_hidden_robot_{i}.png")

    static_track_image = rgb.clone()

    # show robots back
    set_robot_visibility(env, visible=True)
    set_obstacle_visibility(env, visible=True)

    return static_track_image

def reset_robot_root_random(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
    z=0.05,
    yaw_noise_std=0.2,     # radians, e.g. 0.2 ~= 11.5 degrees
    lookahead=5,           # how many points ahead to compute tangent
    xy_rand_radius=0.05,   # random offset radius in XY plane
):
    """
    Reset the robot's root position and orientation to a random location along the track. The robot 
    is positioned at a random track point with a small random offset in the XY plane and a yaw orientation aligned 
    with the track direction, optionally with some noise.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param env_ids: The IDs of the environments to reset.
    :type env_ids: torch.Tensor
    :param asset_cfg: The configuration of the robot asset.
    :type asset_cfg: SceneEntityCfg
    :param z: The Z-coordinate (height) at which to place the robot.
    :type z: float
    :param yaw_noise_std: Standard deviation of the yaw noise to add to the robot's orientation (in radians).
    :type yaw_noise_std: float
    :param lookahead: Number of track points ahead to compute the tangent direction for yaw alignment.
    :type lookahead: int
    :param xy_rand_radius: Radius for random XY offset from the track point in meters.
    :type xy_rand_radius: float
    
    """

    logger = get_logger(env)

    robot = env.scene[asset_cfg.name]
    device = env.device
    n = env_ids.numel()

    tp_local = env.unwrapped.track_points_world  # local-to-env points (x,y)
    if len(tp_local) == 0:
        reset_scene_to_default(env, env_ids)
        return

    origins = env.scene.env_origins

    pos = torch.empty((n, 3), device=device, dtype=torch.float32)
    yaw = torch.empty((n,), device=device, dtype=torch.float32)

    for k, eid in enumerate(env_ids.tolist()):
        pts = tp_local[eid]
        if len(pts) < 2:
            raise RuntimeError("Not enough track points to reset robot position.")
        else:
            L = len(pts)
            j = int(torch.randint(0, L, (1,), device=device).item())

            # current point
            x0, y0 = pts[j]

            # lookahead point (wrap around)
            j2 = (j + lookahead) % L
            x1, y1 = pts[j2]

            # tangent direction
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)

            logger.debug("Resetting robot in env %d to track point %d (lookahead %d to point %d) with delta (%.3f, %.3f).", eid, j, lookahead, j2, dx, dy)

            # if degenerate (same point), try a different lookahead
            if dx == 0.0 and dy == 0.0:
                j2 = (j + 1) % L
                x1, y1 = pts[j2]
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)

            yaw_k = math.atan2(dy, dx)

            # add small noise around track direction so car isn't perfectly aligned
            if yaw_noise_std and yaw_noise_std > 0:
                yaw_k += float(torch.randn((), device=device) * yaw_noise_std)
            
            # 50 % of the time, flip direction (180 deg)
            if torch.rand((), device=device) < 0.5:
                yaw_k += math.pi

        logger.debug("reset position before random shift: (%.3f, %.3f).", x0, y0)
        # add random shift to x,y position within a radius
        angle = torch.rand((), device=device) * 2 * math.pi
        distance = torch.rand((), device=device) * xy_rand_radius
        shift_x = float(distance) * math.cos(float(angle))
        shift_y = float(distance) * math.sin(float(angle))
        x0 += shift_x
        y0 += shift_y
        logger.debug("Applying random shift (%.3f, %.3f) to reset position.", shift_x, shift_y)

        # local -> world
        pos[k, 0] = float(x0) + float(origins[eid, 0])
        pos[k, 1] = float(y0) + float(origins[eid, 1])
        pos[k, 2] = float(z) + float(origins[eid, 2])

        yaw[k] = yaw_k

        logger.debug("Robot reset position for env %d: (%.3f, %.3f, %.3f).", eid, pos[k,0], pos[k,1], pos[k,2])

    # yaw -> quaternion (w,x,y,z)
    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)

    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)