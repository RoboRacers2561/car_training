from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import logging

from PIL import Image
import omni.usd
from pxr import UsdGeom
from isaacsim.core.utils.prims import get_prim_at_path


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

def init_static_track_image(env: ManagerBasedRLEnv, camera_name = "overhead_camera", num_warmup_steps=10):
    logger = get_logger(env)
    logger.info("Initializing static track image from camera: %s", camera_name)


    # Initialize attribute once
    if not hasattr(env, "static_track_image"):
        env.static_track_image = None
        logger.warning("static_track_image attribute not found. Initializing to None.")
    if env.static_track_image is not None:
        return

    # hide robots so they don't appear
    set_robot_visibility(env, visible=False)

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

    # for i in range(rgb.shape[0]):
    #     img = rgb[i].cpu().numpy()
    #     Image.fromarray(img).save(f"{camera_name}_hidden_robot_{i}.png")
    env.static_track_image = rgb.clone()

    # show robots back
    set_robot_visibility(env, visible=True)