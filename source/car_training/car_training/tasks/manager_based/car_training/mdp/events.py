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

    return static_track_image