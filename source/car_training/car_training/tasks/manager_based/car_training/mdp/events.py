from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
import torch
import omni.usd
from pxr import UsdGeom
from isaacsim.core.utils.prims import get_prim_at_path


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def set_robot_visibility(env, visible: bool):
    """
    Set the visibility of the robot in the simulation environment.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param visible: Whether to make the robot visible or invisible.
    :type visible: bool
    """
    robot_paths = [f"{p}/Robot" for p in env.scene.env_prim_paths]

    for p in robot_paths:
        print(f"{'Showing' if visible else 'Hiding'} robot prim at path: {p}")
        prim = get_prim_at_path(p)
        if visible:
            UsdGeom.Imageable(prim).MakeVisible()
        else:
            UsdGeom.Imageable(prim).MakeInvisible()

def init_static_track_image(env, camera_name = "overhead_camera", num_warmup_steps=10):


    # Initialize attribute once
    if not hasattr(env, "static_track_image"):
        env.static_track_image = None
    if env.static_track_image is not None:
        return

    # hide robots so they don't appear
    set_robot_visibility(env, visible=False)

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

    for i in range(rgb.shape[0]):
        img = rgb[i].cpu().numpy()
        Image.fromarray(img).save(f"{camera_name}_hidden_robot_{i}.png")
    env.static_track_image = rgb.clone()

    # show robots back
    set_robot_visibility(env, visible=True)