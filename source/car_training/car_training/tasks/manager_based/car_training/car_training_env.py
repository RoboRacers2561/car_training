from __future__ import annotations

import math
from collections.abc import Sequence
import torch
    
import logging
import cv2
from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_track_points_from_image
from isaaclab.envs import ManagerBasedRLEnv
from car_training.tasks.manager_based.car_training.car_training_env_cfg import CarTrainingEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger, GpuSafeLogger
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords_list

class CarTrainingEnv(ManagerBasedRLEnv):
    cfg: CarTrainingEnvCfg

    def __init__(self, cfg: CarTrainingEnvCfg, render_mode: str | None = None, task_name: str = "Unknown", console_level: int | None = logging.INFO,
        file_level: int | None = logging.INFO, **kwargs):
        
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        super().__init__(cfg, render_mode, **kwargs)

        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset
        self.track_points = [] # will be initialized on first reset

        self.track_points_world = [] # track points in world coordinates, to be filled after reset

        self.action_driver: ThrottleSteerAckermannAction = self.action_manager._terms["drive"] # this is a ThrottleSteerAckermannAction


    def reset(self, seed = None, env_ids = None, options = None) -> tuple[dict[str, torch.Tensor], dict]:
        """
        This is the reset method for the CarTrainingEnv class. This method is called only once at the beginning 
        of the training and is called explicitly by the user through `env.reset()`. This is not the same as the 
        reset method used at the end of each episode, which is handled internally by the environment.
        
        :param seed: Optional seed for the environment reset.
        :type seed: Optional[int]
        :param env_ids: Optional environment IDs to reset.
        :type env_ids: Optional[Sequence[int]]
        :param options: Optional additional options for the reset.
        :type options: Optional[dict]
        """

        obs, info = super().reset(seed, env_ids, options)

        # We perform the usual reset plus an initialization of the static track image if needed
        self.__init_static_track_image_if_needed()
        self.__init_track_points_if_needed()

        # disable camera sensors to save computation
        # self.scene["overhead_camera"].update_period = 5000 # update every 5000 seconds
        
        return obs, info


    def __init_static_track_image_if_needed(self) -> None:
        """
        Initialize the static track image if it hasn't been initialized yet.

        This method checks if the static track image has been initialized. If not, it calls the
        `init_static_track_image` function to set it up using the overhead camera.
        
        """
        if not self._static_track_inited:
            self.static_track_image = init_static_track_image(self, camera_name="overhead_camera")
            self._static_track_inited = True
            self.logger.info("Static track image initialized.")

    def __init_track_points_if_needed(self) -> None:
        """
        Initialize the track points if they haven't been initialized yet.

        This method checks if the track points have been initialized. If not, it processes the
        static track image to extract the track points.
        
        """
        if len(self.track_points) == 0:
            for index in range(self.num_envs):
                image_np = self.static_track_image[index].detach().cpu().numpy()
                track_points = find_track_points_from_image(image_np)
                self.track_points.append(track_points)

                # below is needed for random resetting
                camera_width = self.scene.cfg.overhead_camera.width
                camera_height = self.scene.cfg.overhead_camera.height
                track_dimensions: tuple = self.scene.cfg.ground_plane.ground_plane_params.each_track_size
                # convert to world coordinates
                world_points = convert_coords_list(track_points, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
                self.track_points_world.append(world_points)


