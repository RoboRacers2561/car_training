from __future__ import annotations

import math
from collections.abc import Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

PORJECT_ROOT = Path(__file__).resolve().parents[8]
print(f"Project root: {PORJECT_ROOT}")
ASSET_FOLDER = Path(__file__).resolve().parents[0] / "assets"
    
import logging
import cv2
from car_training.tasks.manager_based.car_training.car_training_env_cfg import OverheadCameraCfg
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
        file_level: int | None = logging.INFO, enable_cameras: bool = True, **kwargs):
        
        self.task_name = task_name
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        self.enable_cameras = enable_cameras
        if self.enable_cameras:
            cfg.scene.overhead_camera = OverheadCameraCfg()

        super().__init__(cfg, render_mode, **kwargs)

        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset
        self.track_points = [] # will be initialized on first reset
        self.track_points_tensor = torch.empty((self.num_envs, 0, 2), device=self.device) # tensor version of track points for easier processing in rewards and termination, to be filled after reset

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
        
        return obs, info


    def __init_static_track_image_if_needed(self) -> None:
        """
        Initialize the static track image if it hasn't been initialized yet.

        This method checks if the static track image has been initialized. If not, it calls the
        `init_static_track_image` function to set it up using the overhead camera.
        
        """
        if not self._static_track_inited:
            if self.enable_cameras:
                self.static_track_image = init_static_track_image(self, camera_name="overhead_camera")
                self._static_track_inited = True
                self.logger.info("Static track image initialized.")
            else:
                # use torch.load to load pre-saved static track image
                self.static_track_image: torch.Tensor = torch.load(f"{ASSET_FOLDER}/scene_assets/static_track_image_12_env.pt").to(self.device)
                if self.num_envs <= self.static_track_image.shape[0]:
                    # only take into account the first N environments
                    self.static_track_image = self.static_track_image[:self.num_envs]
                else:
                    self.logger.warning("Number of environments is greater than the number of pre-saved static track images. Reusing images.")
                    # repeat the images to match the number of environments
                    repeats = math.ceil(self.num_envs / self.static_track_image.shape[0])
                    self.static_track_image = self.static_track_image.repeat(repeats, 1, 1, 1)[:self.num_envs]

                self._static_track_inited = True
                self.logger.info("Static track image loaded from file.")


    def __init_track_points_if_needed(self) -> None:
        """
        Initialize the track points if they haven't been initialized yet.

        This method checks if the track points have been initialized. If not, it processes the
        static track image to extract the track points.
        
        """
        if len(self.track_points) == 0:
            for index in range(self.num_envs):
                image_np = self.static_track_image[index].detach().cpu().numpy()
                track_points = find_track_points_from_image(image_np, debug=index<=30, image_idx=index)
                self.track_points.append(track_points)

                # below is needed for random resetting
                camera_width, camera_height = self.get_camera_dimensions()
                track_dimensions: tuple = self.scene.cfg.ground_plane.ground_plane_params.each_track_size
                # convert to world coordinates
                world_points = convert_coords_list(track_points, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
                self.track_points_world.append(world_points)
            
            seqs = [torch.as_tensor(tp, device=self.device, dtype=torch.float32) for tp in self.track_points]
            self.track_points_tensor = pad_sequence(seqs, batch_first=True, padding_value=0.0)
            self.track_points_lengths = torch.tensor([len(tp) for tp in self.track_points], device=self.device)
            self.logger.info("Track points initialized from static track image.")
            

    def get_camera_dimensions(self) -> tuple[int, int]:
        """
        Get the dimensions of the overhead camera.

        :return: A tuple containing the width and height of the overhead camera.
        :rtype: tuple[int, int]
        """
        if self.enable_cameras:
            camera_cfg = self.scene.cfg.overhead_camera
            return camera_cfg.width, camera_cfg.height
        else:
            return self.static_track_image.shape[2], self.static_track_image.shape[1]


