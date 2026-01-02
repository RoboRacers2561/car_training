from __future__ import annotations

import math
from collections.abc import Sequence
    
import logging

from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image
from isaaclab.envs import ManagerBasedRLEnv
from car_training.tasks.manager_based.car_training.car_training_env_cfg import CarTrainingEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger, GpuSafeLogger

class CarTrainingEnv(ManagerBasedRLEnv):
    cfg: CarTrainingEnvCfg

    def __init__(self, cfg: CarTrainingEnvCfg, render_mode: str | None = None, task_name: str = "Unknown", console_level: int | None = logging.INFO,
        file_level: int | None = logging.INFO, **kwargs):
        
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        super().__init__(cfg, render_mode, **kwargs)

        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset


    def reset(self, seed = None, env_ids = None, options = None):
        obs, info = super().reset(seed, env_ids, options)

        # We perform the usual reset plus an initialization of the static track image if needed
        self.__init_static_track_image_if_needed()
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

