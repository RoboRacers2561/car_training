from __future__ import annotations

import math
from collections.abc import Sequence
    
import logging

from isaaclab.envs import ManagerBasedRLEnv
from car_training.tasks.manager_based.car_training.car_training_env_cfg import CarTrainingEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger

class CarTrainingEnv(ManagerBasedRLEnv):
    cfg: CarTrainingEnvCfg

    def __init__(self, cfg: CarTrainingEnvCfg, render_mode: str | None = None, **kwargs):
        self.logger = setup_logger(task_name="Basic-Car-Training-v0", name="car_training", console_level=logging.INFO, file_level=logging.INFO) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions

        super().__init__(cfg, render_mode, **kwargs)
