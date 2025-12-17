# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from mdp.rewards import get_total_reward
from mdp.observation import gray_img, get_velocity
from mdp.termination import out_of_bounds

import os
usd_path = os.path.join(os.path.dirname(__file__), "../../../assets/robot_assets/leatherback.usd")


## 
# Leatherback Articulation Config
##


LEATHERBACK_CFG = ArticulationCfg(
    usd_path=usd_path,
    prim_path="{ENV_REGEX_NS}/Robot",

    init_state= ArticulationCfg.InitState(
        pos=(0.0, 0.0, 0.0),
        ori=(0.0, 0.0, 0.0, 1.0),
        linear_vel=(0.0, 0.0, 0.0),
        angular_vel=(0.0, 0.0, 0.0),
    ),
    actuators= ArticulationCfg.Actuators(
        # Rear wheels - throttle
        joint_effort=ArticulationCfg.Actuators.JointEffort(
            joint_names=["rear_left_wheel", "rear_right_wheel"],
            scale=1.0,
        ),
        # Front wheels - steering
        joint_position=ArticulationCfg.Actuators.JointPosition(
            joint_names=["front_left_steering", "front_right_steering"],
            scale=1.0,
        ), 
    ),
)


##
# Scene definition
##


@configclass
class TrackCfg(sim_utils.AssetBaseCfg):
    """Per-env ground tile with texture."""

    prim_path: str = "{ENV_REGEX_NS}/track"

    spawn: sim_utils.PrimitiveCfg = sim_utils.PrimitiveCfg(
        prim_type="Plane",
        size=(10.0, 10.0),
        collision=True,
        visual=True,
    )


@configclass
class CameraCfg(AssetBaseCfg):
    """Configuration for the per-env overhead camera."""
    prim_path: str = "{ENV_REGEX_NS}/OverheadCamera"
    spawn: sim_utils.CameraCfg = sim_utils.CameraCfg(
        pos=(0.0, 0.0, 30.0),
        ori=(0.7071, 0.0, 0.0, 0.7071),  # looking down
        horizontal_fov=math.radians(90.0),
        resolution=(256, 256),
        enable_rgb=True,
        enable_depth=False,
        enable_segmentation=False,
    )


@configclass
class CarTrainingSceneCfg(InteractiveSceneCfg):
    """Configuration for a driving scene."""

    # track
    track: TrackCfg = TrackCfg()

    # robot
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # overhead camera
    overhead_camera: CameraCfg = CameraCfg()

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["rear_left_wheel", "rear_right_wheel"], scale=100.0) # TODO: convert to steering/throttle


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        image = ObsTerm(func=mdp.gray_img,
                        params={"asset_cfg": SceneEntityCfg("tiled_camera")},
                        clip=(0.0, 255.0),)
        joint_vel_rel = ObsTerm(func=mdp.get_velocity)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm( # TODO: Convert to resetting the car on the track at start pos and 0 velocity
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Get total reward from mdp.rewards.get_total_reward
    total_reward = RewTerm(
        func=get_total_reward,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "completed_laps": 0,  # TODO: Replace with actual completed laps
            "total_laps": 3,      # TODO: Replace with actual total laps
            "max_speed": 10.0,    # TODO: Replace with actual max speed
        },
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Out of bounds termination
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds, out_of_bounds=True)



##
# Environment configuration
##


@configclass
class CarTrainingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CarTrainingSceneCfg = CarTrainingSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.05
        self.sim.render_interval = self.decimation