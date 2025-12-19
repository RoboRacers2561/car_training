# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg

from pxr import UsdShade

from . import mdp
from car_training.tasks.manager_based.car_training.mdp.rewards import get_total_reward
from car_training.tasks.manager_based.car_training.mdp.observation import gray_image, robot_vel, track_neighborhood_image
from car_training.tasks.manager_based.car_training.mdp.termination import off_track_termination

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[6].resolve()
leatherback_usd_path = os.path.join(PROJECT_ROOT, "source/car_training/car_training/tasks/manager_based/car_training/assets/robot_assets/leatherback_simple_better.usd")
track_usd_path = os.path.join(PROJECT_ROOT, "source/car_training/car_training/tasks/manager_based/car_training/assets/scene_assets/")


## 
# Leatherback Articulation Config
##


LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=leatherback_usd_path,                       
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),),

    init_state= ArticulationCfg.InitialStateCfg(
        pos=(0.0, -3.3, 1),
        # rot=(0.0, 0.0, 0.0, 1.0),
        # lin_vel=(0.0, 0.0, 0.0),
        # ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
            # "Shock__Rear_Right": -0.03,
            # "Shock__Rear_Left":  -0.03,
            # "Shock__Front_Right": 0.03,
            # "Shock__Front_Left":  0.03,

        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel__Upright__Rear.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=1000.0,
            damping=0.0,
        ),
    },
    
)


##
# Scene definition
##


@configclass
class PlaneCfg(AssetBaseCfg):
    """Ground Plane configuration."""
    prim_path="/World/Plane"
    spawn: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg(
        size=(1000.0, 1000.0),
        visible=False,
    )


@configclass
class TrackCfg(AssetBaseCfg):
    """Per-env track configuration."""
    prim_path: str = "{ENV_REGEX_NS}/Track"
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=track_usd_path + "tracks/track_3.usd", 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
            max_depenetration_velocity=0.0,
            enable_gyroscopic_forces=False,
        ),
    )

@configclass
class OverheadCameraCfg(TiledCameraCfg):
    """Configuration for the per-env overhead camera."""
    prim_path: str = "{ENV_REGEX_NS}/OverheadCamera"
    offset: TiledCameraCfg.OffsetCfg = TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 8.66), rot=(1.57, 0.0, 0.0, -0.7071), convention="opengl")
    data_types: list = ["rgb"]
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=18.14756,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 20.0)
    )
    width: int = 512
    height: int = 512


@configclass
class CarTrainingSceneCfg(InteractiveSceneCfg):
    """Configuration for a driving scene."""

    # ground plane
    ground_plane: PlaneCfg = PlaneCfg()

    # track
    track: TrackCfg = TrackCfg()

    # robot
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # overhead camera
    overhead_camera: OverheadCameraCfg = OverheadCameraCfg()

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

    # Throttle control for rear wheels (velocity)
    throttle = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"],
        scale=1.0
    )

    # Steering control for front wheels (position)
    steering = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Knuckle__Upright__Front_Right","Knuckle__Upright__Front_Left"],
        scale=1.0
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        image = ObsTerm(func=track_neighborhood_image,
                        params={"asset_cfg": SceneEntityCfg("robot"),},
                        clip=(0.0, 255.0),)
        joint_vel_rel = ObsTerm(func=robot_vel,
                                params={"asset_cfg": SceneEntityCfg("robot")},
                                clip=(-20.0, 20.0),)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Get total reward from mdp.rewards.get_total_reward
    total_reward = RewTerm(
        func=get_total_reward,
        params={
            "camera_cfg": SceneEntityCfg("overhead_camera"),
            "robot_cfg": SceneEntityCfg("robot"),
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
    out_of_bounds = DoneTerm(func=off_track_termination, 
                             params={"asset_cfg": SceneEntityCfg("overhead_camera"), "robot_cfg": SceneEntityCfg("robot")},)



@configclass
class CommandCfg:
    """No command configuration for the MDP."""
    pass


@configclass
class CurriculumCfg:
    """Curriculum configuration for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class CarTrainingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CarTrainingSceneCfg = CarTrainingSceneCfg(num_envs=1, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # no command config
    commands: CommandCfg = CommandCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 40.0)
        # simulation settings
        self.sim.dt = 0.05
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        # assign tracks to envs
