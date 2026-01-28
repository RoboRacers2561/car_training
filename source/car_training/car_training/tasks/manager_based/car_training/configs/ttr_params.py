from isaaclab.utils import configclass

@configclass
class TTRNanoRigidPropsCfg:
    rigid_body_enabled: bool = True
    max_linear_velocity: float = 1000.0
    max_angular_velocity: float = 1000.0
    max_depenetration_velocity: float = 100.0
    enable_gyroscopic_forces: bool = True


@configclass
class TTRNanoArticulationPropsCfg:
    enabled_self_collisions: bool = False
    solver_position_iteration_count: int = 4
    solver_velocity_iteration_count: int = 0
    sleep_threshold: float = 0.005
    stabilization_threshold: float = 0.001


@configclass
class TTRNanoInitStateCfg:
    pos: tuple[float, float, float] = (0.0, -3.3, 0.01) # This will also dictate the spawn of the car during NORMAL reset mode. If the size of the track changes, this needs to be updated.
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Put your joint defaults here too
    joint_pos: dict[str, float] = {
        "turn_L": 0.0,
        "turn_R": 0.0,
        "move_BL": 0.0,
        "move_BR": 0.0,
        "move_FL": 0.0,
        "move_FR": 0.0,
    }


@configclass
class TTRNanoActuatorParamsCfg:
    # throttle on the back two wheels
    throttle_effort_limit: float = 40000.0
    throttle_velocity_limit: float = 100.0
    throttle_stiffness: float = 0.0
    throttle_damping: float = 100000.0
    throttle_joint_names_expr: list[str] = ["move_.*"] # I am giving to all to the front wheels too so that wobbling is reduced

    # steering
    steering_effort_limit: float = 40000.0
    steering_velocity_limit: float = 100.0
    steering_stiffness: float = 1000.0
    steering_damping: float = 0.0
    steering_joint_names_expr: list[str] = ["turn_.*"]


# The config class below aggregates all the tunable parameters for easy access
@configclass
class TTRNanoParamsCfg:
    """All tunable numbers for the robot in one place."""
    init_state: TTRNanoInitStateCfg = TTRNanoInitStateCfg()
    rigid_props: TTRNanoRigidPropsCfg = TTRNanoRigidPropsCfg()
    articulation_props: TTRNanoArticulationPropsCfg = TTRNanoArticulationPropsCfg()
    actuators: TTRNanoActuatorParamsCfg = TTRNanoActuatorParamsCfg()