from isaaclab.utils import configclass

@configclass
class LeatherbackRigidPropsCfg:
    rigid_body_enabled: bool = True
    max_linear_velocity: float = 1000.0
    max_angular_velocity: float = 1000.0
    max_depenetration_velocity: float = 100.0
    enable_gyroscopic_forces: bool = True


@configclass
class LeatherbackArticulationPropsCfg:
    enabled_self_collisions: bool = False
    solver_position_iteration_count: int = 4
    solver_velocity_iteration_count: int = 0
    sleep_threshold: float = 0.005
    stabilization_threshold: float = 0.001


@configclass
class LeatherbackInitStateCfg:
    pos: tuple[float, float, float] = (0.0, -3.3, 0.01)
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Put your joint defaults here too
    joint_pos: dict[str, float] = {
        "Wheel__Knuckle__Front_Left": 0.0,
        "Wheel__Knuckle__Front_Right": 0.0,
        "Wheel__Upright__Rear_Right": 0.0,
        "Wheel__Upright__Rear_Left": 0.0,
        "Knuckle__Upright__Front_Right": 0.0,
        "Knuckle__Upright__Front_Left": 0.0,
    }


@configclass
class LeatherbackActuatorParamsCfg:
    # throttle on the back two wheels
    throttle_effort_limit: float = 40000.0
    throttle_velocity_limit: float = 100.0
    throttle_stiffness: float = 0.0
    throttle_damping: float = 100000.0
    throttle_joint_names_expr: list[str] = ["Wheel.*"] # I am giving to to all to the front wheels too so that wobbling is reduced

    # steering
    steering_effort_limit: float = 40000.0
    steering_velocity_limit: float = 100.0
    steering_stiffness: float = 1000.0
    steering_damping: float = 0.0
    steering_joint_names_expr: list[str] = ["Knuckle__Upright__Front.*"]


# The config class below aggregates all the tunable parameters for easy access
@configclass
class LeatherbackParamsCfg:
    """All tunable numbers for the robot in one place."""
    init_state: LeatherbackInitStateCfg = LeatherbackInitStateCfg()
    rigid_props: LeatherbackRigidPropsCfg = LeatherbackRigidPropsCfg()
    articulation_props: LeatherbackArticulationPropsCfg = LeatherbackArticulationPropsCfg()
    actuators: LeatherbackActuatorParamsCfg = LeatherbackActuatorParamsCfg()
