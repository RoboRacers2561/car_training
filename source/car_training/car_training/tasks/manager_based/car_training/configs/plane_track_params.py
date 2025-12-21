from isaaclab.utils import configclass


@configclass
class GroundPlaceCfg:
    size: tuple[float, float] = (1000.0, 1000.0)
    visible : bool = True


@configclass
class TrackRigidPropsCfg:
    rigid_body_enabled: bool = True
    max_linear_velocity: float = 0.0
    max_angular_velocity: float = 0.0
    max_depenetration_velocity: float = 0.0
    enable_gyroscopic_forces: bool = False

@configclass
class TrackParamsCfg:
    """All tunable numbers for the ground plane in one place."""
    rigid_props: TrackRigidPropsCfg = TrackRigidPropsCfg()