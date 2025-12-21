from isaaclab.utils import configclass


@configclass
class GroundPlaceCfg:
    size: tuple[float, float] = (1000.0, 1000.0)
    visible : bool = False
    each_track_size: tuple[float, float] = (10.0, 10.0) # Note that this is set inside the USD file directly, but we need to use it in our code. If we change the USD file such that the tracks have different size, we need to change this number. I was not able to cleanly read this size from usd file driectly. (sasan, Dec 2025)


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