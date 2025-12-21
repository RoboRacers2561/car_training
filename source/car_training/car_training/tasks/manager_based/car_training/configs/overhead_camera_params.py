from isaaclab.utils import configclass

@configclass
class OverheadCameraOpticsParamsCfg:
    focal_length: float = 18.14756
    focus_distance: float = 400.0
    horizontal_aperture: float = 20.955
    clipping_range: tuple[float, float] = (0.01, 20.0)


@configclass
class OverheadCameraPoseParamsCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 8.66)
    rot: tuple[float, float, float, float] = (1.57, 0.0, 0.0, -0.7071)
    convention: str = "opengl"


@configclass
class OverheadCameraImageParamsCfg:
    width: int = 512
    height: int = 512
    data_types: list[str] = ["rgb"]


@configclass
class OverheadCameraParamsCfg:
    """All tunable parameters for the overhead camera."""
    prim_path: str = "{ENV_REGEX_NS}/OverheadCamera"

    optics: OverheadCameraOpticsParamsCfg = OverheadCameraOpticsParamsCfg()
    pose: OverheadCameraPoseParamsCfg = OverheadCameraPoseParamsCfg()
    image: OverheadCameraImageParamsCfg = OverheadCameraImageParamsCfg()