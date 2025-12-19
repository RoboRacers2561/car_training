from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.managers import ActionTerm

from isaaclab.managers import SceneEntityCfg


def ackermann_to_wheel_angles(delta_center: torch.Tensor,
                             wheelbase: float,
                             trackwidth: float,
                             eps: float = 1e-6):
    """
    Convert a center steering angle (bicycle model) to left/right steering angles.
    delta_center: (N,) in radians
    returns: delta_left (N,), delta_right (N,)
    """
    # tan(delta) can blow up near pi/2; clamp a bit
    tan_delta = torch.tan(delta_center).clamp(min=-1e3, max=1e3)
    R = wheelbase / (tan_delta + eps)  # signed turning radius

    # If delta ~ 0 => R huge => left/right ~ delta_center (approximately)
    # Compute inner/outer wheel angles
    # delta_left is "inner" when turning left (delta_center > 0)
    R_abs = torch.abs(R) + eps

    # Use formulas: tan(delta_inner)=L/(R - W/2), tan(delta_outer)=L/(R + W/2)
    tan_left  = wheelbase / (R - trackwidth * 0.5 + eps)
    tan_right = wheelbase / (R + trackwidth * 0.5 + eps)

    # When turning right, the "inner" is right; but the formulas with signed R already handle it.
    delta_left  = torch.atan(tan_left)
    delta_right = torch.atan(tan_right)

    return delta_left, delta_right


class ThrottleSteerAckermannAction(ActionTerm):
    """
    Policy gives (N,2): [throttle_cmd, steer_cmd]
    We apply:
      - throttle_cmd -> both rear wheel joint velocity targets
      - steer_cmd    -> ackermann left/right -> both front steering joints position targets
    """

    def __init__(self, cfg: SceneEntityCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        # grab articulation handle
        self.robot = env.scene[cfg.name]

        # cache joint indices
        self.rear_ids = self.robot.find_joints(cfg.rear_wheel_names)
        self.front_ids = self.robot.find_joints(cfg.front_steer_names)

        # geometry (meters)
        self.wheelbase = float(cfg.wheelbase)
        self.trackwidth = float(cfg.trackwidth)

        # scaling / limits
        self.throttle_scale = float(getattr(cfg, "throttle_scale", 1.0))
        self.steer_scale = float(getattr(cfg, "steer_scale", 1.0))
        self.max_steer = float(getattr(cfg, "max_steer", 0.6))  # rad

        print(f"max steer: {self.max_steer}")

        # Isaac Lab expects action dim known
        self._action_dim = 2

    @property
    def action_dim(self):
        return self._action_dim

    def reset(self, env_ids=None):
        pass

    def apply_action(self):
        # actions shape: (num_envs, 2)
        a = self.actions

        throttle = a[:, 0] * self.throttle_scale
        steer_center = (a[:, 1] * self.steer_scale).clamp(-self.max_steer, self.max_steer)

        # Ackermann mapping
        delta_left, delta_right = ackermann_to_wheel_angles(
            steer_center, self.wheelbase, self.trackwidth
        )

        # Build target tensors
        # Rear wheel velocities: (N,2)
        rear_vel = torch.stack([throttle, throttle], dim=-1)

        # Front steer positions: (N,2) order must match your joint list order
        front_pos = torch.stack([delta_right, delta_left], dim=-1)  # adjust if your naming order differs

        # Apply to articulation
        self.robot.set_joint_velocity_target(rear_vel, joint_ids=self.rear_ids)
        self.robot.set_joint_position_target(front_pos, joint_ids=self.front_ids)
