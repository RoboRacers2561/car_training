from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.managers import ActionTerm

from isaaclab.managers import SceneEntityCfg, ActionTermCfg



REAR_WHEEL_NAMES = ["Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"]
FRONT_STEER_NAMES = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]

THROTTLE_SCALE = 1.0
STEER_SCALE = 1.0
MAX_STEER = 0.6  # rad

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

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        # grab articulation handle
        self.robot = env.scene[cfg.asset_name]

        # cache joint indices
        self.rear_ids, _ = self.robot.find_joints(REAR_WHEEL_NAMES)
        self.front_ids, _ = self.robot.find_joints(FRONT_STEER_NAMES)

        # scaling / limits
        self.throttle_scale = THROTTLE_SCALE
        self.steer_scale = STEER_SCALE
        self.max_steer = MAX_STEER

        # Isaac Lab expects action dim known
        self._action_dim = 2

    @property
    def action_dim(self):
        return self._action_dim

    def reset(self, env_ids=None):
        pass

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        actions: (num_envs, 2) coming from env.step(actions).
        Store raw + processed (scaled/clipped).
        """
        self._raw_actions = actions

        throttle = actions[:, 0] * self.throttle_scale
        steer = (actions[:, 1] * self.steer_scale).clamp(-self.max_steer, self.max_steer)

        # store as (num_envs, 2) again, but now processed
        self._processed_actions = torch.stack([throttle, steer], dim=-1)
    
    def apply_actions(self):
        # actions: (num_envs, 2) = [throttle, steer]
        a = self._processed_actions

        throttle = a[:, 0]
        steer = a[:, 1]

        # Rear wheel velocities (same for both rear wheels)
        rear_vel = torch.stack([throttle, throttle], dim=-1)

        # Front steering angles (same for both front joints)
        front_pos = torch.stack([steer, steer], dim=-1)  # order matches your front joint list

        # Apply
        self.robot.set_joint_velocity_target(rear_vel, joint_ids=self.rear_ids)
        self.robot.set_joint_position_target(front_pos, joint_ids=self.front_ids)
