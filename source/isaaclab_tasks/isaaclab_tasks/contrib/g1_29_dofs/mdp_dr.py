# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-informed DR components for the 29-DoF G1 velocity task.

Covers the sim-to-real gaps the deployed policy exposed: actuation latency,
IMU attitude/gyro bias, ankle action chatter, and command-transition realism.
None of these change the obs/action contract — dimensions, order, and scaling
are identical to the stock task.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

##
# Actuation latency + first-order lag (obs -> action -> torque is NOT instantaneous
# on hardware: 50 Hz tick + DDS + firmware PD add up to ~a control step of delay).
##


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action with per-episode delay (0 or 1 control steps) and
    per-episode first-order lag on the applied target.

    applied += beta * (delayed_target - applied), beta resampled per episode.
    The raw action stream (and therefore the last-action observation) is
    untouched — only the target reaching the PD controller is delayed/lagged.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        shape = (self.num_envs, self.action_dim)
        dev = self.device
        self._fresh_prev = torch.zeros(shape, device=dev)
        self._applied = torch.zeros(shape, device=dev)
        self._delay = torch.zeros(self.num_envs, 1, device=dev)
        self._beta = torch.ones(self.num_envs, 1, device=dev)
        self._default_target = (
            self._offset if isinstance(self._offset, torch.Tensor) else torch.zeros(shape, device=dev)
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        fresh = self._processed_actions.clone()
        delayed = self._delay * self._fresh_prev + (1.0 - self._delay) * fresh
        self._applied += self._beta * (delayed - self._applied)
        self._fresh_prev[:] = fresh
        self._processed_actions[:] = self._applied

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        else:
            n = len(env_ids)
        default = self._default_target[env_ids] if isinstance(self._default_target, torch.Tensor) else 0.0
        self._fresh_prev[env_ids] = default
        self._applied[env_ids] = default
        lo, hi = self.cfg.lag_beta_range
        self._beta[env_ids] = torch.rand(n, 1, device=self.device) * (hi - lo) + lo
        self._delay[env_ids] = (torch.rand(n, 1, device=self.device) < self.cfg.delay_prob).float()


@configclass
class DelayedJointPositionActionCfg(mdp.JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    delay_prob: float = 0.5
    """Probability an episode runs with one control step (20 ms) of action delay."""
    lag_beta_range: tuple[float, float] = (0.5, 1.0)
    """First-order lag coefficient range; 1.0 = no lag."""


##
# IMU attitude/gyro bias: per-episode constant error on "which way is up" and on
# the measured angular velocity. Trains the policy to null the drift class that
# is otherwise trimmed by hand on hardware.
##


def _imu_buffers(env):
    if not hasattr(env, "_imu_bias_quat"):
        # identity built through the same helper that produces the biases, so the
        # quaternion component layout is always consistent with quat_apply
        zeros = torch.zeros(env.num_envs, device=env.device)
        env._imu_bias_quat = math_utils.quat_from_euler_xyz(zeros, zeros, zeros)
        env._imu_gyro_bias = torch.zeros(env.num_envs, 3, device=env.device)
    return env._imu_bias_quat, env._imu_gyro_bias


def resample_imu_bias(env, env_ids, max_tilt_deg: float = 2.5, max_gyro_bias: float = 0.05):
    """Reset-mode event: resample the per-episode IMU biases."""
    quat, gyro = _imu_buffers(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    max_tilt = max_tilt_deg * torch.pi / 180.0
    roll = (torch.rand(n, device=env.device) * 2.0 - 1.0) * max_tilt
    pitch = (torch.rand(n, device=env.device) * 2.0 - 1.0) * max_tilt
    yaw = torch.zeros(n, device=env.device)
    quat[env_ids] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    gyro[env_ids] = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * max_gyro_bias


def projected_gravity_biased(env) -> torch.Tensor:
    """Projected gravity rotated by the per-episode attitude bias."""
    quat, _ = _imu_buffers(env)
    return math_utils.quat_apply(quat, mdp.projected_gravity(env))


def base_ang_vel_biased(env) -> torch.Tensor:
    """Base angular velocity plus the per-episode constant gyro bias."""
    _, gyro = _imu_buffers(env)
    return mdp.base_ang_vel(env) + gyro


##
# Ankle-weighted action rate: the ankles carry most of the action chatter; a
# dedicated rate penalty teaches the smooth gait instead of post-hoc filtering.
##


def ankle_action_rate_l2(env) -> torch.Tensor:
    slots = getattr(env, "_ankle_action_slots", None)
    if slots is None:
        term = env.action_manager.get_term("joint_pos")
        slots = torch.tensor(
            [i for i, name in enumerate(term._joint_names) if "ankle" in name],
            device=env.device,
            dtype=torch.long,
        )
        env._ankle_action_slots = slots
    diff = env.action_manager.action[:, slots] - env.action_manager.prev_action[:, slots]
    return torch.sum(torch.square(diff), dim=1)


##
# Stand-still stillness: at near-zero command the optimum must be a settled
# double-support stance, not marching in place (r2 hardware: joint velocities
# never settled, feet alternated forever, freeze-hold 0/5). Both terms gate on
# the FULL command (linear + yaw) so turn-in-place stepping stays free.
##


def _near_zero_command(env, command_name: str, threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd, dim=1) < threshold).float()


def stand_still_feet_off_ground(
    env, command_name: str, sensor_cfg, command_threshold: float = 0.1, force_threshold: float = 1.0
) -> torch.Tensor:
    """Number of feet not in ground contact while the command is near zero.

    Marching keeps one foot perpetually airborne; penalizing airborne feet at
    zero command makes quiet double support the only free stance — this is the
    contact-switching half of the stillness objective."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    )
    return torch.sum((~in_contact).float(), dim=1) * _near_zero_command(env, command_name, command_threshold)


def stand_still_joint_vel(env, command_name: str, asset_cfg, command_threshold: float = 0.1) -> torch.Tensor:
    """L2 joint velocity [rad/s] while the command is near zero.

    The velocity half of the stillness objective: even with both feet planted,
    the joints must actually stop moving."""
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(vel), dim=1) * _near_zero_command(env, command_name, command_threshold)


##
# Slewed velocity commands: faster resampling, joystick-style ramps, and
# oversampled stop-from-backward transitions (deceleration through zero).
##


class SlewedVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command where a fraction of resamples ramp the linear
    command toward the new target at a bounded rate instead of stepping.

    Ramp rate is ``accel``; recovering from negative vx toward zero/positive
    uses ``backward_winddown_accel``. With probability
    ``stop_from_backward_prob`` an env currently commanded backward gets a
    zero-vx target, guaranteeing decelerate-through-zero practice.
    """

    cfg: SlewedVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = torch.zeros_like(self.vel_command_b)
        self._is_ramp_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        old_lin = self.vel_command_b[env_ids, :2].clone()
        super()._resample_command(env_ids)
        # oversample stop-from-backward: currently backward -> target vx = 0
        r = torch.rand(len(env_ids), device=self.device)
        stop_mask = (old_lin[:, 0] < -0.1) & (r < self.cfg.stop_from_backward_prob)
        if stop_mask.any():
            ids = (
                env_ids[stop_mask]
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.device)[stop_mask]
            )
            self.vel_command_b[ids, 0] = 0.0
        self._target[env_ids] = self.vel_command_b[env_ids].clone()
        # ramp envs keep their previous linear command and slew toward the target
        ramp = torch.rand(len(env_ids), device=self.device) < self.cfg.ramp_fraction
        self._is_ramp_env[env_ids] = ramp
        if ramp.any():
            ids = (
                env_ids[ramp] if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)[ramp]
            )
            self.vel_command_b[ids, :2] = old_lin[ramp]

    def _update_command(self):
        dt = self._env.step_dt
        active = self._is_ramp_env & ~self.is_standing_env
        if active.any():
            cur = self.vel_command_b[active, :2]
            tgt = self._target[active, :2]
            delta = tgt - cur
            rate = torch.full_like(cur, self.cfg.accel)
            winddown = (cur[:, 0] < 0.0) & (delta[:, 0] > 0.0)
            rate[winddown, 0] = self.cfg.backward_winddown_accel
            self.vel_command_b[active, :2] = cur + torch.clamp(delta, -rate * dt, rate * dt)
        super()._update_command()


@configclass
class SlewedVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = SlewedVelocityCommand
    ramp_fraction: float = 0.5
    accel: float = 1.5
    """Linear-command slew rate [m/s^2] for ramped transitions."""
    backward_winddown_accel: float = 0.25
    """Slew rate [m/s^2] when recovering from negative vx toward zero/positive."""
    stop_from_backward_prob: float = 0.3


##
# Standing task: upper-body motion driver + stillness/contact rewards + helpers.
##

from collections.abc import Sequence  # noqa: E402

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg  # noqa: E402


class UpperBodyMotionCommand(CommandTerm):
    """Drives the non-policy waist/arm joints like an external upper-body
    controller: resample random joint targets on a timer, slew toward them at a
    per-episode speed, and write PD position targets every step.

    The per-step target write also keeps these joints from driving to the zero
    pose (on this fork, joints without an action term have a zero-initialized
    position target that is never re-seeded on reset).
    """

    cfg: UpperBodyMotionCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self.robot.find_joints(cfg.joint_names)
        ids = torch.tensor(self._joint_ids, device=self.device, dtype=torch.long)
        self._default = self.robot.data.default_joint_pos.torch[:, ids].clone()
        lower = self.robot.data.joint_pos_limits_lower.torch[:, ids]
        upper = self.robot.data.joint_pos_limits_upper.torch[:, ids]
        f = cfg.range_fraction
        self._lo = self._default + f * (lower - self._default)
        self._hi = self._default + f * (upper - self._default)
        self._current = self._default.clone()
        self._target = self._default.clone()
        self._activity = torch.ones(self.num_envs, 1, device=self.device)
        self._speed = torch.ones(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Current commanded upper-body joint targets [rad]."""
        return self._current

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        a_lo, a_hi = self.cfg.activity_range
        self._activity[env_ids] = torch.rand(n, 1, device=self.device) * (a_hi - a_lo) + a_lo
        slo, shi = self.cfg.speed_range
        self._speed[env_ids] = torch.rand(n, 1, device=self.device) * (shi - slo) + slo
        # episode starts at the nominal pose (matches the robot's reset state)
        self._current[env_ids] = self._default[env_ids]
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        u = torch.rand(len(env_ids), self._lo.shape[1], device=self.device)
        raw = self._lo[env_ids] + u * (self._hi[env_ids] - self._lo[env_ids])
        self._target[env_ids] = self._default[env_ids] + self._activity[env_ids] * (raw - self._default[env_ids])

    def _update_command(self):
        step = self._speed * self._env.step_dt
        self._current += torch.clamp(self._target - self._current, -step, step)
        self.robot.set_joint_position_target_index(target=self._current, joint_ids=self._joint_ids, env_ids=None)

    def _update_metrics(self):
        pass


@configclass
class UpperBodyMotionCommandCfg(CommandTermCfg):
    class_type: type = UpperBodyMotionCommand
    asset_name: str = "robot"
    joint_names: list[str] = ["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
    resampling_time_range: tuple[float, float] = (1.5, 5.0)
    range_fraction: float = 0.7
    """Fraction of the joint-limit range (about the default pose) targets are drawn from."""
    activity_range: tuple[float, float] = (0.1, 1.0)
    """Per-episode amplitude scale: quiet vs aggressive upper-body episodes."""
    speed_range: tuple[float, float] = (0.5, 3.0)
    """Per-episode slew speed toward targets [rad/s]."""
    debug_vis: bool = False


def upper_body_command(env) -> torch.Tensor:
    """Commanded upper-body joint targets relative to defaults [rad].

    Anticipatory balance input: at training time these are the motion
    emulator's slew targets; at deploy time the upper-body stack's commanded
    joint targets. The legs can lean BEFORE an arm swing instead of reacting
    to its measured aftermath. No noise — commands are known exactly."""
    term = env.command_manager.get_term("upper_body")
    return term.command - term._default


def root_stillness_exp(env, std: float = 0.25) -> torch.Tensor:
    """exp(-(|v_lin|^2 + 0.5*|w|^2)/std^2): the standing task's main positive term."""
    asset = env.scene["robot"]
    lin = asset.data.root_lin_vel_b.torch
    ang = asset.data.root_ang_vel_b.torch
    err = torch.sum(torch.square(lin), dim=1) + 0.5 * torch.sum(torch.square(ang), dim=1)
    return torch.exp(-err / std**2)


def feet_contact_break(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Number of feet NOT in ground contact (0, 1, or 2). Penalizing this keeps
    both feet planted normally while leaving recovery steps available."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )
    return torch.sum((~in_contact).float(), dim=1)


def base_height_band(env, min_height: float, max_height: float) -> torch.Tensor:
    """Squared distance of root height outside [min_height, max_height] [m].

    A band instead of a target: standing tall and squatting are both free —
    only collapsing below the band (or hopping above it) is penalized, so
    hip-strategy CoM adjustment stays available."""
    h = env.scene["robot"].data.root_pos_w.torch[:, 2]
    below = (min_height - h).clamp(min=0.0)
    above = (h - max_height).clamp(min=0.0)
    return torch.square(below + above)


def anchor_feet(env, env_ids, settle_steps: int = 20) -> None:
    """Reset event: start the settle countdown for foot-anchor capture.

    Anchor capture is deferred ``settle_steps`` control steps past reset, for
    two reasons: inside the reset event ``body_pos_w`` can still hold pre-reset
    poses (refreshed on the next physics step, while ``reset_base`` teleports
    the robot up to ±0.5 m), and the randomized reset pose forces the PD-held
    legs to drag the feet a few cm while straightening. The robot settles
    freely during the window; then the feet are law."""
    if not hasattr(env, "_feet_anchor_countdown"):
        env._feet_anchor_countdown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._feet_anchor_countdown[env_ids] = settle_steps


def feet_displacement(
    env, threshold: float = 0.03, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
) -> torch.Tensor:
    """Terminate when either foot moves more than ``threshold`` [m] from its
    episode anchor: the no-step rule. Kills stepping, stance-widening, and yaw
    drift in one constraint — planted feet pin position and facing.

    Runs after the physics step, so poses are fresh here. Envs still inside
    their settle window are exempt; the anchor is captured when the window
    expires. This term owns the countdown mutation — it runs once per step,
    before rewards; :func:`feet_displacement_penalty` only reads."""
    robot = env.scene[asset_cfg.name]
    cur = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]
    if not hasattr(env, "_feet_anchor_xy"):
        env._feet_anchor_xy = cur.clone()
    if not hasattr(env, "_feet_anchor_countdown"):
        env._feet_anchor_countdown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cd = env._feet_anchor_countdown
    settling = cd > 0
    cd[settling] -= 1
    arm_now = settling & (cd == 0)
    if arm_now.any():
        env._feet_anchor_xy[arm_now] = cur[arm_now]
    disp = (cur - env._feet_anchor_xy).norm(dim=-1).max(dim=1)[0]
    return (disp > threshold) & ~settling


def feet_displacement_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
) -> torch.Tensor:
    """Dense drift-from-anchor cost [m] (max over feet; 0 during the settle
    window). The learnable path to "feet planted": the hard termination alone
    is a cliff random exploration cannot survive, so it gives no gradient."""
    if not hasattr(env, "_feet_anchor_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    cur = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]
    disp = (cur - env._feet_anchor_xy).norm(dim=-1).max(dim=1)[0]
    return disp * (env._feet_anchor_countdown == 0).float()


def hold_joints_at_default(env, env_ids, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Write PD position targets = default joint pos for the given (non-actioned)
    joints. On this fork, joints without an action term keep position target 0.0
    — they silently drive to the ZERO pose, not ``init_state.joint_pos``. One
    write per reset persists for everything the action manager never touches."""
    robot = env.scene[robot_cfg.name]
    targets = robot.data.default_joint_pos.torch[env_ids][:, robot_cfg.joint_ids]
    robot.set_joint_position_target_index(target=targets, joint_ids=robot_cfg.joint_ids, env_ids=env_ids)
