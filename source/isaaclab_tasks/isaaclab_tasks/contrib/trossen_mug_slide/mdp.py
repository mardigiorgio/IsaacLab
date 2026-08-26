# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slide-task MDP terms for the Trossen mug slide.

Extracted verbatim from the mug-lift task's mdp module at the task split:
the slide's terms live with the slide, implemented against the stable
generic layer only (``isaaclab.envs.mdp`` plus the ``.torch`` views of the
warp-backed data buffers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finite(t: torch.Tensor) -> torch.Tensor:
    """Zero non-finite entries.

    Rewards are computed BEFORE terminations in the manager step, so the step
    in which a world diverges would hand PPO a non-finite reward even though
    the speed/non-finite termination resets that world in the same step. A
    diverging world therefore contributes a finite zero for its single dying
    step; the event itself stays measured via the termination counters.
    """
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


class arm_action_jerk_l2(ManagerTermBase):
    """NaN-safe penalty on the CHANGE of the arm action delta (jerk).

    Action-rate taxes any fast command; jerk taxes direction reversals and
    twitching specifically, leaving a smooth sustained push unpenalized. Arm
    dims only — the binary gripper flip is exploration, not jitter. Buffers
    reset per episode so the first two steps of an episode are never charged.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.action_manager.get_term("arm_action").action_dim
        self._prev_delta = torch.zeros(env.num_envs, n, device=env.device)
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_delta[env_ids] = 0.0
        self._steps[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, term_name: str = "arm_action") -> torch.Tensor:
        n = self._prev_delta.shape[1]
        delta = env.action_manager.action[:, :n] - env.action_manager.prev_action[:, :n]
        jerk = torch.sum(torch.square(delta - self._prev_delta), dim=1)
        # steps 0 and 1 have no meaningful delta history
        jerk = torch.where(self._steps >= 2, jerk, torch.zeros_like(jerk))
        self._prev_delta.copy_(delta)
        self._steps += 1
        return _finite(jerk)


def arm_action_rate_l2(env: ManagerBasedRLEnv, term_name: str = "arm_action") -> torch.Tensor:
    """NaN-safe action-rate penalty over the ARM action dims only.

    The binary gripper command is exempt: an open/close flip is exploration of
    grasp timing, not jitter, and taxing it teaches the policy not to try
    grasping. Relies on the action manager concatenating terms in declaration
    order with the arm term declared first.
    """
    n = env.action_manager.get_term(term_name).action_dim
    delta = env.action_manager.action[:, :n] - env.action_manager.prev_action[:, :n]
    return _finite(torch.sum(torch.square(delta), dim=1))


def _sensor_force_mag(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Total filtered contact-force magnitude per env for a shape-filtered sensor.

    Order-invariant reduction over sensor bodies and filtered shapes (the
    resolved shape order is not stable across builds), NaN-safe like every
    reward input in this task.
    """
    forces = env.scene.sensors[sensor_name].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    net = forces.torch.sum(dim=2)
    mag = torch.linalg.vector_norm(net, dim=-1)
    return mag.reshape(env.num_envs, -1).sum(dim=-1).nan_to_num(0.0)


def body_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while any arm link presses the mug's BODY (wall/base) pieces."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def object_off_table(
    env: ManagerBasedRLEnv,
    x_bound: float,
    y_bound: float,
    z_bound: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose object left the workspace (env-local xy footprint or height).

    xy bounds catch objects knocked sideways onto the floor. The height bound
    catches near-vertical contact ejections, which keep the object inside the
    xy footprint for the whole flight and would otherwise coast until timeout
    while feeding the physics error norm ever-larger coordinates.
    """
    obj = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w.torch[:, :3] - env.scene.env_origins[:, :3]
    return (pos_local[:, 0].abs() > x_bound) | (pos_local[:, 1].abs() > y_bound) | (pos_local[:, 2] > z_bound)


def robot_state_abnormal(
    env: ManagerBasedRLEnv,
    max_joint_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate envs whose ROBOT state is non-finite or implausibly fast.

    The object-side speed valve cannot see an arm-side blowup: a constraint
    explosion through a grasped/hooked object drives the low-inertia wrist
    joints non-finite, and joint state feeds the policy observations directly.
    Same containment contract as :func:`object_speed_exceeds`, robot side.
    """
    asset = env.scene[asset_cfg.name]
    qpos = asset.data.joint_pos.torch
    qvel = asset.data.joint_vel.torch
    finite = torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1)
    return (qvel.abs().amax(dim=1) > max_joint_vel) | ~finite


def physics_diverged(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate envs whose world latched the adaptive solver's divergence flag.

    A latched world (NaN/divergent state, or an inner solve still failing at
    the dt floor under SAP containment) held its last committed FINITE state
    while its clock skipped to the boundary: the solver refused to commit a
    step it could not converge. Every state-space valve above therefore sees a
    plausible frozen scene and stays silent, while actions no longer influence
    the state -- the episode is broken by construction. The manager preserves
    the transient solver latch in a per-env mask that survives until read
    here; terminating hands the env to the reset machinery, which clears the
    mask for the reset envs. All-False when no adaptive Newton solver is
    active (nothing can latch).
    """
    try:
        from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
    except ImportError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mask = NewtonMJWarpManager.get_diverged_env_mask()
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask[: env.num_envs] != 0


def _object_calm(env: ManagerBasedRLEnv, max_speed: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bool per env: the object moves at carry speed, not ballistically.

    ``inf`` disables the gate. A deliberate carry stays well under any sane
    bound; a flick crosses it by an order of magnitude, so gating income on
    this removes the fling channel's revenue without taxing transport.
    """
    if max_speed == float("inf"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    obj = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1)
    return torch.isfinite(speed) & (speed < max_speed)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    kernel: str = "tanh",
) -> torch.Tensor:
    """Reach shaping on |object - ee|, using the ee_frame sensor's first target.

    ``kernel="tanh"`` (default) is ``1 - tanh(d/std)``; ``kernel="gaussian"`` is
    ``exp(-(d/std)^2)``. The Gaussian is sharper near the target and decays far
    faster far from it -- at d = 3*std it pays 1.2e-4 against tanh's 5e-3, so a
    std sized for tanh starves a Gaussian of gradient at approach distances.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    if kernel == "gaussian":
        return _finite(torch.exp(-((distance / std) ** 2)))
    return _finite(1.0 - torch.tanh(distance / std))


def object_goal_distance_on_table(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    z_max: float = 0.06,
    min_up_cos: float = 0.87,
    max_speed: float = float("inf"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Slide-task transport shaping: pays while the mug approaches the
    commanded position UPRIGHT, ON the table, at push speed.

    The upright gate (default 30 degrees) and table-height gate make a tipped
    or airborne mug worthless — sliding A to B without tipping IS the task —
    and the speed gate closes the smack-it-across-the-table channel the same
    way the lift task's carry gate does. Planar (xy) distance only: the
    commanded z is not the mug's to control while it stays on the slab.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    distance = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
    quat = obj.data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
    z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    gate = (up_z > min_up_cos) & (z_local < z_max) & _object_calm(env, max_speed, object_cfg)
    return _finite(gate * (1.0 - torch.tanh(distance / std)))


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root position expressed in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return object_pos_b


# ---------------------------------------------------------------------------- slidev2: moving goal

from isaaclab.utils.configclass import configclass  # noqa: E402

# Tracking band [m]: the slide's success currency is the fraction of the
# trajectory the mug spends inside this planar radius of the CURRENT goal.
TRACK_BAND = 0.025


class MovingGoalCommand(UniformPoseCommand):  # noqa: F405
    """Pose command that follows a per-episode randomized trajectory program
    from the fixed start to a sampled endpoint, pacing the slide.

    Per env and episode the program samples: endpoint (via cfg ranges — the
    direction randomization), glide speed, start delay (phase), an optional
    pause window, and an optional reversal window. The observed command
    follows the program; income through the tracking terms exists only near
    the CURRENT goal, so the policy must follow whatever pace and shape the
    program takes — parking, smacking, and endpoint-camping all decouple
    from income by construction.

    Logged metrics: ``position_error`` (object to CURRENT goal),
    ``orientation_error`` (tilt), ``in_band`` (mean fraction of envs inside
    the tracking band — its iteration mean IS time-in-band), and the sticky
    endpoint success (auxiliary; NOT the task's success definition).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object = env.scene["object"]
        self._step_dt = env.step_dt
        n = self.num_envs
        dev = self.device
        self._start = torch.tensor(cfg.start_pos, device=dev).expand(n, 3).contiguous()
        self._end = torch.zeros(n, 3, device=dev)
        self._dir = torch.zeros(n, 2, device=dev)
        self._length = torch.ones(n, device=dev)
        self._speed = torch.zeros(n, device=dev)
        self._t0 = torch.zeros(n, device=dev)
        self._pause0 = torch.full((n,), 1e9, device=dev)
        self._pause1 = torch.full((n,), 1e9, device=dev)
        self._rev0 = torch.full((n,), 1e9, device=dev)
        self._rev1 = torch.full((n,), 1e9, device=dev)
        self._t = torch.zeros(n, device=dev)
        self._s = torch.zeros(n, device=dev)
        self.goal_vel = torch.zeros(n, 2, device=dev)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._end[env_ids] = self.pose_command_b[env_ids, :3].clone()
        delta = self._end[env_ids, :2] - self._start[env_ids, :2]
        length = torch.linalg.vector_norm(delta, dim=1).clamp(min=1e-6)
        self._length[env_ids] = length
        self._dir[env_ids] = delta / length.unsqueeze(1)
        n = len(env_ids) if not isinstance(env_ids, slice) else self.num_envs
        dev = self.device
        r = self.cfg
        self._speed[env_ids] = r.speed_range[0] + (r.speed_range[1] - r.speed_range[0]) * torch.rand(n, device=dev)
        self._t0[env_ids] = r.start_delay_range[0] + (r.start_delay_range[1] - r.start_delay_range[0]) * torch.rand(
            n, device=dev
        )
        # Optional pause and reversal windows, each with independent
        # probability; a window of zero length is a no-op.
        travel = self._length[env_ids] / self._speed[env_ids]
        pause_on = (torch.rand(n, device=dev) < r.pause_prob).float()
        p0 = self._t0[env_ids] + torch.rand(n, device=dev) * travel
        plen = pause_on * (r.pause_len_range[0] + (r.pause_len_range[1] - r.pause_len_range[0]) * torch.rand(n, device=dev))
        self._pause0[env_ids] = p0
        self._pause1[env_ids] = p0 + plen
        rev_on = (torch.rand(n, device=dev) < r.reversal_prob).float()
        r0 = self._t0[env_ids] + torch.rand(n, device=dev) * travel
        rlen = rev_on * (r.reversal_len_range[0] + (r.reversal_len_range[1] - r.reversal_len_range[0]) * torch.rand(n, device=dev))
        self._rev0[env_ids] = r0
        self._rev1[env_ids] = r0 + rlen
        self._t[env_ids] = 0.0
        self._s[env_ids] = 0.0
        self.pose_command_b[env_ids, :3] = self._start[env_ids]

    def _update_command(self):
        super()._update_command()
        self._t += self._step_dt
        moving = self._t >= self._t0
        paused = (self._t >= self._pause0) & (self._t < self._pause1)
        reversing = (self._t >= self._rev0) & (self._t < self._rev1)
        rate = self._speed * moving.float() * (~paused).float() * torch.where(reversing, -1.0, 1.0)
        self._s = (self._s + rate * self._step_dt).clamp(min=0.0)
        done = self._s >= self._length
        self._s = torch.minimum(self._s, self._length)
        rate = torch.where(done | ~moving | paused, torch.zeros_like(rate), rate)
        self.pose_command_b[:, :2] = self._start[:, :2] + self._dir * self._s.unsqueeze(1)
        self.goal_vel = self._dir * rate.unsqueeze(1)

    def _compute_error(self):
        # World-frame command refresh: the debug marker reads pose_command_w,
        # so this is what makes the goal visibly follow its program.
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # Error against the CURRENT goal; the band metric shares it.
        position_error = torch.linalg.vector_norm(
            self.pose_command_w[:, :3] - self._object.data.root_pos_w.torch, dim=1
        )
        band = torch.linalg.vector_norm(
            self.pose_command_w[:, :2] - self._object.data.root_pos_w.torch[:, :2], dim=1
        ) < TRACK_BAND
        self.metrics["in_band"] = band.float()
        quat = self._object.data.root_quat_w.torch
        up = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
        tilt = torch.acos(up.clamp(-1.0, 1.0))
        return position_error, tilt


@configclass  # noqa: F405
class MovingPoseCommandCfg(UniformPoseCommandCfg):  # noqa: F405
    """Cfg for :class:`MovingGoalCommand`: ranges sample the ENDPOINT; the
    trajectory program starts at ``start_pos`` and is randomized per episode."""

    class_type: type = MovingGoalCommand
    start_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_range: tuple[float, float] = (0.05, 0.12)
    start_delay_range: tuple[float, float] = (0.0, 1.0)
    pause_prob: float = 0.5
    pause_len_range: tuple[float, float] = (0.3, 1.5)
    reversal_prob: float = 0.3
    reversal_len_range: tuple[float, float] = (0.3, 0.8)


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> MovingGoalCommand:
    return env.command_manager.get_term(command_name)


def goal_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The current goal's planar velocity [m/s] — the pacing signal. A policy
    cannot match a pace it cannot observe."""
    return _finite(_cmd(env, command_name).goal_vel)


def goal_velocity_matching(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    z_max: float = 0.06,
    min_up_cos: float = 0.87,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Pays for matching the goal's planar velocity, upright and on the
    table: following the pace, not just standing near the path."""
    obj = env.scene[object_cfg.name]
    dv = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch[:, :2] - _cmd(env, command_name).goal_vel, dim=1)
    quat = obj.data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
    z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    gate = (up_z > min_up_cos) & (z_local < z_max)
    return _finite(gate * (1.0 - torch.tanh(dv / std)))


class object_goal_progress_clipped(ManagerTermBase):
    """Clipped potential-based progress toward the CURRENT goal: the per-step
    change of planar goal distance, clipped to ``max_step`` and normalized to
    [-1, 1]. Clipping bounds what a fling can earn in one step; the
    potential-based form makes camping worthless and regression negative."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = torch.full((env.num_envs,), float("nan"), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev[env_ids] = float("nan")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        max_step: float = 0.01,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        cmd = _cmd(env, command_name)
        dist = torch.linalg.vector_norm(
            cmd.pose_command_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1
        ).nan_to_num(nan=float("nan"))
        delta = (self._prev - dist).nan_to_num(nan=0.0)
        self._prev = dist
        return _finite(delta.clamp(-max_step, max_step) / max_step)


class in_band_sustained(ManagerTermBase):
    """The band annuity: pays inside the tracking band, scaled up by how long
    the mug has STAYED inside (consecutive steps / ``ramp_steps``, capped at
    1) — sustained tracking earns the full rate, a drive-by earns a sliver.
    Gated upright and on the table like every income term."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        ramp_steps: int = 30,
        z_max: float = 0.06,
        min_up_cos: float = 0.87,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        cmd = _cmd(env, command_name)
        dist = torch.linalg.vector_norm(cmd.pose_command_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
        quat = obj.data.root_quat_w.torch
        up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
        z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        inside = (dist < TRACK_BAND) & (up_z > min_up_cos) & (z_local < z_max)
        self._count = torch.where(inside, self._count + 1.0, torch.zeros_like(self._count))
        return _finite(inside.float() * (self._count / ramp_steps).clamp(max=1.0))
