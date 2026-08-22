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


class object_goal_progress_on_table(ManagerTermBase):
    """Slide-task ratchet: 1.0 per ``min_improvement`` [m] of NEW episode-best
    planar goal distance, credited only upright, on the table, at push speed.

    The absolute kernel pays a parked mug whenever goals land near spawn; the
    ratchet pays nothing until the mug actually gains ground, and ground given
    back cannot be re-earned. Bar re-seeds when the command resamples.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._prev_cmd: torch.Tensor | None = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best[env_ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_improvement: float = 0.005,
        z_max: float = 0.06,
        min_up_cos: float = 0.87,
        max_speed: float = float("inf"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        obj = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        error = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1).nan_to_num(
            nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        quat = obj.data.root_quat_w.torch
        up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
        z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        gate = (up_z > min_up_cos) & (z_local < z_max) & _object_calm(env, max_speed, object_cfg)
        cmd = command[:, :3]
        if self._prev_cmd is None:
            self._prev_cmd = cmd.clone()
        else:
            self.best[(self._prev_cmd != cmd).any(dim=1)] = float("inf")
            self._prev_cmd.copy_(cmd)
        unseeded = torch.isinf(self.best)
        self.best[unseeded] = error[unseeded]
        improved = gate & (error < self.best - min_improvement)
        self.best[improved] = error[improved]
        return improved.float()


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
