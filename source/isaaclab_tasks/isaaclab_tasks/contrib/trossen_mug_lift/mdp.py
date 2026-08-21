# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classic Franka-cube-lift MDP terms for the Trossen mug task.

The four functions are the reference lift task's reach / lift / goal-track terms and
the object-position observation, implemented against the stable generic layer only
(``isaaclab.envs.mdp`` plus the ``.torch`` views of the warp-backed data buffers), so
this task does not depend on the evolving ``core.lift`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, sample_uniform, subtract_frame_transforms

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


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NaN-safe action-rate penalty (see :func:`_finite`)."""
    return _finite(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1))


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


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NaN-safe action-magnitude penalty (see :func:`_finite`)."""
    return _finite(torch.sum(torch.square(env.action_manager.action), dim=1))


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """NaN-safe joint-velocity penalty (see :func:`_finite`)."""
    asset = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1))


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


def handle_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while the finger pads press the mug's HANDLE pieces."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def body_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while any arm link presses the mug's BODY (wall/base) pieces."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def pad_contact_bipolar(
    env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1, off_scale: float = 0.025
) -> torch.Tensor:
    """+1 while the pads press the object, ``-off_scale`` while they do not.

    The asymmetry is deliberate: under strict terminations a symmetric
    penalty is a suicide pump — bleeding -w for a full episode costs more
    than the -10 terminal price of batting the mug away, so the policy ends
    the bleed by killing the object. ``off_scale`` is sized so a full episode
    of never-touching (150 steps) still costs LESS than one failure
    termination.
    """
    c = (_sensor_force_mag(env, sensor_name) > threshold).float()
    return _finite(c - off_scale * (1.0 - c))


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


def object_speed_exceeds(
    env: ManagerBasedRLEnv,
    max_linear_speed: float,
    max_angular_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose object exceeds any physically reachable speed.

    MJWarp does not consume the authored body velocity clamps (Newton drops
    ``velocity_limit_sim`` when constructing the solver model), so the required
    speed bound is enforced here, per the Newton asset-migration guide: a
    diverging contact event is terminated at the first control step it exceeds
    speeds no legitimate trajectory reaches, before the state can escalate to
    non-finite. Non-finite velocities terminate unconditionally (NaN compares
    false, so the isfinite clause cannot be folded into the thresholds).
    """
    obj = env.scene[object_cfg.name]
    lin = torch.norm(obj.data.root_lin_vel_w.torch, dim=1)
    ang = torch.norm(obj.data.root_ang_vel_w.torch, dim=1)
    finite = torch.isfinite(lin) & torch.isfinite(ang)
    return (lin > max_linear_speed) | (ang > max_angular_speed) | ~finite


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


# Handle center in the mug body frame (SDF handle_middle frame).
_HANDLE_OFFSET_B = (0.062, 0.0, 0.058)


def handle_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """World position of the mug's handle center."""
    obj = env.scene[object_cfg.name]
    offset = torch.tensor(_HANDLE_OFFSET_B, device=env.device).expand(env.num_envs, 3)
    return obj.data.root_pos_w.torch + quat_apply(obj.data.root_quat_w.torch, offset)


def handle_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    ee_body_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_name: str | None = None,
) -> torch.Tensor:
    """Reach shaping toward the HANDLE, alignment-weighted.

    Distance alone is a point condition any wrist drape can satisfy, so the
    shaping is multiplied by how well the gripper's approach axis (the EE
    link's +x, through the fingers) points at the handle: parked poses with
    the TCP near the handle but the fingers aimed elsewhere earn nothing."""
    robot = env.scene[robot_cfg.name]
    idx = robot.body_names.index(ee_body_name)
    link_pos = robot.data.body_pos_w.torch[:, idx]
    link_quat = robot.data.body_quat_w.torch[:, idx]
    axis = quat_apply(link_quat, torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(env.num_envs, 3))
    handle = handle_pos_w(env, object_cfg)
    to_handle = handle - link_pos
    align = torch.clamp((axis * to_handle).sum(dim=1) / to_handle.norm(dim=1).clamp_min(1e-6), min=0.0)
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(handle - ee_pos_w, dim=1)
    shaped = (1.0 - torch.tanh(distance / std)) * align
    if contact_sensor_name is not None:
        # Contact counts as distance zero: while the pads press the handle the
        # term pays its maximum, so attempting and holding contact strictly
        # dominates hovering at the shaping optimum. Without this, the dense
        # reach income competes against the sparse touch it should lead into,
        # and the hover optimum wins.
        touching = handle_grasped(env, contact_sensor_name)
        shaped = torch.maximum(shaped, touching)
    return _finite(shaped)


def handle_grasped(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while the finger pads carry force on the HANDLE pieces — the strict
    by-the-handle possession predicate (proximity alone does not qualify)."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def lift_progress_by_handle(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    sensor_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense lift progress, payable only while the handle is gripped.

    The lift indicator is a step at ``target_height``: the first centimeters of
    a lift pay nothing, so no gradient leads a holding policy upward. This term
    pays the PATH — height above rest, normalized to the target — under the
    same handle-grip gate as the lift income itself.
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2]
    progress = torch.clamp((z - rest_height) / (target_height - rest_height), min=0.0, max=1.0)
    return _finite(progress * handle_grasped(env, sensor_name))


def object_lifted_by_handle(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    sensor_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Lift income payable only while the handle is grasped."""
    return object_is_lifted(env, minimal_height, object_cfg) * handle_grasped(env, sensor_name)


def object_goal_distance_by_handle(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal income payable only while the handle is grasped."""
    return object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg) * handle_grasped(
        env, sensor_name
    )


class dropped_after_lift(ManagerTermBase):
    """Terminate envs where a LIFTED object has come back down.

    A per-env latch records that the object cleared ``lift_height`` at any
    point in the episode; once latched, the object returning below
    ``drop_height`` ends the episode. Losing a lift is terminal, while never
    lifting is judged by the other terms — so the gate never punishes
    pre-grasp exploration.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._was_lifted[:] = False
        else:
            self._was_lifted[env_ids] = False
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height: float,
        drop_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        z = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2]
        finite = torch.isfinite(z)
        self._was_lifted |= finite & (z > lift_height)
        return self._was_lifted & finite & (z < drop_height)


def object_tipped(
    env: ManagerBasedRLEnv,
    min_up_cos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object has tipped past the spill threshold.

    The mug's body frame is Z-up; the tilt is the angle between its body +Z
    and world +Z, read as the z-component of the rotated up axis:
    ``1 - 2(x^2 + y^2)``. The Newton data API returns quaternions in
    (x, y, z, w) order (Warp's layout), so x, y are indices 0, 1 — indexing
    this wxyz-style false-fired on the mug's authored 90-degree yaw and
    insta-terminated every episode. ``min_up_cos`` 0.5 = leaning past 60
    degrees — knocked onto its side or carried spilling, either way the
    episode is over. Fires only on finite state so a diverged world is judged
    by the containment valves, not here.
    """
    quat = env.scene[object_cfg.name].data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
    return torch.isfinite(up_z) & (up_z < min_up_cos)


def mug_on_side(
    env: ManagerBasedRLEnv,
    min_up_cos: float = 0.5,
    z_max: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 while the mug rests on its WALL at table height — the knocked-over
    state, penalizable without touching legitimate lifts.

    A one-wall pinch tilts a held mug, so tilt alone must not be punished;
    the conjunction with table height is what distinguishes "carried at an
    angle" (airborne, no penalty) from "tipped onto the table" (leaning past
    ``min_up_cos`` with the root still low — resting on wall or rim). Finite
    inputs only, like every reward input in this task.
    """
    obj = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
    z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    finite = torch.isfinite(up_z) & torch.isfinite(z_local)
    return _finite((finite & (up_z < min_up_cos) & (z_local < z_max)).float())


def object_knocked_from_spot(
    env: ManagerBasedRLEnv,
    xy_tol: float,
    z_max: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object sits at table height OUTSIDE its spawn box.

    A mug that is at table height (env-local z below ``z_max``) but displaced
    more than ``xy_tol`` from its authored spawn was batted, dragged, or
    tipped-and-rolled there — a clean carry leaves the table vertically from
    the spawn spot. Complements ``object_tipped``: this fires on knocked-AWAY
    even when the mug stays upright. Finite-only, like every valve.
    """
    obj = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w.torch[:, :3] - env.scene.env_origins[:, :3]
    disp = pos_local[:, :2] - obj.data.default_root_pose.torch[:, :2]
    finite = torch.isfinite(pos_local).all(dim=-1)
    return finite & (pos_local[:, 2] < z_max) & (disp.abs().amax(dim=-1) > xy_tol)


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


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_speed: float = float("inf"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 when the object root is above ``minimal_height`` [m] (world z),
    moving no faster than ``max_speed`` [m/s] (see :func:`_object_calm`)."""
    obj = env.scene[object_cfg.name]
    lifted = obj.data.root_pos_w.torch[:, 2] > minimal_height
    return _finite((lifted & _object_calm(env, max_speed, object_cfg)).float())


def object_vertical_velocity_shaped(
    env: ManagerBasedRLEnv,
    up_scale: float,
    down_scale: float,
    clamp: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward rising, penalize falling harder than rising pays.

    A knock produces a rise then a fall of comparable magnitude; with
    ``down_scale > up_scale`` the round trip nets negative, so only a
    sustained hold (velocity near zero once aloft) is profitable -- a knock
    can no longer pay for itself the way a flat height threshold does.
    Clamped so one violent ejection cannot dominate the batch-mean advantage.
    """
    obj = env.scene[object_cfg.name]
    vz = torch.clamp(obj.data.root_lin_vel_w.torch[:, 2], min=-clamp, max=clamp)
    return _finite(torch.where(vz > 0.0, up_scale * vz, down_scale * vz))


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


def _mug_rim_decomposition(
    env: ManagerBasedRLEnv,
    rim_height: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TCP position and its axial/planar split about the mug's rim circle.

    Returns ``(tcp, center, axis, axial, planar_vec)`` in world frame, where
    ``center`` is the rim-circle center, ``axis`` the mug's unit axis, and
    ``tcp - center == axial * axis + planar_vec``.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    tcp = ee_frame.data.target_pos_w.torch[..., 0, :]
    root = obj.data.root_pos_w.torch
    quat = obj.data.root_quat_w.torch
    axis = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=root.device).expand_as(root))
    center = root + rim_height * axis
    d = tcp - center
    axial = (d * axis).sum(dim=-1)
    planar_vec = d - axial.unsqueeze(-1) * axis
    return tcp, center, axis, axial, planar_vec


def mug_rim_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    rim_height: float,
    rim_radius: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    kernel: str = "tanh",
) -> torch.Tensor:
    """Reach shaping on the TCP's distance to the NEAREST point of the mug's
    rim circle — the one-wall pinch target (one pad outside, one inside the
    opening).

    The root sits at the mug's bottom center, which the TCP can never reach
    without penetrating the mug or the slab; every rim point IS reachable, so
    the gradient's optimum is a graspable pose. The nearest-point form is
    indifferent to which side the approach comes from and follows the mug's
    orientation if it tips. Distance to a circle in 3D is the closed form
    sqrt((|d_planar| - R)^2 + d_axial^2), exact on the axis as well.
    """
    _, _, _, axial, planar_vec = _mug_rim_decomposition(env, rim_height, object_cfg, ee_frame_cfg)
    planar = torch.linalg.vector_norm(planar_vec, dim=-1)
    distance = torch.sqrt((planar - rim_radius) ** 2 + axial**2)
    shaped = torch.exp(-((distance / std) ** 2)) if kernel == "gaussian" else 1.0 - torch.tanh(distance / std)
    return _finite(shaped)


def mug_rim_look_at(
    env: ManagerBasedRLEnv,
    rim_height: float,
    rim_radius: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the finger axis pointing AT the nearest rim point.

    The finger axis is the TCP frame's local +X (``EE_TCP_OFFSET`` is authored
    along link-6 x). Reward is ``(1 + cos)/2`` between that axis and the
    TCP-to-rim-point direction, so approaching the lip nose-first pays and
    approaching side-on or backwards does not. Near-degenerate geometry (TCP
    on the mug axis, or already at the rim point) yields a direction of zero
    length; those envs take the neutral 0.5 rather than a spurious extreme.
    """
    tcp, center, _, _, planar_vec = _mug_rim_decomposition(env, rim_height, object_cfg, ee_frame_cfg)
    planar = torch.linalg.vector_norm(planar_vec, dim=-1, keepdim=True)
    rim_pt = center + rim_radius * planar_vec / planar.clamp_min(1e-6)
    to_rim = rim_pt - tcp
    to_rim_norm = torch.linalg.vector_norm(to_rim, dim=-1, keepdim=True)
    ee_quat = env.scene[ee_frame_cfg.name].data.target_quat_w.torch[..., 0, :]
    finger_axis = quat_apply(ee_quat, torch.tensor([1.0, 0.0, 0.0], device=tcp.device).expand_as(tcp))
    cos = (finger_axis * to_rim).sum(dim=-1) / to_rim_norm.squeeze(-1).clamp_min(1e-6)
    degenerate = (planar.squeeze(-1) < 1e-5) | (to_rim_norm.squeeze(-1) < 1e-5)
    return _finite(torch.where(degenerate, torch.full_like(cos, 0.5), 0.5 * (1.0 + cos)))


def close_near_rim(
    env: ManagerBasedRLEnv,
    dist_threshold: float,
    rim_height: float,
    rim_radius: float,
    action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1.0 while the gripper COMMANDS close with the TCP within
    ``dist_threshold`` [m] of the nearest rim point.

    The binary close channel earns nothing anywhere else in the reward, so
    exploration on that dim dies once its mean drifts open and sigma hits the
    floor. This pays for the action itself, exactly where closing is correct:
    no contact is required (not a touch bonus) and closing away from the rim
    pays nothing (not spammable).
    """
    tcp, _, _, axial, planar_vec = _mug_rim_decomposition(env, rim_height, object_cfg, ee_frame_cfg)
    planar = torch.linalg.vector_norm(planar_vec, dim=-1)
    distance = torch.sqrt((planar - rim_radius) ** 2 + axial**2)
    closing = torch.any(env.action_manager.get_term(action_name).raw_actions < 0.0, dim=1)
    return _finite((closing & (distance < dist_threshold)).float())


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    max_speed: float = float("inf"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    kernel: str = "tanh",
) -> torch.Tensor:
    """Goal-tracking shaping, gated on the object being lifted and moving at
    carry speed (see :func:`_object_calm` — a flicked object passing the goal
    ballistically earns nothing).

    ``kernel``: ``"tanh"`` (default) or ``"gaussian"`` = ``exp(-(d/std)^2)``.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    shaped = torch.exp(-((distance / std) ** 2)) if kernel == "gaussian" else 1.0 - torch.tanh(distance / std)
    lifted_calm = (obj.data.root_pos_w.torch[:, 2] > minimal_height) & _object_calm(env, max_speed, object_cfg)
    return _finite(lifted_calm * shaped)


def object_held(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1.0 when the object rides within ``threshold`` [m] of the TCP.

    Possession predicate for a parallel-jaw gripper: a grasped object moves
    with the TCP, a batted one separates within a step, so gating income on
    this indicator makes ballistic strategies unpayable.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    return torch.where(distance < threshold, 1.0, 0.0)


def object_lifted_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Lift indicator payable only while the object is possessed."""
    return object_is_lifted(env, minimal_height, object_cfg) * object_held(env, threshold, object_cfg, ee_frame_cfg)


def object_goal_distance_held(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Goal-tracking income payable only while the object is possessed."""
    return object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg) * object_held(
        env, threshold, object_cfg, ee_frame_cfg
    )


def pregrasp_pose_match(
    env: ManagerBasedRLEnv,
    pose: dict[str, float],
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense kernel on the joint-space distance to THE pre-grasp configuration.

    One authored demonstration defines the whole approach gradient: from
    anywhere in the workspace the optimum is exactly the configuration whose
    next action is the close command. Joint space, so no frames and no
    object-geometry assumptions enter.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids, resolved = asset.find_joints(list(pose.keys()), preserve_order=True)
    target = torch.tensor([pose[n] for n in resolved], device=env.device, dtype=torch.float32)
    q = asset.data.joint_pos.torch[:, joint_ids]
    distance = torch.linalg.vector_norm(q - target, dim=-1)
    return _finite(1.0 - torch.tanh(distance / std))


def mug_disturbed_ungrasped(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    contact_threshold: float = 0.5,
    speed_clamp: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Mug speed while NOT held in an opposed grasp — the knock signal.

    Bleeds for any mug motion the arm causes before it actually holds the
    mug (approach knocks, brushes, bats); gates off entirely once both pads
    press, so the grasp and carry are penalty-free. Hovering is neither
    rewarded nor punished by this term.
    """
    obj = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1).clamp(max=speed_clamp)
    held = opposed_grasp(env, sensor_name, contact_threshold)
    return _finite(torch.where(held, torch.zeros_like(speed), speed))


def reset_arm_reverse_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose: dict[str, float],
    bank_fraction: float,
    noise: float,
    alpha_min: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reverse-curriculum start states: interpolate home -> pre-grasp.

    Selected envs start at ``q = home + alpha * (pose - home)`` with
    ``alpha ~ U(alpha_min, 1)``: at ``alpha_min = 1`` every bank start is the
    pre-grasp itself; as the curriculum lowers ``alpha_min`` toward 0 the
    start distribution grows back along the approach path until home starts
    dominate — the Florensa reverse curriculum on our own two anchors.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids, resolved = asset.find_joints(list(pose.keys()), preserve_order=True)
    # The interpolation anchor must be the AUTHORED home. default_joint_pos is
    # retargeted per episode below (so that zero action HOLDS each start pose),
    # which makes the live tensor unusable as the anchor — cache it before the
    # first retarget can touch it.
    if not hasattr(env, "_bank_home_anchor"):
        env._bank_home_anchor = asset.data.default_joint_pos.torch[0:1, joint_ids].clone()
    home = env._bank_home_anchor.expand(env_ids.shape[0], -1)
    sel = torch.rand(env_ids.shape[0], device=env.device) < bank_fraction
    target = torch.tensor([pose[n] for n in resolved], device=env.device, dtype=torch.float32)
    alpha = sample_uniform(alpha_min, 1.0, (env_ids.shape[0], 1), env.device)
    start = home + alpha * (target.unsqueeze(0) - home)
    start = start + sample_uniform(-noise, noise, start.shape, start.device)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids[:, None], joint_ids]
    start = torch.where(sel.unsqueeze(1), start.clamp(limits[..., 0], limits[..., 1]), home)
    # Zero action must HOLD the start pose: with a home-anchored offset the PD
    # rips a banked arm back toward home on the first steps — through the mug,
    # for an engaged pre-grasp. The arm action term's offset is a CLONE of
    # default_joint_pos taken at construction, so the retarget must be written
    # into the term's own tensor; writes to the data tensor never reach the PD
    # path. Home envs get home written back, undoing any earlier retarget.
    # Requires ABSOLUTE joint_pos observations: relative-to-default obs would
    # alias across start poses.
    term = env.action_manager.get_term("arm_action")
    if not hasattr(env, "_bank_offset_cols"):
        env._bank_offset_cols = torch.tensor(
            [resolved.index(n) for n in term._joint_names], device=env.device
        )
    term._offset[env_ids] = start[:, env._bank_offset_cols]
    asset.write_joint_position_to_sim_index(position=start, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(start), joint_ids=joint_ids, env_ids=env_ids)


def anneal_reverse_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    start_step: int = 0,
    end_step: int = 240_000,
    event_name: str = "reset_arm_grasp_bank",
):
    """Linearly grow the reverse curriculum's reach back toward home.

    Lowers the reset event's ``alpha_min`` from 1 to 0 between
    ``start_step`` and ``end_step`` env steps, then leaves it at 0 (bank
    starts sampled along the ENTIRE approach path). Step-scheduled: simple,
    deterministic, and resumable.
    """
    step = int(env.common_step_counter)
    frac = min(max((step - start_step) / max(end_step - start_step, 1), 0.0), 1.0)
    term = env.event_manager.get_term_cfg(event_name)
    term.params["alpha_min"] = 1.0 - frac
    return torch.tensor(term.params["alpha_min"])


def reset_arm_to_grasp_bank(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose: dict[str, float],
    bank_fraction: float,
    noise: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start a fraction of the resetting envs with the arm posed at the object;
    the rest keep the home pose the default reset already wrote.

    Runs after the scene reset, overwriting only the joints named in ``pose``
    for the selected envs. The gripper carriages are not in ``pose``: they keep
    the default open state, so closing the grasp stays the policy's own first
    action. Which envs draw a bank start re-randomizes every reset, so every
    env sees both the straddle regime and the full approach from home. The
    action term's position targets stay referenced to the default pose, so a
    bank start begins with a home-ward PD pull the policy must learn to
    counter-hold — the bank teaches holding near the object, not a frozen
    start state.
    """
    asset = env.scene[asset_cfg.name]
    sel = torch.rand(env_ids.shape[0], device=env.device) < bank_fraction
    if not sel.any():
        return
    ids = env_ids[sel]
    joint_ids, resolved_names = asset.find_joints(list(pose.keys()), preserve_order=True)
    target = torch.tensor([pose[name] for name in resolved_names], device=env.device, dtype=torch.float32)
    joint_pos = target.unsqueeze(0).expand(ids.shape[0], -1).clone()
    joint_pos += sample_uniform(-noise, noise, joint_pos.shape, joint_pos.device)
    limits = asset.data.soft_joint_pos_limits.torch[ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=ids)


def _pad_force_mags(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Per-pad filtered contact-force magnitude, shape ``(num_envs, num_pads)``.

    Reduces over the filtered shapes only (their resolved order is not stable
    across builds), keeping the sensor's body axis so opposed-grasp logic can
    see each jaw separately. NaN-safe like every reward input in this task.
    """
    forces = env.scene.sensors[sensor_name].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    net = forces.torch.sum(dim=2)
    return torch.linalg.vector_norm(net, dim=-1).nan_to_num(0.0)


def mug_grasped(env: ManagerBasedRLEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    """Reward form of the opposed-grasp gate: 1 while both pads clamp the mug.

    The rung between hovering and airborne income: closing on the mug must pay
    BEFORE any lift succeeds, or the close-and-hold that precedes every lift is
    a reward dead zone the policy has no gradient into."""
    return opposed_grasp(env, sensor_name, threshold).float()


def opposed_grasp(env: ManagerBasedRLEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    """Bool per env: EVERY pad presses the object above ``threshold`` [N].

    The parallel-jaw form of the dexsuite thumb-plus-opposing-finger gate: a
    one-jaw push or brush cannot open it; only a closed clamp with the object
    between both pads can.
    """
    return (_pad_force_mags(env, sensor_name) > threshold).all(dim=1)


def fingers_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    sensor_name: str,
    contact_threshold: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Approach flow (dexsuite ``object_ee_distance`` shape): tanh kernel on the
    WORST pad-to-object distance, paid at 10% without an opposed grasp and in
    full with one.

    The max over pads means both jaws must straddle the object for the kernel
    to saturate, and the wide ``std`` keeps a usable gradient across the whole
    workspace instead of dying at spawn distance.
    """
    asset = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    pad_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    distance = torch.linalg.vector_norm(pad_pos_w - obj.data.root_pos_w.torch[:, None, :], dim=-1).max(dim=-1).values
    scale = torch.where(opposed_grasp(env, sensor_name, contact_threshold), 1.0, 0.1)
    return _finite((1.0 - torch.tanh(distance / std)) * scale)


class position_command_progress(ManagerTermBase):
    """Pay 1.0 per ``min_improvement`` [m] of NEW best object-to-goal distance,
    creditable only while the object is held in an opposed grasp.

    The bar is the smallest credited error this episode. It never retreats, so
    ground given back cannot be re-earned by backing off and re-approaching,
    and gains made while not holding (a batted flight) move nothing and pay
    nothing. Seeded from the first step's error so holding the spawn pose
    earns nothing; re-seeded whenever the command resamples, so credit never
    carries across goals.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best_error = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._prev_command: torch.Tensor | None = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best_error[env_ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_name: str,
        contact_threshold: float = 0.1,
        min_improvement: float = 0.0025,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        obj = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        # A non-finite state must not be creditable NOR allowed to poison the
        # bar; +inf does neither.
        error = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1).nan_to_num(
            nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        cmd = command[:, :3]
        if self._prev_command is None:
            self._prev_command = cmd.clone()
        else:
            self.best_error[(self._prev_command != cmd).any(dim=1)] = float("inf")
            self._prev_command.copy_(cmd)
        unseeded = torch.isinf(self.best_error)
        self.best_error[unseeded] = error[unseeded]
        improved = opposed_grasp(env, sensor_name, contact_threshold) & (error < self.best_error - min_improvement)
        self.best_error[improved] = error[improved]
        return improved.float()


def success_kernel(
    env: ManagerBasedRLEnv,
    pos_std: float,
    command_name: str,
    sensor_name: str,
    contact_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success kernel at the commanded pose (dexsuite ``success_reward``,
    position-only branch): squared tanh kernel on goal distance, paid only in
    an opposed grasp — the object cannot be knocked into place for credit."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    gate = opposed_grasp(env, sensor_name, contact_threshold)
    return _finite(((1.0 - torch.tanh(distance / pos_std)) ** 2) * gate.float())


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
