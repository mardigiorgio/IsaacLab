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
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NaN-safe action-magnitude penalty (see :func:`_finite`)."""
    return _finite(torch.sum(torch.square(env.action_manager.action), dim=1))


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


def reset_arm_reverse_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose: dict[str, float],
    bank_fraction: float,
    noise: float,
    alpha_min: float = 1.0,
    track_object_xy: list | None = None,
    nominal_object_pos: tuple | None = None,
    safe_yaw_range: tuple | None = None,
    gripper_offset: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reverse-curriculum start states: interpolate home -> pre-grasp.

    ``gripper_offset`` (flip, 2026-08-28): when set, bank starts command the
    gripper to this joint position instead of the seeded one, so a pose seeded
    at jaw-object CONTACT (no penetration, no pop) is squeezed from step 0 --
    a held start whose object rides with the jittering hand instead of being
    batted away (measured: open-jaw starts beside the inverted mug lose it to
    arm exploration within 15 steps at every sigma down to 0.25).

    Selected envs start at ``q = home + alpha * (pose - home)`` with
    ``alpha ~ U(alpha_min, 1)``: at ``alpha_min = 1`` every bank start is the
    pre-grasp itself; as the curriculum lowers ``alpha_min`` toward 0 the
    start distribution grows back along the approach path until home starts
    dominate — the Florensa reverse curriculum on our own two anchors.

    No start is ever seeded mid-grasp: closing is the policy's own act,
    discoverable because the gripper action scale puts the full open-to-seat
    travel inside the initial exploration envelope (see the DISCOVERY
    CONSTRAINT note on the action cfg).
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
    # Under randomized placement the authored pose must FOLLOW the mug: one
    # precomputed Jacobian step translates the pre-grasp by each env's
    # placement delta — sub-mm over the +-1 cm DR range
    # (probe_bank_jacobian.py). The handle is the mug's only asymmetric
    # part, so bank envs also re-sample yaw into the handle-safe arc before
    # the arm is placed around the mug.
    if track_object_xy is not None and sel.any():
        obj = env.scene[object_cfg.name]
        if safe_yaw_range is not None:
            rows = torch.nonzero(sel).squeeze(1)
            ids_sel = env_ids[rows]
            yaw = sample_uniform(safe_yaw_range[0], safe_yaw_range[1], (rows.shape[0],), env.device)
            pose7 = obj.data.root_pose_w.torch[ids_sel].clone()
            pose7[:, 3] = 0.0
            pose7[:, 4] = 0.0
            pose7[:, 5] = torch.sin(yaw / 2)
            pose7[:, 6] = torch.cos(yaw / 2)
            obj.write_root_pose_to_sim_index(root_pose=pose7, env_ids=ids_sel)
            obj.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(rows.shape[0], 6, device=env.device), env_ids=ids_sel
            )
        nominal = torch.tensor(nominal_object_pos, device=env.device)
        delta = obj.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2] - nominal
        M = torch.tensor(track_object_xy, device=env.device, dtype=torch.float32)
        dq = delta @ M.T
        arm_cols = torch.tensor([resolved.index(f"follower_left_joint_{i}") for i in range(6)], device=env.device)
        start[:, arm_cols] = torch.where(sel.unsqueeze(1), start[:, arm_cols] + dq, start[:, arm_cols])
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
    if not hasattr(env, "_bank_offset_map"):
        env._bank_offset_map = []
        for term_name in ("arm_action", "gripper_action"):
            term = env.action_manager.get_term(term_name)
            offset = getattr(term, "_offset", None)
            if not torch.is_tensor(offset):
                continue  # binary-style terms hold no offset (the slide's gripper)
            cols = torch.tensor([resolved.index(n) for n in term._joint_names], device=env.device)
            env._bank_offset_map.append((term, cols))
    for term, cols in env._bank_offset_map:
        term._offset[env_ids] = start[:, cols]
    if gripper_offset is not None:
        gterm = env.action_manager.get_term("gripper_action")
        goff = getattr(gterm, "_offset", None)
        if torch.is_tensor(goff):
            goff[env_ids] = torch.where(sel.unsqueeze(1), torch.full_like(goff[env_ids], float(gripper_offset)), goff[env_ids])
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


def pad_contact_count(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.01) -> torch.Tensor:
    """Number of finger pads in contact with the mug (0, 1 or 2) — upstream's
    contact_count in parallel-jaw form: partial contact earns partial credit,
    which grades the approach into touch before an opposed grasp exists."""
    return _finite((_pad_force_mags(env, sensor_name) > threshold).float().sum(dim=1))


def success_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    sensor_name: str,
    contact_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Upstream success_reward, parallel-jaw form: tanh kernel on the
    object-to-commanded-pose error, paid only while held in an opposed grasp.
    The upstream function itself expects per-fingertip named sensors whose
    layout a parallel jaw does not have; the kernel and gating semantics are
    theirs, the grasp gate is the two-pad opposition."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    err = torch.linalg.vector_norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    held = opposed_grasp(env, sensor_name, contact_threshold)
    return _finite(held * (1.0 - torch.tanh(err / pos_std)))


def pad_contact_forces(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Net contact force on each finger pad from the mug [N], world frame,
    flattened (num_envs, num_pads * 3).

    The touch channel (upstream franka lift feeds fingertip contact forces to
    the policy the same way): without it the policy must close a
    millimeter-tolerance pinch blind, inferring contact from proprioception
    alone. The world frame doubles as the base frame here — the arm base is
    env-aligned and fixed."""
    forces = env.scene.sensors[sensor_name].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    net = forces.torch.sum(dim=2)
    return _finite(net.reshape(net.shape[0], -1))


# Mug body cylinder for the lowest-point kernel: the asset's mesh-asserted
# rim height and body radius (see assets/convert_mug.py).
MUG_BODY_HEIGHT = 0.0973
MUG_BODY_RADIUS = 0.040


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


def object_pose_match(
    env: ManagerBasedRLEnv,
    pos_std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Match the object's POSE to the commanded pose: position kernel times an
    up-axis alignment kernel (yaw-free -- a mug is a surface of revolution).

    Ungated by design: it pays exactly when the object rests in the goal pose,
    held or not, so together with an arm-finish term it rewards placing and
    withdrawing rather than hovering.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3], command[:, 3:7]
    )
    err = torch.linalg.vector_norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    up = torch.tensor([0.0, 0.0, 1.0], device=err.device).expand(env.num_envs, 3)
    cos_tilt = (quat_apply(obj.data.root_quat_w.torch, up) * quat_apply(des_quat_w, up)).sum(dim=-1)
    return _finite((1.0 - torch.tanh(err / pos_std)) * (1.0 + cos_tilt) / 2.0)


def arm_finish_pose_match(
    env: ManagerBasedRLEnv,
    pose: dict[str, float],
    std: float,
    pos_std: float,
    command_name: str,
    sensor_name: str,
    contact_threshold: float = 0.01,
    max_speed: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Arm parked at the authored FINISH pose, GATED ON SUCCESS: credited only
    times the object's pose match, only with both pads OFF the object, and only
    while the object is settled. The withdrawal annuity exists strictly on top
    of a completed, released, calm placement.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
    target = torch.tensor([pose[n] for n in pose], device=env.device)
    err = torch.linalg.vector_norm(robot.data.joint_pos.torch[:, ids] - target, dim=1)
    arm_k = 1.0 - torch.tanh(err / std)
    released = ~opposed_grasp(env, sensor_name, contact_threshold) & (
        _pad_force_mags(env, sensor_name).max(dim=1).values < contact_threshold
    )
    calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
    gate = (released & calm).float()
    return _finite(arm_k * gate * object_pose_match(env, pos_std, command_name, robot_cfg, object_cfg))


def finished_at_pose(
    env: ManagerBasedRLEnv,
    pose: dict[str, float],
    joint_tol: float,
    pos_tol: float,
    min_up_cos: float,
    command_name: str,
    sensor_name: str,
    contact_threshold: float = 0.01,
    max_speed: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The POSITIVE termination: object at the commanded pose, released and calm,
    AND the arm parked at the finish pose. Ends the episode as a completed job;
    pay the completion through ``is_terminated_term`` with a positive weight.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3], command[:, 3:7]
    )
    pos_ok = torch.linalg.vector_norm(des_pos_w - obj.data.root_pos_w.torch, dim=1) < pos_tol
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    tilt_ok = (quat_apply(obj.data.root_quat_w.torch, up) * quat_apply(des_quat_w, up)).sum(dim=-1) > min_up_cos
    released = _pad_force_mags(env, sensor_name).max(dim=1).values < contact_threshold
    calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
    ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
    target = torch.tensor([pose[n] for n in pose], device=env.device)
    arm_ok = torch.abs(robot.data.joint_pos.torch[:, ids] - target).max(dim=1).values < joint_tol
    return pos_ok & tilt_ok & released & calm & arm_ok
# ---------------------------------------------------------------------------- shared success metric

# ONE success semantics for every task in the campaign, the committed
# evaluator's gates (scripts/probes/probe_eval_success.py): the OBJECT within
# 5 cm of the commanded position and upright within acos(0.87), yaw free —
# every object in the family is rotationally symmetric about z for the
# purpose of "delivered". Logged by the command term as Metrics/success_rate.
import math as _math  # noqa: E402

SUCCESS_POS_THRESHOLD = 0.05
SUCCESS_TILT_THRESHOLD = _math.acos(0.87)
# A delivery COUNTS only when held: the success latch requires this many
# consecutive in-gate steps (1 s at 30 Hz — the committed evaluator's hold
# window). A drive-by that clips the gates for a step is not a success.
SUCCESS_HOLD_STEPS = 30


class ObjectPoseSuccessCommand(UniformPoseCommand):  # noqa: F405
    """UniformPoseCommand whose error — and success metric — is the OBJECT's
    root pose against the commanded pose, with TILT-only orientation.

    The stock command gates success on the commanded robot body reaching the
    pose, which counts an empty gripper at the goal; and its quaternion error
    charges yaw, which a delivered round object may hold at any value. The
    campaign's claim is "the object arrived upright", so that is the metric.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object = env.scene["object"]
        self._success_hold = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        out = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._success_hold[ids] = 0.0
        return out

    def _update_metrics(self):
        position_error, orientation_error = self._compute_error()
        self.metrics["position_error"] = position_error
        self.metrics["orientation_error"] = orientation_error
        if self._track_success:
            inside = self._compute_success(position_error, orientation_error)
            self._success_hold = torch.where(
                inside, self._success_hold + 1.0, torch.zeros_like(self._success_hold)
            )
            self._succeeded |= self._success_hold >= SUCCESS_HOLD_STEPS

    def _compute_error(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        position_error = torch.linalg.vector_norm(
            self.pose_command_w[:, :3] - self._object.data.root_pos_w.torch, dim=1
        )
        quat = self._object.data.root_quat_w.torch
        up = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
        tilt = torch.acos(up.clamp(-1.0, 1.0))
        return position_error, tilt
