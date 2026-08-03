# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation, event, termination, and reward functions.

Minimal pick-up set: reach the handle, close the
fingers near it, lift. The no-blade rule lives in the terminations, not the
rewards. Every reward is ``nan_to_num``: rewards are computed before resets,
so a solver-diverged state must not leak NaN into the rollout.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Helpers
##


def _handle_grasp_point_w(
    env: ManagerBasedRLEnv, grasp_offset_b: tuple[float, float, float], object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """World position of the handle grasp point [m] (body-frame offset on the handle centerline)."""
    obj = env.scene[object_cfg.name]
    off = torch.tensor(grasp_offset_b, device=env.device).expand(env.num_envs, 3)
    return obj.data.root_pos_w.torch + quat_apply(obj.data.root_quat_w.torch, off)


def _handle_segment_geometry(
    env: ManagerBasedRLEnv,
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    bodies_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance [m] and local-y offset [m] of robot links vs the handle centerline segment.

    Both returns are [num_envs, num_bodies], columns in the RESOLVED body order
    of ``bodies_cfg`` (asset order, not the cfg's name-list order). The math
    runs in the spatula body frame, where the segment is constant and
    point-to-segment distance equals the world-frame distance.
    """
    robot = env.scene[bodies_cfg.name]
    obj = env.scene[object_cfg.name]
    pts_w = robot.data.body_pos_w.torch[:, bodies_cfg.body_ids, :]
    quat = obj.data.root_quat_w.torch.unsqueeze(1).expand(-1, pts_w.shape[1], -1)
    pts_b = quat_apply_inverse(quat, pts_w - obj.data.root_pos_w.torch.unsqueeze(1))
    p0 = torch.tensor(handle_p0_b, device=env.device)
    seg = torch.tensor(handle_p1_b, device=env.device) - p0
    t = ((pts_b - p0) * seg).sum(dim=-1).div(seg.dot(seg)).clamp(0.0, 1.0)
    dist = torch.linalg.vector_norm(pts_b - (p0 + t.unsqueeze(-1) * seg), dim=-1)
    return dist, pts_b[..., 1]


def _grasp(
    env: ManagerBasedRLEnv,
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    palm_radius: float,
    thumb_radius: float,
    contact_radius: float,
    palm_cfg: SceneEntityCfg,
    fingertips_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Privileged power-grasp bool, shape [num_envs]: handle in the palm, fingers wrapped.

    True when at least one of index/middle tips is within ``contact_radius``
    of the handle segment on the OPPOSITE local-y side from the thumb, the
    thumb tip is within ``thumb_radius`` (pad on the near face), and the palm
    link is within ``palm_radius``. Radii are calibrated to the G1 TriHand's
    LINK ORIGINS in the pronated vertical-curl grip (probe-measured): the
    distal link frames sit at the knuckles, so the thumb origin stays ~5-6 cm
    from the centerline even with the pad pressed on the handle — hence the
    wider thumb radius. NaN-safe: non-finite positions compare False.
    """
    palm_d, _ = _handle_segment_geometry(env, handle_p0_b, handle_p1_b, palm_cfg, object_cfg)
    tip_d, y = _handle_segment_geometry(env, handle_p0_b, handle_p1_b, fingertips_cfg, object_cfg)
    # column order follows body_ids (ASSET order) — cfg.body_names keeps the
    # user-given order in this fork, so map columns through the articulation
    robot = env.scene[fingertips_cfg.name]
    names = [robot.body_names[i] for i in fingertips_cfg.body_ids]
    ti = names.index("right_hand_thumb_2_link")
    ii = names.index("right_hand_index_1_link")
    mi = names.index("right_hand_middle_1_link")
    opposed = ((tip_d[:, ii] < contact_radius) & (y[:, ti] * y[:, ii] < 0)) | (
        (tip_d[:, mi] < contact_radius) & (y[:, ti] * y[:, mi] < 0)
    )
    return opposed & (tip_d[:, ti] < thumb_radius) & (palm_d.squeeze(1) < palm_radius)


def _sensor_peak_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Peak per-body NET filtered contact force magnitude [N], shape [num_envs].

    Reads the Newton filtered force matrix ([envs, bodies, filter prims, 3]),
    sums over the filter-prim axis BEFORE the norm, then takes the max over
    sensor bodies so multi-body sensors report their hardest-pressed link.
    """
    forces = env.scene.sensors[sensor_name].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    net = forces.torch.sum(dim=2)
    mag = torch.linalg.vector_norm(net, dim=-1)
    return mag.reshape(env.num_envs, -1).max(dim=1).values.nan_to_num(0.0)


##
# Observations
##


def digit_handle_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """PRIVILEGED per-digit handle contact force [N], shape [num_envs, 3].

    Thumb/index/middle peak filtered force on the handle prim — the exact
    quantities the grasp reward, lift gate, and success gate threshold on.
    Sim-only (the real TriHand has no fingertip force sensing): teacher obs,
    stripped at distillation.
    """
    return torch.stack(
        [
            _sensor_peak_force(env, "thumb_handle_contact"),
            _sensor_peak_force(env, "index_handle_contact"),
            _sensor_peak_force(env, "middle_handle_contact"),
        ],
        dim=1,
    )


def blade_contact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """PRIVILEGED hand-on-blade contact force [N], shape [num_envs, 1] — feel
    the forbidden contact before the termination threshold trips."""
    return _sensor_peak_force(env, "hand_blade_contact").unsqueeze(-1)


def fingertip_cage_geometry(
    env: ManagerBasedRLEnv,
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    fingertips_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_.*_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """PRIVILEGED straddle geometry, shape [num_envs, 6]: each fingertip's
    distance to the handle centerline segment [m] and its handle-local-y side
    [m] (sign = which face of the handle the tip is on — the cage state the
    privileged grasp gates use). Columns follow the resolved asset body order.
    """
    dist, y = _handle_segment_geometry(env, handle_p0_b, handle_p1_b, fingertips_cfg, object_cfg)
    return torch.cat([dist, y], dim=1).nan_to_num(0.0)


def object_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Spatula linear + angular velocity in the robot root frame, shape
    [num_envs, 6] — slip and tumble dynamics for reactive grip corrections."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    quat = robot.data.root_quat_w.torch
    lin = quat_apply_inverse(quat, obj.data.root_lin_vel_w.torch)
    ang = quat_apply_inverse(quat, obj.data.root_ang_vel_w.torch)
    return torch.cat([lin, ang], dim=1).nan_to_num(0.0)


def handle_frame_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    grasp_offset_b: tuple[float, float, float],
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Handle grasp point + handle axis unit vector in the robot root frame,
    shape [num_envs, 6] — the pinch target line, without requiring the policy
    to re-derive it from the spatula root pose."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    gp_w = _handle_grasp_point_w(env, grasp_offset_b, object_cfg)
    axis_b = torch.tensor(handle_p1_b, device=env.device) - torch.tensor(handle_p0_b, device=env.device)
    axis_w = quat_apply(
        obj.data.root_quat_w.torch, torch.nn.functional.normalize(axis_b, dim=0).expand(env.num_envs, 3)
    )
    gp_r, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, gp_w)
    axis_r = quat_apply_inverse(robot.data.root_quat_w.torch, axis_w)
    return torch.cat([gp_r, axis_r], dim=1).nan_to_num(0.0)


def palm_aim_obs(
    env: ManagerBasedRLEnv,
    grasp_offset_b: tuple[float, float, float],
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """The signed palm-aim cosine as an observation, shape [num_envs, 1]."""
    return palm_aim(env, grasp_offset_b, palm_cfg, object_cfg).unsqueeze(-1)


def lift_progress_obs(env: ManagerBasedRLEnv, rest_height: float, target_height: float) -> torch.Tensor:
    """Task phase: spatula height progress from rest toward the success
    height, clamped to [0, 1], shape [num_envs, 1]."""
    z = env.scene["spatula"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    frac = (z - rest_height) / (target_height - rest_height)
    return frac.clamp(0.0, 1.0).unsqueeze(-1).nan_to_num(0.0)


def time_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the episode remaining, shape [num_envs, 1] — finite-horizon
    value estimation with a terminal bonus needs to see the clock."""
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Spatula position [m] expressed in the robot root frame, shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Spatula orientation expressed in the robot root frame, shape [num_envs, 4]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return object_quat_b


def palm_to_handle_vector(
    env: ManagerBasedRLEnv,
    grasp_offset_b: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Vector from the right palm to the handle grasp point in the robot root frame [m], shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    palm_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :].squeeze(1)
    vec_w = _handle_grasp_point_w(env, grasp_offset_b, object_cfg) - palm_pos_w
    vec_b, _ = subtract_frame_transforms(torch.zeros_like(vec_w), robot.data.root_quat_w.torch, vec_w)
    return vec_b


##
# Rewards
##


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using an L2-squared kernel, clamped for solver-divergence safety."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000).nan_to_num(0.0)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using an L2-squared kernel, clamped."""
    return (
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
        .clamp(-1000, 1000)
        .nan_to_num(0.0)
    )


def palm_to_handle_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_offset_b: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Kernelized palm-to-grasp-point distance, in (0, 1]. ``std`` [m] sets the kernel width."""
    robot = env.scene[robot_cfg.name]
    palm = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :].squeeze(1)
    gp = _handle_grasp_point_w(env, grasp_offset_b, object_cfg)
    dist = torch.linalg.vector_norm(palm - gp, dim=-1)
    return torch.exp(-dist / std).nan_to_num(0.0)


def fingers_closed_near_handle(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    grasp_offset_b: tuple[float, float, float],
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    fingers_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["right_hand_.*_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Mean finger flexion, gated to the palm within reach of the grasp point
    AND graded by the palm pointing at it.

    Flexion is measured against each joint's larger-|limit| bound, so it is
    sign-correct for the TriHand thumb. The aim grade is the clamped-positive
    ``palm_aim`` dot product, multiplicative: closure only pays while the palm
    faces the handle, so knuckle-first or back-of-hand closure earns nothing.
    """
    robot = env.scene[palm_cfg.name]
    palm = robot.data.body_pos_w.torch[:, palm_cfg.body_ids, :].squeeze(1)
    gp = _handle_grasp_point_w(env, grasp_offset_b, object_cfg)
    near = torch.linalg.vector_norm(palm - gp, dim=-1) < distance_threshold
    palm_quat = robot.data.body_quat_w.torch[:, palm_cfg.body_ids, :].squeeze(1)
    palm_normal = quat_apply(palm_quat, torch.tensor((0.0, 0.0, -1.0), device=env.device).expand(env.num_envs, 3))
    aim = (palm_normal * torch.nn.functional.normalize(gp - palm, dim=-1)).sum(dim=-1).clamp(0.0, 1.0)
    joint_pos = robot.data.joint_pos.torch[:, fingers_cfg.joint_ids]
    limits = robot.data.joint_pos_limits.torch[:, fingers_cfg.joint_ids, :]
    closed_mag = limits.abs().max(dim=-1).values.clamp(min=1.0e-6)
    flexion = (joint_pos.abs() / closed_mag).clamp(max=1.0).mean(dim=1)
    return (flexion * near.float() * aim).nan_to_num(0.0)


def _opposed_handle_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Grasp gate bool [num_envs], the dexsuite recipe: thumb pressing the
    handle above ``threshold`` [N] AND at least one opposing digit doing the
    same. Multiplicatively zeroes all task income when False."""
    thumb = _sensor_peak_force(env, "thumb_handle_contact") >= threshold
    index = _sensor_peak_force(env, "index_handle_contact") >= threshold
    middle = _sensor_peak_force(env, "middle_handle_contact") >= threshold
    return thumb & (index | middle)


def fingers_to_handle(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_offset_b: tuple[float, float, float],
    contact_threshold: float | None = None,
    bodies_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Reach/enclosure flow (dexsuite ``object_ee_distance`` shape): kernel on
    the WORST distance over palm + fingertips to the handle grasp point, so
    the whole hand must wrap the handle, times a contact bonus that floors at
    0.1 without the opposed-contact gate and 1.0 with it.

    ``std`` is wide (0.4 m in the proven recipe) ON PURPOSE: the kernel keeps
    gradient across the whole workspace — no absorbing far region.
    ``contact_threshold`` of ``None`` drops the contact bonus entirely and
    returns the bare reach kernel — the claw-grasp recipe, where the opposed
    thumb/finger force signature a pinch produces may never appear.
    """
    robot = env.scene[bodies_cfg.name]
    bodies = robot.data.body_pos_w.torch[:, bodies_cfg.body_ids, :]
    gp = _handle_grasp_point_w(env, grasp_offset_b, object_cfg).unsqueeze(1)
    worst = torch.linalg.vector_norm(bodies - gp, dim=-1).max(dim=1).values
    reach = 1.0 - torch.tanh(worst / std)
    if contact_threshold is None:
        return reach.nan_to_num(0.0)
    bonus = _opposed_handle_contact(env, contact_threshold).float().clamp(0.1, 1.0)
    return (reach * bonus).nan_to_num(0.0)


def object_lifted(
    env: ManagerBasedRLEnv,
    rest_height: float,
    minimal_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Binary lift indicator, shape [num_envs]: 1.0 while the object root is above rest.

    The Franka-lift height gate with NO contact gate (the published lift
    recipe): income for the object being off the table, whatever is holding it
    there. The no-catapult job belongs to the ``spatula_dropped`` termination
    and to the fact that a flung object stops accruing carry-point income
    within a few frames.

    Args:
        env: The RL environment instance.
        rest_height: Object root height at rest in the env frame [m].
        minimal_offset: Rise above ``rest_height`` that counts as lifted [m].
        object_cfg: Entity config for the lifted object.
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z > (rest_height + minimal_offset)).float().nan_to_num(0.0)


def track_carry_point(
    env: ManagerBasedRLEnv,
    carry_point: tuple[float, float, float],
    stds: tuple[float, ...],
    weights: tuple[float, ...],
    rest_height: float,
    minimal_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Multi-scale carry-point tracking income, shape [num_envs].

    ``1(z > rest + offset) * sum_k w_k [1 - tanh(d_goal / sigma_k)]`` where
    ``d_goal`` [m] is the object root to ``carry_point`` in the env frame. Two
    kernels — one coarse, one fine — give gradient from across the workspace
    while still rewarding the last centimetre. Gated on height only, never on
    contact.

    Args:
        env: The RL environment instance.
        carry_point: Target position in the env frame [m].
        stds: Kernel widths [m], one per scale.
        weights: Per-kernel weights, same length as ``stds``.
        rest_height: Object root height at rest in the env frame [m].
        minimal_offset: Rise above ``rest_height`` that enables the income [m].
        object_cfg: Entity config for the tracked object.

    Raises:
        ValueError: If ``stds`` and ``weights`` differ in length.
    """
    if len(stds) != len(weights):
        raise ValueError(f"stds and weights must be the same length, got {len(stds)} and {len(weights)}")
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w.torch - env.scene.env_origins
    target = torch.tensor(carry_point, device=env.device).expand(env.num_envs, 3)
    dist = torch.linalg.vector_norm(pos - target, dim=-1)
    lifted = pos[:, 2] > (rest_height + minimal_offset)
    income = torch.zeros_like(dist)
    for std, weight in zip(stds, weights):
        income = income + weight * (1.0 - torch.tanh(dist / std))
    return (income * lifted.float()).nan_to_num(0.0)


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities with an L2-squared kernel, clamped for solver-divergence safety.

    Mirrors :func:`action_l2_clamped`. ``isaaclab.envs.mdp.joint_vel_l2`` is
    unclamped; one diverged env would poison the whole batch, which this
    module's NaN/clamp invariant exists to prevent.

    Args:
        env: The RL environment instance.
        asset_cfg: Entity config selecting the penalized joints.
    """
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1).clamp(-1000, 1000).nan_to_num(0.0)


def handle_contact_count(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Fraction of digits pressing the handle above ``threshold`` [N] — the
    ungated bootstrap gradient toward touching (dexsuite ``contact_count``)."""
    digits = torch.stack(
        [
            _sensor_peak_force(env, "thumb_handle_contact") >= threshold,
            _sensor_peak_force(env, "index_handle_contact") >= threshold,
            _sensor_peak_force(env, "middle_handle_contact") >= threshold,
        ],
        dim=1,
    )
    return digits.float().mean(dim=1).nan_to_num(0.0)


def lifted_with_handle_contact(
    env: ManagerBasedRLEnv, rest_height: float, minimal_offset: float, contact_threshold: float
) -> torch.Tensor:
    """Binary lifted flow (the Franka-lift height gate, contact-gated for the
    no-catapult rule): 1.0 every step the spatula root is ``minimal_offset``
    above rest WHILE the opposed-contact gate holds."""
    z = env.scene["spatula"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    lifted = z > (rest_height + minimal_offset)
    return (lifted & _opposed_handle_contact(env, contact_threshold)).float()


def hold_at_carry_point(
    env: ManagerBasedRLEnv, carry_point: tuple[float, float, float], std: float, contact_threshold: float
) -> torch.Tensor:
    """THE task income: squared tanh kernel on spatula-root distance to the
    fixed carry point, hard-gated on opposed handle contact. Continuous flow
    — being grasped at the carry point is the best-paying state in the MDP,
    every step, with no success termination to game."""
    pos = env.scene["spatula"].data.root_pos_w.torch - env.scene.env_origins
    target = torch.tensor(carry_point, device=env.device).expand(env.num_envs, 3)
    dist = torch.linalg.vector_norm(pos - target, dim=-1)
    gate = _opposed_handle_contact(env, contact_threshold).float()
    return (gate * (1.0 - torch.tanh(dist / std)) ** 2).nan_to_num(0.0)


def _force_grasp(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Force-closure grasp bool [num_envs], the ManiSkill G1 pick recipe:
    adapted to a full three-finger CLAW: thumb AND index AND middle all
    pressing the handle above ``threshold`` [N]. Sensors cover WHOLE digits
    (both phalanx links) — the TriHand's pinch lands on the thumb's
    middle-phalanx pad, which a distal-only sensor misses."""
    thumb = _sensor_peak_force(env, "thumb_handle_contact") >= threshold
    index = _sensor_peak_force(env, "index_handle_contact") >= threshold
    middle = _sensor_peak_force(env, "middle_handle_contact") >= threshold
    return thumb & index & middle


def pinch_center_reaching(
    env: ManagerBasedRLEnv,
    grasp_offset_b: tuple[float, float, float],
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_thumb_2_link"]),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_index_1_link"]),
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Reaching kernel ``1 - tanh(5 d)`` on the PINCH CENTER (thumb/index tip
    midpoint) to the handle grasp point — the ManiSkill TCP anchor."""
    robot = env.scene[thumb_cfg.name]
    thumb = robot.data.body_pos_w.torch[:, thumb_cfg.body_ids, :].squeeze(1)
    finger = robot.data.body_pos_w.torch[:, finger_cfg.body_ids, :].squeeze(1)
    pinch_center = 0.5 * (thumb + finger)
    gp = _handle_grasp_point_w(env, grasp_offset_b, object_cfg)
    dist = torch.linalg.vector_norm(pinch_center - gp, dim=-1)
    return (1.0 - torch.tanh(5.0 * dist)).nan_to_num(0.0)


def palm_aim(
    env: ManagerBasedRLEnv,
    grasp_offset_b: tuple[float, float, float],
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """SIGNED palm aim in [-1, 1]: the arrow out of the palm (local -z,
    probe-verified) dotted with the direction to the handle grasp point.
    Pointing at the handle pays +1; pointing away is PENALIZED (negative) —
    back-of-hand approaches lose money every step."""
    robot = env.scene[palm_cfg.name]
    palm_pos = robot.data.body_pos_w.torch[:, palm_cfg.body_ids, :].squeeze(1)
    palm_quat = robot.data.body_quat_w.torch[:, palm_cfg.body_ids, :].squeeze(1)
    palm_normal = quat_apply(palm_quat, torch.tensor((0.0, 0.0, -1.0), device=env.device).expand(env.num_envs, 3))
    gp = _handle_grasp_point_w(env, grasp_offset_b, object_cfg)
    to_grasp = torch.nn.functional.normalize(gp - palm_pos, dim=-1)
    return (palm_normal * to_grasp).sum(dim=-1).clamp(-1.0, 1.0).nan_to_num(0.0)


def grasp_force_reward(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Per-digit claw credit in {0, 1/3, 2/3, 1}: each digit pressing the
    handle above ``threshold`` [N] pays a third — a dense bridge toward the
    full three-finger claw that the lift and success gates require."""
    digits = torch.stack(
        [
            _sensor_peak_force(env, "thumb_handle_contact") >= threshold,
            _sensor_peak_force(env, "index_handle_contact") >= threshold,
            _sensor_peak_force(env, "middle_handle_contact") >= threshold,
        ],
        dim=1,
    )
    return digits.float().mean(dim=1).nan_to_num(0.0)


def grasp_handle(
    env: ManagerBasedRLEnv,
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    palm_radius: float,
    thumb_radius: float,
    contact_radius: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    fingertips_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """1.0 while the privileged power grasp holds (see :func:`_grasp`)."""
    return _grasp(
        env, handle_p0_b, handle_p1_b, palm_radius, thumb_radius, contact_radius, palm_cfg, fingertips_cfg, object_cfg
    ).float()


def blade_pointed_inward(
    env: ManagerBasedRLEnv,
    inward_dir_w: tuple[float, float, float],
    handle_p0_b: tuple[float, float, float],
    handle_p1_b: tuple[float, float, float],
    palm_radius: float,
    thumb_radius: float,
    contact_radius: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    fingertips_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Use-ready aim while grasped, in [0, 1]: blade axis (body -X) dotted with ``inward_dir_w``.

    Gated on the grasp so the resting spatula (which already points inward)
    pays nothing.
    """
    obj = env.scene[object_cfg.name]
    blade_b = torch.tensor((-1.0, 0.0, 0.0), device=env.device).expand(env.num_envs, 3)
    blade_w = quat_apply(obj.data.root_quat_w.torch, blade_b)
    inward = torch.tensor(inward_dir_w, device=env.device).expand(env.num_envs, 3)
    aim = (blade_w * inward).sum(dim=-1).clamp(0.0, 1.0)
    grasp = _grasp(
        env, handle_p0_b, handle_p1_b, palm_radius, thumb_radius, contact_radius, palm_cfg, fingertips_cfg, object_cfg
    )
    return (aim * grasp.float()).nan_to_num(0.0)


def spatula_lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    force_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """Height progress of the spatula root from ``rest_height`` toward ``target_height`` [m], in [0, 1].

    Gated on the FORCE-CLOSURE grasp (ManiSkill G1 pick recipe): lift income
    requires the fingers actually pressing the handle, so nudging the spatula
    upward without holding it pays nothing.
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    progress = ((z - rest_height) / (target_height - rest_height)).clamp(0.0, 1.0)
    return (progress * _force_grasp(env, force_threshold).float()).nan_to_num(0.0)


##
# Terminations
##


def object_lifted_above(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """True when the spatula root rises above ``minimum_height`` [m] (env frame) — the pick succeeded."""
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z > minimum_height).nan_to_num(False)


def object_lifted_above_while_grasped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    force_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """True when the spatula rises above ``minimum_height`` [m] WHILE force-grasped — the pick succeeded.

    The grasp gate kills the catapult exploit: a flung spatula leaves the
    hand, so the finger contact forces are zero at the height crossing and
    the launch never terminates as success.
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return ((z > minimum_height) & _force_grasp(env, force_threshold)).nan_to_num(False)


def blade_contact(env: ManagerBasedRLEnv, threshold: float, sensor_name: str = "hand_blade_contact") -> torch.Tensor:
    """True when any right-hand link presses the BLADE above ``threshold`` [N].

    The task is pick-it-up-BY-THE-HANDLE — grabbing or pushing the blade ends
    the episode. The threshold forgives feather grazes.
    """
    return _sensor_peak_force(env, sensor_name) > threshold


def object_out_of_bound(
    env: ManagerBasedRLEnv,
    in_bound_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """True when the object leaves the env-frame workspace box (dexsuite ``out_of_bound`` convention)."""
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w.torch - env.scene.env_origins
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for axis, (lo, hi) in in_bound_range.items():
        a = "xyz".index(axis)
        out = out | (pos[:, a] < lo) | (pos[:, a] > hi)
    return out.nan_to_num(True)


def robot_or_object_state_invalid(
    env: ManagerBasedRLEnv,
    vel_limit_factor: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> torch.Tensor:
    """True when a watched joint exceeds ``vel_limit_factor``x its velocity limit or any state goes non-finite.

    The per-joint limit scaling is the dexsuite ``abnormal_robot_state``
    convention (factor 2). Keep the fingers out of ``robot_cfg``: their tiny
    links spike legitimately on contact.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, robot_cfg.joint_ids]
    vel_limits = robot.data.joint_vel_limits.torch[:, robot_cfg.joint_ids]
    bad_robot = (
        ~torch.isfinite(joint_pos) | ~torch.isfinite(joint_vel) | (joint_vel.abs() > vel_limits * vel_limit_factor)
    ).any(dim=1)
    bad_object = ~torch.isfinite(obj.data.root_pos_w.torch).all(dim=1)
    return bad_robot | bad_object


##
# Events
##


def hold_joints_at_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write PD position targets = default joint pos for the given (non-actioned) joints.

    In this fork, joints without an action term keep position target 0.0 —
    they silently drive to the ZERO pose, not ``init_state.joint_pos``. The
    action manager only writes targets for actuated joints, so one write per
    reset persists for everything else.
    """
    robot = env.scene[robot_cfg.name]
    targets = robot.data.default_joint_pos.torch[env_ids][:, robot_cfg.joint_ids]
    robot.set_joint_position_target_index(target=targets, joint_ids=robot_cfg.joint_ids, env_ids=env_ids)


##
# Events — reset map
##

_GRASP_MAP_CACHE: dict[str, dict | None] = {}
_GRASP_MAP_WARNED: set[str] = set()


def reset_from_grasp_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    map_path: str,
    stage_probs: tuple[float, ...],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("spatula"),
) -> None:
    """Teleport a sampled subset of resetting envs into curated states along the task.

    The grasp map is a torch file holding ``{"stages": [...]}`` where each
    stage is ``{"name", "joint_pos" (J,), "joint_vel" (J,), "object_pos" (3,)
    env-frame, "object_quat" (4, in (x, y, z, w) — this fork's convention)}``.
    ``stage_probs[0]`` is the nominal stage (leave the preceding default reset
    untouched); ``stage_probs[i]`` selects ``stages[i-1]``. Runs AFTER the
    default joint/object reset events, overwriting state only for envs that
    drew a non-nominal stage.

    Missing map file → the term is a no-op (nominal resets only), so training
    runs before the map is authored.
    """
    key = os.path.abspath(map_path)
    if key not in _GRASP_MAP_CACHE:
        if os.path.exists(key):
            try:
                _GRASP_MAP_CACHE[key] = torch.load(key, weights_only=True)
            except Exception as err:
                # e.g. an unsmudged git-lfs pointer file — same contract as a
                # missing map: warn once, nominal resets only
                _GRASP_MAP_CACHE[key] = None
                if key not in _GRASP_MAP_WARNED:
                    _GRASP_MAP_WARNED.add(key)
                    print(f"[g1_spatula_lift] grasp map at {key} failed to load ({err}); resets are nominal-only")
        else:
            _GRASP_MAP_CACHE[key] = None
    grasp_map = _GRASP_MAP_CACHE[key]
    if grasp_map is None:
        if key not in _GRASP_MAP_WARNED:
            _GRASP_MAP_WARNED.add(key)
            print(f"[g1_spatula_lift] grasp map not found at {key}; resets are nominal-only")
        return

    num_stages = min(len(stage_probs) - 1, len(grasp_map["stages"]))
    probs = torch.tensor(stage_probs[: num_stages + 1], device=env.device, dtype=torch.float)
    draws = torch.multinomial(probs.expand(len(env_ids), -1), 1).squeeze(-1)
    robot = env.scene[robot_cfg.name]
    for stage_idx in range(1, num_stages + 1):
        sel = env_ids[draws == stage_idx]
        if len(sel) == 0:
            continue
        stage = grasp_map["stages"][stage_idx - 1]
        joint_pos = stage["joint_pos"].to(env.device).unsqueeze(0).expand(len(sel), -1)
        joint_vel = stage["joint_vel"].to(env.device).unsqueeze(0).expand(len(sel), -1)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=sel)
        # DELIBERATELY no object write: the spatula pose was already set by the
        # preceding reset_all event, and a SECOND free-body root write in the
        # same reset (after the joint teleport) can eject the object. Stages
        # reposition the ROBOT only.


def reset_to_joint_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_pose: dict[str, float],
    probability: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Teleport a sampled share of resetting envs into a named joint pose.

    The reset MIXTURE: with ``probability`` 0.5 half the environments start in
    the authored pregrasp (digits already around the handle, so the policy only
    has to learn the close and the lift) and half start at the nominal ready
    pose (so the reach is still learned and the pregrasp is not the only state
    the policy ever sees).

    Values are clamped into each joint's limits, so an authored pose slightly
    outside a bound is admitted rather than rejected at env construction.

    Unlike :func:`reset_from_grasp_map` this takes the pose inline instead of
    loading a ``.pt``: ``*.pt`` is routed through git-lfs in this repo and
    arrives unsmudged on some machines, which silently disables the stage.

    Args:
        env: The RL environment instance.
        env_ids: Environments being reset.
        joint_pose: Joint name to position [rad]. Names not on the robot are ignored.
        probability: Share of resetting envs teleported into the pose, in [0, 1].
        robot_cfg: Entity config for the robot.
    """
    if len(env_ids) == 0 or probability <= 0.0:
        return
    robot = env.scene[robot_cfg.name]
    picked = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    if len(picked) == 0:
        return
    joint_pos = robot.data.default_joint_pos.torch[picked].clone()
    limits = robot.data.joint_pos_limits.torch[0]
    for name, value in joint_pose.items():
        if name not in robot.joint_names:
            continue
        idx = robot.joint_names.index(name)
        lo, hi = limits[idx, 0].item(), limits[idx, 1].item()
        joint_pos[:, idx] = min(max(value, lo), hi)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=picked)
