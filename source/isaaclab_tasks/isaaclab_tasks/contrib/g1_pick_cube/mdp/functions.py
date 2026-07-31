# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation, termination, event, and reward functions.

Minimal pick-and-hold set: reach the cube, close the fingers near it, lift,
carry to a robot-relative hold target. Every reward is ``nan_to_num``:
rewards are computed before resets, so a solver-diverged state must not leak
NaN into the rollout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Helpers
##


def _target_pos_w(
    env: ManagerBasedRLEnv, target_offset: tuple[float, float, float], robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """World position of a robot-root-relative target offset [m], shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    offset = torch.tensor(target_offset, device=env.device).expand(env.num_envs, 3)
    return robot.data.root_pos_w.torch + quat_apply(robot.data.root_quat_w.torch, offset)


##
# Observations
##


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube position [m] expressed in the robot root frame, shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube orientation expressed in the robot root frame, shape [num_envs, 4]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return object_quat_b


def palm_to_object_vector(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Vector from the right palm to the cube center in the robot root frame [m], shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :].squeeze(1)
    vec_w = obj.data.root_pos_w.torch - palm_pos_w
    vec_b, _ = subtract_frame_transforms(torch.zeros_like(vec_w), robot.data.root_quat_w.torch, vec_w)
    return vec_b


def object_to_target_vector(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Vector from the cube center to the robot-relative hold target, env frame [m], shape [num_envs, 3]."""
    obj = env.scene[object_cfg.name]
    return _target_pos_w(env, target_offset, robot_cfg) - obj.data.root_pos_w.torch


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


def palm_to_object_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Kernelized palm-to-cube distance, in (0, 1]. ``std`` [m] sets the kernel width."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    palm = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :].squeeze(1)
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - palm, dim=-1)
    return torch.exp(-dist / std).nan_to_num(0.0)


def fingers_closed_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
    fingers_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["right_hand_.*_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Mean finger flexion gated to when the palm is within reach of the cube.

    The dense bridge between "reached" and "grasped". Flexion is measured
    against each joint's larger-|limit| bound, so it is sign-correct for the
    TriHand thumb.
    """
    robot = env.scene[palm_cfg.name]
    obj = env.scene[object_cfg.name]
    palm = robot.data.body_pos_w.torch[:, palm_cfg.body_ids, :].squeeze(1)
    near = torch.linalg.vector_norm(obj.data.root_pos_w.torch - palm, dim=-1) < distance_threshold
    joint_pos = robot.data.joint_pos.torch[:, fingers_cfg.joint_ids]
    limits = robot.data.joint_pos_limits.torch[:, fingers_cfg.joint_ids, :]
    closed_mag = limits.abs().max(dim=-1).values.clamp(min=1.0e-6)
    flexion = (joint_pos.abs() / closed_mag).clamp(max=1.0).mean(dim=1)
    return (flexion * near.float()).nan_to_num(0.0)


def object_lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Height progress of the cube from ``rest_height`` toward ``target_height`` [m], in [0, 1]."""
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return ((z - rest_height) / (target_height - rest_height)).clamp(0.0, 1.0).nan_to_num(0.0)


def object_to_target_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Kernelized cube distance to the robot-relative hold target, in (0, 1]."""
    obj = env.scene[object_cfg.name]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - _target_pos_w(env, target_offset, robot_cfg), dim=1)
    return torch.exp(-dist / std).nan_to_num(0.0)


##
# Terminations
##


def object_near_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube center is within ``threshold`` [m] of the hold target — the pick succeeded."""
    obj = env.scene[object_cfg.name]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - _target_pos_w(env, target_offset, robot_cfg), dim=1)
    return (dist < threshold).nan_to_num(False)


def object_out_of_bound(
    env: ManagerBasedRLEnv,
    in_bound_range: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
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
