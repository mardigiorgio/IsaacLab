# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation, termination, and reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    """Cube orientation expressed in the robot root frame, shape [num_envs, 4].

    Quaternion is in (x, y, z, w) convention, matching
    :func:`isaaclab.utils.math.subtract_frame_transforms`.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return object_quat_b


def cube_lifted(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube center exceeds ``minimum_height`` [m] above the env origin."""
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return height > minimum_height


def palms_to_object_vector(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Vectors from each palm to the object center in the robot root frame [m], shape [num_envs, 3 * num_palms]."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :]
    vec_w = obj.data.root_pos_w.torch[:, None, :] - palm_pos_w
    num_envs, num_palms = vec_w.shape[0], vec_w.shape[1]
    root_quat = robot.data.root_quat_w.torch[:, None, :].expand(-1, num_palms, -1)
    vec_b = quat_apply_inverse(root_quat.reshape(-1, 4), vec_w.reshape(-1, 3))
    return vec_b.reshape(num_envs, 3 * num_palms)


def palms_to_cube_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Kernelized closest-palm-to-cube distance, in (0, 1].

    Uses the minimum distance over the given palm bodies so either hand can
    serve the reach; ``std`` [m] sets the kernel width.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_pos = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch[:, None, :] - palm_pos, dim=-1).min(dim=1).values
    # nan_to_num: rewards are computed before resets, so a solver-diverged
    # (non-finite) state must not leak NaN into the rollout
    return torch.exp(-dist / std).nan_to_num(0.0)


def fingers_closed_near_cube(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    palms_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
    left_fingers_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hand_index_.*", "left_hand_middle_.*"]
    ),
    right_fingers_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["right_hand_index_.*", "right_hand_middle_.*"]
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Mean finger flexion of the palm nearest the cube, gated to when that palm is within reach.

    Flexion is the normalized excursion from the open (zero) pose toward the
    joint limit of larger magnitude, averaged over the index and middle finger
    joints of the nearest hand (thumb joints have symmetric limits and are
    excluded). The reward is zero when the nearest palm is farther than
    ``distance_threshold`` [m] from the cube center, so closing the hand only
    pays off near the cube.
    """
    robot = env.scene[palms_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_pos = robot.data.body_pos_w.torch[:, palms_cfg.body_ids, :]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch[:, None, :] - palm_pos, dim=-1)
    nearest_dist, nearest_palm = dist.min(dim=1)

    def _flexion(fingers_cfg: SceneEntityCfg) -> torch.Tensor:
        joint_pos = robot.data.joint_pos.torch[:, fingers_cfg.joint_ids]
        limits = robot.data.joint_pos_limits.torch[:, fingers_cfg.joint_ids, :]
        closed_mag = limits.abs().max(dim=-1).values.clamp(min=1.0e-6)
        return (joint_pos.abs() / closed_mag).clamp(max=1.0).mean(dim=1)

    # palms_cfg body order is [left, right] (body-index order on the G1)
    flexion = torch.where(nearest_palm == 0, _flexion(left_fingers_cfg), _flexion(right_fingers_cfg))
    return (flexion * (nearest_dist < distance_threshold)).nan_to_num(0.0)


def _target_pos_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_offset: tuple[float, float, float]):
    """World position of a robot-root-relative target offset [m], shape [num_envs, 3]."""
    robot = env.scene[robot_cfg.name]
    offset = torch.tensor(target_offset, device=env.device).expand(env.num_envs, 3)
    return robot.data.root_pos_w.torch + quat_apply(robot.data.root_quat_w.torch, offset)


def object_to_target_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Kernelized cube distance to a robot-relative target point, in (0, 1].

    ``target_offset`` [m] is expressed in the robot root frame, so the target
    follows the robot's (jittered) placement; ``std`` [m] sets the kernel
    width.
    """
    obj = env.scene[object_cfg.name]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - _target_pos_w(env, robot_cfg, target_offset), dim=1)
    return torch.exp(-dist / std).nan_to_num(0.0)


def object_near_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube center is within ``threshold`` [m] of the robot-relative target point."""
    obj = env.scene[object_cfg.name]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - _target_pos_w(env, robot_cfg, target_offset), dim=1)
    return dist < threshold


def robot_or_object_state_invalid(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the robot joint state runs away or any watched state goes non-finite.

    Solver-divergence guard: blowups usually pass through finite joint
    velocities above ``max_velocity`` [rad/s] on the way to NaN, but a single
    step can also jump straight to non-finite — and NaN never trips a
    magnitude comparison. Both cases must terminate so the env resets instead
    of poisoning the rollout.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, robot_cfg.joint_ids]
    bad_robot = (~torch.isfinite(joint_pos) | ~torch.isfinite(joint_vel) | (joint_vel.abs() > max_velocity)).any(dim=1)
    bad_object = ~torch.isfinite(obj.data.root_pos_w.torch).all(dim=1)
    return bad_robot | bad_object


def object_lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Cube height progress from ``rest_height`` toward ``target_height`` [m], clamped to [0, 1]."""
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return ((height - rest_height) / (target_height - rest_height)).clamp(0.0, 1.0).nan_to_num(0.0)
