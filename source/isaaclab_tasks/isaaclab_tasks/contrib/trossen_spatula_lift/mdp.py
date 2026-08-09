# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classic Franka-cube-lift MDP terms for the Trossen spatula task.

The four functions are the reference lift task's reach / lift / goal-track terms and
the object-position observation, implemented against the stable generic layer only
(``isaaclab.envs.mdp`` plus the ``.torch`` views of the warp-backed data buffers), so
this task does not depend on the evolving ``core.lift`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def nonfinite_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose simulation state went non-finite.

    Float32 MJWarp under the fixed solver sporadically collapses a single world to NaN
    in one step during grasp contact (measured ~1 per 5.8M env-steps at 8192 envs; the
    healthy population shows tame dynamics when it happens). Terminating and resetting
    the world contains the event; rewards are NaN-sanitized so the dying step cannot
    poison the rollout buffer. The term is symmetric across both experiment arms, so
    its trigger rate is itself an experiment metric (expected ~0 under the adaptive
    solver).
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    jp = robot.data.joint_pos.torch
    jv = robot.data.joint_vel.torch
    op = obj.data.root_pos_w.torch
    ov = obj.data.root_lin_vel_w.torch
    # A collapsing world passes through ASTRONOMICAL-BUT-FINITE state for a step
    # before NaN (measured: value loss 1e26 from a joint_vel ~1e15 leaking into the
    # critic through a finite reward while isfinite-only guarding let it through).
    # Absurdity thresholds sit ~7x above healthy maxima (13 rad/s joint vel,
    # 1.3 m/s object vel at 8192 envs mid-lift), so false positives are impossible
    # for physical states while any collapse precursor trips them.
    op_local = op - env.scene.env_origins  # world positions grow with the env grid
    valid = (
        torch.isfinite(jp).all(dim=-1)
        & torch.isfinite(jv).all(dim=-1)
        & torch.isfinite(op).all(dim=-1)
        & torch.isfinite(ov).all(dim=-1)
        & (jp.abs().amax(dim=-1) < 1.0e2)
        & (jv.abs().amax(dim=-1) < 1.0e2)
        & (op_local.abs().amax(dim=-1) < 1.0e1)
        & (ov.abs().amax(dim=-1) < 5.0e1)
    )
    return ~valid


def _finite(reward: torch.Tensor) -> torch.Tensor:
    """Sanitize a reward term: no NaN/inf, magnitude clamped to a sane band.

    The env producing garbage is terminated by :func:`nonfinite_state`; the clamp
    keeps the one garbage step from reaching the critic as an outlier target.
    """
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0e4, 1.0e4)


def joint_vel_l2_safe(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """NaN-safe :func:`isaaclab.envs.mdp.joint_vel_l2` (see :func:`nonfinite_state`)."""
    return _finite(joint_vel_l2(env, asset_cfg))  # noqa: F405


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """1.0 when the object root is above ``minimal_height`` [m] (world z)."""
    obj = env.scene[object_cfg.name]
    return _finite(torch.where(obj.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0))


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reach shaping ``1 - tanh(|object - ee| / std)`` using the ee_frame sensor's first target."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    return _finite(1.0 - torch.tanh(distance / std))


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal-tracking shaping, gated on the object being lifted."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    return _finite((obj.data.root_pos_w.torch[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std)))


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
