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


def object_off_table(
    env: ManagerBasedRLEnv,
    x_bound: float,
    y_bound: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose object left the tabletop footprint (env-local xy).

    The rig's tabletop slab rests directly on the ground plane, so an object knocked
    off the table always ends up on the floor beyond the slab footprint; xy bounds
    alone decide the condition, independent of lift height or reset settling.
    """
    obj = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return (pos_local[:, 0].abs() > x_bound) | (pos_local[:, 1].abs() > y_bound)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """1.0 when the object root is above ``minimal_height`` [m] (world z)."""
    obj = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


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
    return 1.0 - torch.tanh(distance / std)


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
    return (obj.data.root_pos_w.torch[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std))


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
    return object_is_lifted(env, minimal_height, object_cfg) * object_held(
        env, threshold, object_cfg, ee_frame_cfg
    )


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
