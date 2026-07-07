# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation, termination, and reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

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


def cube_lifted(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """True when the cube center exceeds ``minimum_height`` [m] above the env origin."""
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return height > minimum_height


def ee_to_cube_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Kernelized right-wrist-to-cube distance, in (0, 1]. PLACEHOLDER shaping term."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos = robot.data.body_pos_w.torch[:, robot_cfg.body_ids[0], :]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - ee_pos, dim=1)
    return torch.exp(-dist / std)
