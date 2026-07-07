# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_grasped_heuristic(
    env: ManagerBasedRLEnv,
    height_delta: float = 0.03,
    dist_threshold: float = 0.25,
    palm_link_name: str = "right_hand_palm_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """True when the object has been lifted off its rest height near the given palm.

    Record-time grasp signal: the object center is more than ``height_delta``
    [m] above its default (resting) height and within ``dist_threshold`` [m]
    of the palm link, i.e. it moved because the hand picked it up.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    rest_z = obj.data.default_root_state.torch[:, 2] + env.scene.env_origins[:, 2]
    lifted = obj.data.root_pos_w.torch[:, 2] > rest_z + height_delta
    palm_idx = robot.body_names.index(palm_link_name)
    palm_pos = robot.data.body_pos_w.torch[:, palm_idx, :]
    near = torch.linalg.vector_norm(obj.data.root_pos_w.torch - palm_pos, dim=1) < dist_threshold
    return lifted & near


@configclass
class SubtaskTermsObsCfg(ObsGroup):
    """Subtask termination signals for record-time annotation."""

    grasp_1 = ObsTerm(func=object_grasped_heuristic)

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class PickPlaceFixedBaseG1MimicEnvCfg(FixedBaseUpperBodyIKG1EnvCfg, MimicEnvCfg):
    """Configuration for the fixed-base upper-body-IK G1 pick-place Mimic environment."""

    def __post_init__(self):
        super().__post_init__()

        # subtask signals are computed live so demos can be recorded with
        # --record_subtask_signals (no replay-based annotation pass needed)
        self.observations.subtask_terms = SubtaskTermsObsCfg()

        self.datagen_config.name = "demo_src_g1_fixedbase_pickplace_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 1

        # right arm: pick the object (ends on grasp_1), then place it (final)
        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasp_1",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]

        # left arm: single passive subtask (mirrors the locomanip config)
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        ]
