# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from .locomanipulation_g1_mimic_env import LocomanipulationG1MimicEnv

# Hand-joint action indices, derived from G1_UPPER_BODY_IK_ACTION_CFG's
# hand_joint_names order (INTERLEAVED left/right, offset by the 14 pose dims).
# The parent class's contiguous 14:21 / 21:28 split mixes physical hands; that
# is harmless when both "hands" are resampled at the same source time index,
# but corrupts grasps once per-arm subtask segmentation diverges (this task:
# two right-arm subtasks vs one left-arm subtask).
_LEFT_HAND_ACTION_IDX = [14, 15, 16, 20, 21, 22, 26]
_RIGHT_HAND_ACTION_IDX = [17, 18, 19, 23, 24, 25, 27]


class PickPlaceFixedBaseG1MimicEnv(LocomanipulationG1MimicEnv):
    """Fixed-base upper-body-IK G1 pick-place Mimic environment.

    Shares the locomanipulation G1 action layout (left/right eef pose + hand
    joints via Pink IK) and eef observation names, and adds record-time
    subtask term signals so source demos can be recorded with
    ``--record_subtask_signals`` (no replay-based annotation needed).
    """

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Assemble an action, scattering each hand's joints to their interleaved indices."""
        # let the parent build the pose part (and apply noise); its gripper
        # concatenation is then overwritten with the correct index mapping
        action = super().target_eef_pose_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict)
        action[_LEFT_HAND_ACTION_IDX] = gripper_action_dict["left"]
        action[_RIGHT_HAND_ACTION_IDX] = gripper_action_dict["right"]
        return action

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-hand joint actions using the interleaved hand-joint index map."""
        return {"left": actions[..., _LEFT_HAND_ACTION_IDX], "right": actions[..., _RIGHT_HAND_ACTION_IDX]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Subtask completion flags from the ``subtask_terms`` observation group.

        Args:
            env_ids: Environment indices to get the signals for. If None, all envs are considered.

        Returns:
            Dictionary of subtask termination flags for each subtask (final subtask needs no signal).
        """
        if env_ids is None:
            env_ids = slice(None)

        # every named subtask signal must have a matching subtask_terms obs
        # term; final subtasks (signal None) need no entry
        subtask_terms = self.obs_buf["subtask_terms"]
        signals: dict[str, torch.Tensor] = {}
        for subtask_configs in self.cfg.subtask_configs.values():
            for subtask_config in subtask_configs:
                name = subtask_config.subtask_term_signal
                if name is not None and name not in signals:
                    signals[name] = subtask_terms[name][env_ids]
        return signals
