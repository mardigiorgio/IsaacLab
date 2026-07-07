# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from .locomanipulation_g1_mimic_env import LocomanipulationG1MimicEnv


class PickPlaceFixedBaseG1MimicEnv(LocomanipulationG1MimicEnv):
    """Fixed-base upper-body-IK G1 pick-place Mimic environment.

    Shares the locomanipulation G1 action layout (left/right eef pose + hand
    joints via Pink IK) and eef observation names, and adds record-time
    subtask term signals so source demos can be recorded with
    ``--record_subtask_signals`` (no replay-based annotation needed).
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Subtask completion flags from the ``subtask_terms`` observation group.

        Args:
            env_ids: Environment indices to get the signals for. If None, all envs are considered.

        Returns:
            Dictionary of subtask termination flags for each subtask (final subtask needs no signal).
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        signals = {"grasp_1": subtask_terms["grasp_1"][env_ids]}
        # final subtask (place) needs no termination signal
        return signals
