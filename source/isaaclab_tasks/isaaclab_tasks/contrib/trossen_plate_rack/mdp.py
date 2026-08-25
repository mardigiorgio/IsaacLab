# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the plate tasks: the mug lift's set, verbatim.

The plate tasks are deliberately the same MDP on different geometry — the
rim-circle reach kernel, the grasp-gated ratchet, the success gate and the
containment terminations all read the scene through entity names and live
poses, none of them mug-specific. Anything plate-only added later lives here.
"""

from isaaclab_tasks.contrib.trossen_mug_lift.mdp import *  # noqa: F401,F403


import math

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

# The success gates are the committed evaluator's (probe_eval_success.py):
# 5 cm to the commanded pose, upright within acos(0.87) ~= 0.515 rad. The
# online Metrics/success_rate is the per-episode "ever inside both gates"
# fraction; delivery-with-hold remains the evaluator's post-hoc criterion.
SUCCESS_POS_THRESHOLD = 0.05
SUCCESS_ORI_THRESHOLD = math.acos(0.87)


class ObjectPoseSuccessCommand(UniformPoseCommand):
    """UniformPoseCommand whose error - and therefore whose success metric -
    is measured on the OBJECT's root pose, not the commanded robot body.

    The stock command gates success on the EE reaching the commanded pose,
    which counts an empty gripper at the goal as a success. The task's claim
    is about the object arriving, so the metric must read the object."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object = env.scene["object"]

    def _compute_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self._object.data.root_pos_w.torch,
            self._object.data.root_quat_w.torch,
        )
        return torch.linalg.norm(pos_error, dim=-1), torch.linalg.norm(rot_error, dim=-1)
