# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recorder terms for mimic datagen-info and subtask-signal annotation.

Shared by the annotation tool (replay-time recording), the teleop recorder
(record-time annotation), and consolidated_demo.
"""

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask start observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask start observations recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()
