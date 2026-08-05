# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preset-pose finger control for spatula teleoperation.

The stock :class:`TriHandMotionControllerRetargeter` maps trigger/squeeze onto
finger joints through a fixed analytic formula. That is fine for a power grasp
but useless for the claw: the operator needs the digits to land in the AUTHORED
pregrasp straddle (thumb one side of the handle, index+middle the other) and
then rake shut, and no scalar formula reproduces that specific pose.

This retargeter interpolates between three named poses instead:

    no buttons      -> OPEN      (digits clear of the table)
    trigger         -> PREGRASP  (the authored claw straddle)
    trigger+squeeze -> CLOSED    (claw raked shut on the handle)

Both inputs are analog, so the operator gets continuous control of how far into
each pose the hand goes rather than a snap.
"""

from __future__ import annotations

import numpy as np
from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, RobotHandJoints

##
# Poses — TriHand 7-DOF output order
##

HAND_JOINT_NAMES = [
    "thumb_rotation",
    "thumb_proximal",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
]
"""TriHand output order. Positionally mapped by the pipeline's ``TensorReorderer``
onto ``thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1``."""

OPEN_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Digits straight — the hand can be flown around without fouling the table."""

PREGRASP_POSE = (-0.5265, -0.5245, -0.8727, 1.1309, 1.2593, 1.0978, 1.2207)
"""The REAL caged pregrasp, read straight out of ``grasp_map.pt`` stage
``pre_grasp_caged`` — the active reset map, which is what the RL env actually
resets into. Do NOT source this from ``PREGRASP_JOINT_POS``: that dict has the
index/middle knuckles at 0.0017 rad (dead straight) where the reset map has them
at 1.13/1.10 rad (bent around the handle), so a pose built from the dict leaves
the first finger joint visibly inert."""

LEFT_IDLE_POSE = (0.0, 0.0, 0.0, 0.35, 0.55, 0.35, 0.55)
"""Left hand parking pose — digits softly curled so they do not stick out into
the workspace. The left arm is out of the IK chain entirely; this only stops the
left fingers from splaying."""

CLOSED_POSE = (-0.8500, -0.8000, -1.2500, 1.4500, 1.5500, 1.4200, 1.5500)
"""Claw raked shut — a further curl from the caged pregrasp above, staying
inside every joint limit: ``thumb_0 [-1.047, 1.047]``, ``thumb_1 [-1.047, 0.724]``,
``thumb_2 [-1.745, 0]``, ``index_0/middle_0 [0, 1.571]``, ``index_1/middle_1
[0, 1.745]``. Still the value most worth retuning from in-headset."""


class TriHandPosePresetConfig:
    """Configuration for :class:`TriHandPosePresetRetargeter`.

    Args:
        hand_joint_names: Joint names in TriHand output order.
        controller_side: Which controller drives this hand, ``"left"`` or ``"right"``.
        open_pose: Joint targets [rad] with neither input pressed.
        pregrasp_pose: Joint targets [rad] at full trigger.
        closed_pose: Joint targets [rad] at full trigger AND full squeeze.
    """

    def __init__(
        self,
        hand_joint_names: list[str],
        controller_side: str = "right",
        open_pose: tuple[float, ...] = OPEN_POSE,
        pregrasp_pose: tuple[float, ...] = PREGRASP_POSE,
        closed_pose: tuple[float, ...] = CLOSED_POSE,
    ) -> None:
        self.hand_joint_names = hand_joint_names
        self.controller_side = controller_side
        self.open_pose = open_pose
        self.pregrasp_pose = pregrasp_pose
        self.closed_pose = closed_pose


class TriHandPosePresetRetargeter(BaseRetargeter):
    """Blend the TriHand between open, pregrasp, and closed poses from two analog inputs.

    ``trigger`` interpolates open -> pregrasp; ``squeeze`` then interpolates that
    result -> closed. Squeeze is scaled by trigger so that squeezing without the
    trigger cannot jump the hand straight to the closed pose from open, which
    would skip the straddle and let the digits close on top of the handle instead
    of around it.
    """

    def __init__(self, config: TriHandPosePresetConfig, name: str) -> None:
        self._config = config
        self._hand_joint_names = config.hand_joint_names
        self._controller_side = config.controller_side.lower()
        if self._controller_side not in ("left", "right"):
            raise ValueError(f"controller_side must be 'left' or 'right', got: {self._controller_side}")
        self._open = np.asarray(config.open_pose, dtype=float)
        self._pregrasp = np.asarray(config.pregrasp_pose, dtype=float)
        self._closed = np.asarray(config.closed_pose, dtype=float)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Optional controller input for the configured side."""
        return {f"controller_{self._controller_side}": OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        """Seven joint targets for this hand."""
        return {"hand_joints": RobotHandJoints(f"hand_joints_{self._controller_side}", self._hand_joint_names)}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Blend the presets from trigger/squeeze and write the joint targets."""
        out = outputs["hand_joints"]
        controller = inputs[f"controller_{self._controller_side}"]

        # controller not tracked this frame: hold the open pose rather than
        # zeroing, so a dropout does not fling the fingers
        if controller.is_none:
            for i, v in enumerate(self._open):
                out[i] = float(v)
            return

        trigger = float(np.clip(controller[ControllerInputIndex.TRIGGER_VALUE], 0.0, 1.0))
        squeeze = float(np.clip(controller[ControllerInputIndex.SQUEEZE_VALUE], 0.0, 1.0))

        pose = self._open + trigger * (self._pregrasp - self._open)
        pose = pose + (squeeze * trigger) * (self._closed - self._pregrasp)

        for i in range(min(len(self._hand_joint_names), pose.shape[0])):
            out[i] = float(pose[i])


def build_spatula_teleop_pipeline():
    """Build the retargeting pipeline for the spatula teleop task.

    Mirrors the G1 upper-body IK pipeline (two ``Se3AbsRetargeter`` wrist
    trackers, a ``TensorReorderer`` flattening to the 28D action tensor) but
    swaps the RIGHT hand onto :class:`TriHandPosePresetRetargeter`. The left hand
    keeps the stock controller mapping — it is inert for this task.

    Returns:
        Tuple of the ``OutputCombiner`` pipeline and the retargeters to expose in
        the tuning UI.
    """
    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # wrist pose trackers — offsets copied verbatim from the base G1 IK pipeline
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=45.0,
            target_offset_pitch=180.0,
            target_offset_yaw=-90.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=-135.0,
            target_offset_pitch=0.0,
            target_offset_yaw=90.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # LEFT HAND: pinned, not driven. Every preset is the same relaxed pose, so
    # the left controller cannot splay the left fingers into the scene while the
    # operator is working the right hand.
    left_trihand = TriHandPosePresetRetargeter(
        TriHandPosePresetConfig(
            hand_joint_names=HAND_JOINT_NAMES,
            controller_side="left",
            open_pose=LEFT_IDLE_POSE,
            pregrasp_pose=LEFT_IDLE_POSE,
            closed_pose=LEFT_IDLE_POSE,
        ),
        name="trihand_left_idle",
    )
    connected_left_trihand = left_trihand.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    # right hand: preset poses on trigger / squeeze
    right_trihand = TriHandPosePresetRetargeter(
        TriHandPosePresetConfig(hand_joint_names=HAND_JOINT_NAMES, controller_side="right"),
        name="trihand_right_presets",
    )
    connected_right_trihand = right_trihand.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_hand_elements = [f"l_{n}" for n in HAND_JOINT_NAMES]
    right_hand_elements = [f"r_{n}" for n in HAND_JOINT_NAMES]

    # MEASURED slot -> joint mapping, not the one the inherited pipeline assumes.
    # The action term applies the 14 hand values in the ARTICULATION's joint
    # order (alphabetical per hand: index_0, index_1, middle_0, middle_1,
    # thumb_0, thumb_1, thumb_2), NOT in hand_joint_names order. The inherited
    # builder interleaves them as [all 0-joints, all 1-joints, thumb_2s], so
    # every finger value landed on the wrong joint and the right index was
    # driven by nothing at all. Verified by fingerprinting each slot and reading
    # back which joint moved.
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            "l_index_proximal",  # slot 14 -> left_hand_index_0_joint
            "l_index_distal",  # slot 15 -> left_hand_index_1_joint
            "l_middle_proximal",  # slot 16 -> left_hand_middle_0_joint
            "l_middle_distal",  # slot 17 -> left_hand_middle_1_joint
            "l_thumb_rotation",  # slot 18 -> left_hand_thumb_0_joint
            "l_thumb_proximal",  # slot 19 -> left_hand_thumb_1_joint
            "l_thumb_distal",  # slot 20 -> left_hand_thumb_2_joint
            "r_index_proximal",  # slot 21 -> right_hand_index_0_joint
            "r_index_distal",  # slot 22 -> right_hand_index_1_joint
            "r_middle_proximal",  # slot 23 -> right_hand_middle_0_joint
            "r_middle_distal",  # slot 24 -> right_hand_middle_1_joint
            "r_thumb_rotation",  # slot 25 -> right_hand_thumb_0_joint
            "r_thumb_proximal",  # slot 26 -> right_hand_thumb_1_joint
            "r_thumb_distal",  # slot 27 -> right_hand_thumb_2_joint
        ]
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_elements,
            "right_hand_joints": right_hand_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_trihand.output("hand_joints"),
            "right_hand_joints": connected_right_trihand.output("hand_joints"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_se3, right_se3]
