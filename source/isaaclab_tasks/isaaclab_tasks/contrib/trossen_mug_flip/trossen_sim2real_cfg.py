# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real variant of the Trossen mug flip: slight placement shifts only.

The rig protocol (trossen_mug_lift/REAL_SETUP.md) places the mug by tape
measure on the shared table mark — upside-down for this task — and every
real episode starts from the arm's home pose. The sim2real model is
therefore the residual of a CORRECT setup: a slightly shifted home pose
and a slightly shifted mug.

All reset banks are OFF here: their arm poses are authored against the
exact mug spawn and carry no placement Jacobian (the bedrock event's
``track_object_xy`` was never probed for this scene), so a jittered mug
under a banked start would misalign the ~1 %-of-path pinch window — and
the real rig cannot start banked anyway. Home-start noise (±0.03 rad)
still applies: it lives on the bank event's home path, which serves every
episode once the fractions are zero.
"""

from isaaclab.utils.configclass import configclass

from .trossen_mug_flip_env_cfg import TrossenMugFlipEnvCfg

S2R_MUG_XY_SHIFT_M = 0.01
S2R_MUG_YAW_SHIFT_RAD = 0.17

_BANK_EVENTS = (
    "reset_arm_grasp_bank",
    "reset_arm_lift_bank",
    "reset_arm_rotate_bank",
    "reset_arm_hover_bank",
    "reset_arm_home_via_bank",
)


@configclass
class TrossenMugFlipS2REnvCfg(TrossenMugFlipEnvCfg):
    """Flip under slight placement shifts only: mug ±1 cm / ±10° yaw on the
    taped mark (still inverted), home starts only (banks off), the authored
    ±0.03 rad home noise. Everything else identical to the campaign task."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-S2R_MUG_XY_SHIFT_M, S2R_MUG_XY_SHIFT_M),
            "y": (-S2R_MUG_XY_SHIFT_M, S2R_MUG_XY_SHIFT_M),
            "z": (0.0, 0.0),
            "yaw": (-S2R_MUG_YAW_SHIFT_RAD, S2R_MUG_YAW_SHIFT_RAD),
        }
        for name in _BANK_EVENTS:
            getattr(self.events, name).params["bank_fraction"] = 0.0
