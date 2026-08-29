# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The start-state recipe that cracked the mug lift, as shared machinery.

Three interlocking pieces, applied together or not at all:

1. **Wide-scale gripper exploration** (the discovery constraint): the
   gripper action scale must put the full open-to-close travel inside ~1
   sigma of the initial policy's exploration, or no reward can ever elicit
   a close. The mug lift's ActionsCfg carries the arithmetic; scenes reuse
   that cfg, and :func:`apply_reverse_curriculum` asserts the scale rather
   than trusting inheritance.
2. **IK pre-grasp bank**: half of all resets teleport the arm to a
   scene-specific pre-grasp (a hover short of contact, never a seeded
   grasp), re-solved per env for the object's actual placement through the
   bank pose's XY Jacobian.
3. **Reverse curriculum** (Florensa): bank starts begin AT the pre-grasp
   and anneal back toward home over the first ~100 iterations; the anneal
   must end soon after close-discovery saturates or the run pays the
   moving-target tax (see the mug lift's CurriculumCfg for the measured
   horizon reasoning).

A scene may run without a bank (home scatter + wide exploration only) —
that is the un-cracked bootstrap mode, and it is how a scene runs until its
bank pose has passed the scripted close-and-raise existence proof. A bank
pose that never passed that probe is a liability, not a head start.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp

# The measured floor of the discovery arithmetic (see ActionsCfg): at
# init_std 0.5, a gripper scale below this leaves the closed position
# multiple sigma from the start pose and closing is never sampled.
_MIN_DISCOVERY_SCALE = 0.035


def apply_reverse_curriculum(
    cfg,
    bank_pose: dict[str, float],
    bank_xy_jacobian: list[list[float]] | None = None,
    nominal_object_xy: tuple[float, float] | None = None,
    bank_fraction: float = 0.5,
    end_step: int = 2_400,
    gripper_offset: float | None = None,
    home_offset_pose: dict[str, float] | None = None,
    home_noise: float = 0.0,
) -> None:
    """Attach the bank event and its anneal to a task cfg, with the guards.

    Args:
        cfg: A ManagerBasedRLEnvCfg whose actions follow the mug family's
            ActionsCfg and whose events carry ``randomize_arm_start``.
        bank_pose: Joint-name -> position pre-grasp hover for THIS scene.
            Must have passed probe_scripted_grasp on this scene first.
        bank_xy_jacobian: Optional 6x2 joint response to planar object
            shift, for placement-DR tracking (probe_bank_jacobian).
        nominal_object_xy: Object placement the bank pose was authored at;
            required with ``bank_xy_jacobian``.
        bank_fraction: Fraction of resets that start from the bank.
        end_step: Anneal horizon in env steps (iterations x steps_per_env).
    """
    scale = float(cfg.actions.gripper_action.scale)
    if scale < _MIN_DISCOVERY_SCALE:
        raise ValueError(
            f"gripper_action.scale={scale} is below the discovery floor "
            f"{_MIN_DISCOVERY_SCALE}: closing would be a multi-sigma action the "
            "initial policy cannot sample, and the bank cannot rescue that."
        )
    if not hasattr(cfg.events, "randomize_arm_start"):
        raise ValueError(
            "events must declare randomize_arm_start BEFORE the bank event: the "
            "home half's start diversity is part of the recipe, not an option."
        )
    params: dict = {
        "pose": dict(bank_pose),
        "bank_fraction": float(bank_fraction),
        "noise": 0.0,
        "alpha_min": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    if gripper_offset is not None:
        params["gripper_offset"] = float(gripper_offset)
    if home_offset_pose is not None:
        params["home_offset_pose"] = dict(home_offset_pose)
    if home_noise:
        params["home_noise"] = float(home_noise)
    if bank_xy_jacobian is not None:
        if nominal_object_xy is None:
            raise ValueError("bank_xy_jacobian requires nominal_object_xy.")
        params["track_object_xy"] = [list(row) for row in bank_xy_jacobian]
        params["nominal_object_pos"] = tuple(nominal_object_xy)
    cfg.events.reset_arm_grasp_bank = EventTerm(func=mdp.reset_arm_reverse_curriculum, mode="reset", params=params)

    @configclass
    class _BedrockCurriculumCfg:
        grow_approach = CurrTerm(
            func=mdp.anneal_reverse_curriculum,
            params={"start_step": 0, "end_step": int(end_step), "event_name": "reset_arm_grasp_bank"},
        )

    cfg.curriculum = _BedrockCurriculumCfg()
