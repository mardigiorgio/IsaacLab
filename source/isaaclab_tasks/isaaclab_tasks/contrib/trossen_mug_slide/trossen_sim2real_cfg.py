# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real variant of the Trossen mug slide: the domain-randomized TEACHER
environment for the teacher->student distillation pipeline.

The randomization set mirrors the in-house-validated G1 walking recipe
(``core/velocity/velocity_env_cfg.py``), adapted to manipulation scale:
multiplicative log-uniform mass, per-shape friction bucketing, small COM
shifts, and interval nudges standing in for real-world disturbances. The
teacher keeps CLEAN privileged observations (exact object state) — dynamics
are randomized, sensors are not; heavy sensor corruption belongs to the
camera STUDENT, which is distilled from this teacher with rsl_rl's
distillation runner.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .trossen_mug_slide_env_cfg import TrossenMugSlideEnvCfg


def _apply_teacher_dr(cfg) -> None:
    """Attach the shared dynamics-randomization set to a task cfg's events."""
    # Friction: the mug's authored ceramic mu is 0.2; randomize around it.
    # Pads and table keep their authored values through the robot's own cfg.
    cfg.events.mug_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.1, 0.4),
            "dynamic_friction_range": (0.1, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Mass: multiplicative +-30% log-uniform (G1 pattern; scale-invariant).
    cfg.events.mug_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (1 / 1.3, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # COM: a few millimeters — liquid in a real mug, asymmetric handles. ICF
    # reads body_com live from the model, unlike the MJWarp path where the G1
    # recipe disables this term.
    cfg.events.mug_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.01, 0.01)},
        },
    )
    # Interval nudge on the mug: bumps, tablecloth drag, imperfect resets.
    cfg.events.mug_nudge = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        },
    )


@configclass
class TrossenMugSlideTeacherEnvCfg(TrossenMugSlideEnvCfg):
    """Slide teacher: slidev1 rewards + dynamics randomization, clean obs."""

    def __post_init__(self):
        super().__post_init__()
        _apply_teacher_dr(self)
