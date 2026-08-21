# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug SLIDE: push the mug from its tape-measure spawn
to a commanded position on the table WITHOUT tipping it.

A sibling of the lift task sharing its scene, assets and mdp module. The
grasp-discovery machinery (rim targets, close bootstrap, pre-grasp bank, lift
income) is absent by construction: pushing is discoverable from plain reach
shaping, and the physics of interest is sustained frictional sliding — the
regime where fixed-step contact distorts normal forces and tips the mug.
"""

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .trossen_mug_lift_env_cfg import (
    CARRY_SPEED_MAX,
    OBJECT_REST_Z,
    TrossenMugLiftEnvCfg,
)

# Push speed cap [m/s]: below the lift task's carry cap — a controlled push
# stays under it, a smack exceeds it immediately. Loose enough that contact
# transients do not zero honest pushes.
PUSH_SPEED_MAX = 0.75


@configclass
class TrossenMugSlideEnvCfg(TrossenMugLiftEnvCfg):
    """Slide variant: lift rewards and grasp bootstraps replaced by upright
    on-table transport income; goals commanded at table height."""

    def __post_init__(self):
        super().__post_init__()

        # ONE mug position across lift and slide, deliberately: a single
        # tape-measure placement serves both hardware protocols.

        # Goals ON the table, far to ONE side of the spawn: the slide is a
        # cross-table traverse, not a nudge — a goal near spawn lets a parked
        # mug farm kernel income. Marker visualization on, so the target is
        # visible in the viewer and training clips.
        self.commands.object_pose.ranges.pos_z = (OBJECT_REST_Z, OBJECT_REST_Z)
        # B at the SIDE edge across the table (the mounted-camera side), one
        # mug-base-radius before the rail: lateral slide on the mug's line.
        # Probe-measured mapping (scripts/probes/probe_goal_map.py): command
        # frame IS the env frame; side edge x=+0.375, minus one mug radius.
        self.commands.object_pose.ranges.pos_x = (0.30, 0.30)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.debug_vis = True

        # Rewards: reach toward the mug root (for a push, low on the wall IS
        # the right approach point), then upright-on-table transport at two
        # scales. Tipping bleeds hard; there is nothing to close or lift.
        # Weights are an expected-value design, not taste: hover-never-touch
        # must earn LESS than a clumsy first push. Hover annuity = reach only
        # (0.5 x 150dt = 2.5/ep). A 5 cm push with a 50% mid-episode tip =
        # 2.5 + 1.7 (ratchet) - 1.2 (bleed) = 3.0 > 2.5, so contact is
        # EV-positive from the first unskilled attempt; a completed slide
        # earns ~40 via the arrival kernel. Tipping is priced mildly because
        # the upright/on-table/speed GATES already zero a tipped mug's
        # income — unpayment enforces style, the bleed only breaks ties.
        self.rewards.reaching_object = RewTerm(
            func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.5
        )
        # No tip fine at all: a tipped mug already earns nothing through the
        # gates and wastes its episode — measured at any fine size, the policy
        # hovers at the mug and refuses the touch. Style is enforced purely by
        # unpayment.
        self.rewards.mug_tipped_on_table = None
        self.rewards.looking_at_rim = None
        self.rewards.close_at_rim = None
        self.rewards.lifting_object = None
        # Transport = progress ratchet (a parked mug earns nothing, ever) plus
        # a tight arrival kernel that only pays AT the goal.
        # Upright gate at ~53 degrees: a pushed mug rocks well past 30 and a
        # gate that zeroes every real push teaches hovering, measured twice.
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_progress_on_table,
            params={
                "min_improvement": 0.005,
                "min_up_cos": 0.6,
                "max_speed": PUSH_SPEED_MAX,
                "command_name": "object_pose",
            },
            weight=5.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_goal_distance_on_table,
            params={
                "std": 0.05,
                "min_up_cos": 0.6,
                "max_speed": PUSH_SPEED_MAX,
                "command_name": "object_pose",
            },
            weight=16.0,
        )

        # No pre-contact bank for the slide: every episode starts from home.
        self.events.reset_arm_grasp_bank = None

        # A mug pushed off the slab is unrecoverable and the episode's income
        # is already forfeit under the gates — end it.
        self.terminations.object_out_of_bound = DoneTerm(
            func=mdp.object_off_table,
            params={"x_bound": 0.38, "y_bound": 0.62, "z_bound": 1.0},
        )


@configclass
class TrossenMugSlideEnvCfg_PLAY(TrossenMugSlideEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
