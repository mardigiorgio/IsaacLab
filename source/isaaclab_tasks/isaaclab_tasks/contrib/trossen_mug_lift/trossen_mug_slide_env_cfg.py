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

# Push speed cap [m/s]: gentler than the lift task's carry cap — a controlled
# push never approaches it, a smack exceeds it immediately.
PUSH_SPEED_MAX = 0.5


@configclass
class TrossenMugSlideEnvCfg(TrossenMugLiftEnvCfg):
    """Slide variant: lift rewards and grasp bootstraps replaced by upright
    on-table transport income; goals commanded at table height."""

    def __post_init__(self):
        super().__post_init__()

        # Goals ON the table, in the arm's reachable band. z is the command
        # frame's, not the mug's; the transport term measures xy only.
        self.commands.object_pose.ranges.pos_z = (OBJECT_REST_Z, OBJECT_REST_Z)

        # Rewards: reach toward the mug root (for a push, low on the wall IS
        # the right approach point), then upright-on-table transport at two
        # scales. Tipping bleeds hard; there is nothing to close or lift.
        self.rewards.reaching_object = RewTerm(
            func=mdp.object_ee_distance, params={"std": 0.2}, weight=1.0
        )
        self.rewards.looking_at_rim = None
        self.rewards.close_at_rim = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_distance_on_table,
            params={"std": 0.3, "max_speed": PUSH_SPEED_MAX, "command_name": "object_pose"},
            weight=16.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_goal_distance_on_table,
            params={"std": 0.05, "max_speed": PUSH_SPEED_MAX, "command_name": "object_pose"},
            weight=16.0,
        )

        # Pushing needs no pre-grasp bank: any contact en route moves the mug
        # and the transport gradient takes over.
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
