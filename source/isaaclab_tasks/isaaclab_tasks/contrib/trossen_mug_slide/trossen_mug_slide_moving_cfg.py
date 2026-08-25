# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen mug slide, slidev2: the goal MOVES.

The v1 fixed goal let the policy choose the push speed, and the trained
solution was an unphysically fast shove. Here the commanded goal starts ON
the mug's tape-measured spawn and glides to the same final target at
``GOAL_SPEED``; the tracking income exists only near the CURRENT goal, so
the mug must follow the pace — a controlled, sustained frictional slide,
which is both the physical protocol and the harder regime for fixed step.

Everything not about the goal is slidev1 verbatim (actions, observations,
terminations, events, scene, control rate); v1 stays registered and frozen.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_X,
    _SPAWN_Y,
    EE_LINK,
    OBJECT_REST_Z,
)

from . import mdp
from .trossen_mug_slide_env_cfg import (
    PUSH_SPEED_MAX,
    SlideActionsCfg,
    SlideCurriculumCfg,
    SlideEventCfg,
    SlideObservationsCfg,
    SlideTerminationsCfg,
    TrossenMugSlideEnvCfg,
)

# The v1 tape-measured final target (see SlideCommandsCfg for its siting).
FINAL_GOAL = (0.285, 0.0)
# Goal glide speed [m/s]: 0.305 m of travel in ~3.8 s, inside a 6 s episode
# with reach and settle margin. A deliberate paced push; PROVISIONAL until
# Marco confirms the number.
GOAL_SPEED = 0.08


@configclass
class SlideMovingCommandsCfg:
    object_pose = mdp.MovingPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        start_pos=(_SPAWN_X, _SPAWN_Y, OBJECT_REST_Z),
        goal_speed=GOAL_SPEED,
        ranges=mdp.MovingPoseCommandCfg.Ranges(
            pos_x=(FINAL_GOAL[0], FINAL_GOAL[0]),
            pos_y=(FINAL_GOAL[1], FINAL_GOAL[1]),
            pos_z=(OBJECT_REST_Z, OBJECT_REST_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # Online Metrics/success_rate against the FINAL target, tilt-only
        # orientation (the mug may spin about z while sliding).
        position_success_threshold=0.05,
        orientation_success_threshold=0.5156,
    )


@configclass
class SlideMovingRewardsCfg:
    """slidev2 rewards: v1's expected-value design with the ratchet replaced
    by the moving-goal tracking kernel.

    v1's ratchet existed because an absolute kernel on a FIXED goal pays a
    parked mug; with a moving goal a parked mug's income vanishes as the
    goal walks away, so the absolute kernel is safe and IS the pacing
    mechanism. Arrival and rest read the final target, not the moving one —
    they must not pay mid-path."""

    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50,
        params={"term_keys": ["robot_abnormal", "physics_diverged"]},
    )
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.5)
    # Pacing income: pays near the CURRENT (gliding) goal, upright, on the
    # table, at push speed. std 0.08 ~= one goal-second of lag tolerance.
    goal_following = RewTerm(
        func=mdp.object_goal_distance_on_table,
        params={
            "std": 0.08,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
            "command_name": "object_pose",
        },
        weight=5.0,
    )
    # Arrival annuity at the FINAL spot, v1's kernel and gates.
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_fixed_goal_distance_on_table,
        params={
            "std": 0.05,
            "goal_pos": FINAL_GOAL,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
        },
        weight=16.0,
    )
    arm_settled = RewTerm(
        func=mdp.arm_settled_at_fixed_goal,
        weight=2.0,
        params={
            "std": 0.05,
            "vel_std": 4.0,
            "goal_pos": FINAL_GOAL,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
        },
    )
    table_scrape = RewTerm(
        func=mdp.body_contact, weight=-2.0, params={"sensor_name": "arm_table_contact", "threshold": 1.0}
    )
    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-3e-3)
    action_jerk = RewTerm(func=mdp.arm_action_jerk_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["follower_left_joint_[0-5]"])},
    )


@configclass
class TrossenMugSlideMovingEnvCfg(TrossenMugSlideEnvCfg):
    """slidev2 assembled as slidev1 with the moving goal program."""

    commands: SlideMovingCommandsCfg = SlideMovingCommandsCfg()
    rewards: SlideMovingRewardsCfg = SlideMovingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 6 s: 3.8 s of goal travel plus reach and settle margin.
        self.episode_length_s = 6.0


@configclass
class TrossenMugSlideMovingEnvCfg_PLAY(TrossenMugSlideMovingEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
