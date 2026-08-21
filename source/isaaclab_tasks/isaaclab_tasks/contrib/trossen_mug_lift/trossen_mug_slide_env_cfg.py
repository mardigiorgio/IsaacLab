# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug SLIDE: push the mug from its tape-measure spawn
to a commanded position on the table WITHOUT tipping it.

Shares the lift's scene, assets, actions and observations, but every
BEHAVIORAL manager (rewards, events, terminations, curriculum) is declared
complete in this file and replaces the inherited one wholesale — nothing the
lift adds can flow into the slide implicitly. The grasp machinery is absent
by construction: pushing is discoverable from plain reach shaping, and the
physics of interest is sustained frictional sliding — the regime where
fixed-step contact distorts normal forces and tips the mug.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .trossen_mug_lift_env_cfg import (
    OBJECT_REST_Z,
    TrossenMugLiftEnvCfg,
)

# Push speed cap [m/s]: below the lift task's carry cap — a controlled push
# stays under it, a smack exceeds it immediately. Loose enough that contact
# transients do not zero honest pushes.
PUSH_SPEED_MAX = 0.75


@configclass
class SlideObservationsCfg:
    """The slide's observation set, pinned verbatim to the slidev1 recipe.

    The lift moved to ABSOLUTE joint angles because its grasp-bank reset
    retargets default_joint_pos per env; the slide has no bank and its
    trained baselines (the slidev1 K-ladder) observed relative-to-default
    angles, so the frozen recipe is declared here rather than inherited."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # The raw last action feeds back into the policy input; unclipped, the
        # loop goes exponentially unstable once the network's feedback gain
        # crosses 1, which ends in a NaN in the PPO update. The clip bounds the
        # loop; 6 sigma does not bind for a healthy policy.
        actions = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SlideRewardsCfg:
    """The slide's complete reward set — self-contained, no lift terms.

    Weights are an expected-value design, not taste: hover-never-touch must
    earn LESS than a clumsy first push (hover annuity = reach only, ~2.5 per
    episode; a 5 cm push with a 50% mid-episode failure still nets more).
    Style is enforced by UNPAYMENT: the upright/on-table/speed gates zero a
    tipped, airborne, or smacked mug's income; there is no tip fine, which at
    any size taught hovering (measured twice)."""

    # Reach toward the mug root: for a push, low on the wall IS the right
    # approach point.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.5)
    # Transport = progress ratchet (a parked mug earns nothing, ever): pays
    # per 5 mm of NEW episode-best planar goal distance, upright (gate at
    # ~53 degrees — a pushed mug rocks well past 30, and a gate that zeroes
    # every real push teaches hovering), on the table, at push speed.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_progress_on_table,
        params={
            "min_improvement": 0.005,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
            "command_name": "object_pose",
        },
        weight=5.0,
    )
    # Tight arrival kernel that only pays AT the goal, same gates.
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_on_table,
        params={
            "std": 0.05,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
            "command_name": "object_pose",
        },
        weight=16.0,
    )
    # Scraping or pressing the tabletop with any robot body is never part of
    # a correct push.
    table_scrape = RewTerm(
        func=mdp.body_contact, weight=-2.0, params={"sensor_name": "arm_table_contact", "threshold": 1.0}
    )
    # Erratic-arm suppression, gripper exempt; jerk targets direction-flips
    # specifically, leaving a smooth sustained push unpenalized.
    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-3e-3)
    action_jerk = RewTerm(func=mdp.arm_action_jerk_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["follower_left_joint_[0-5]"])},
    )


@configclass
class SlideTerminationsCfg:
    """The slide's complete termination set."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Fires only on constraint blowups through the arm.
    robot_abnormal = DoneTerm(func=mdp.robot_state_abnormal, params={"max_joint_vel": 25.0})
    # Solver-level containment valve (see the lift cfg's notes).
    physics_diverged = DoneTerm(func=mdp.physics_diverged)
    # A mug pushed off the slab is unrecoverable and the episode's income is
    # already forfeit under the gates — end it.
    object_out_of_bound = DoneTerm(
        func=mdp.object_off_table,
        params={"x_bound": 0.38, "y_bound": 0.62, "z_bound": 1.0},
    )


@configclass
class SlideEventCfg:
    """The slide's complete event set: home starts only, fixed mug spawn."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # ZERO spawn jitter, deliberately: the tape-measure protocol.
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class SlideCurriculumCfg:
    """No curriculum: the slide trains from home starts throughout."""


@configclass
class TrossenMugSlideEnvCfg(TrossenMugLiftEnvCfg):
    """Slide task: shared scene/actions/observations from the lift base;
    behavioral managers replaced wholesale by the slide's own."""

    observations: SlideObservationsCfg = SlideObservationsCfg()
    rewards: SlideRewardsCfg = SlideRewardsCfg()
    terminations: SlideTerminationsCfg = SlideTerminationsCfg()
    events: SlideEventCfg = SlideEventCfg()
    curriculum: SlideCurriculumCfg = SlideCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # ONE mug position across lift and slide, deliberately: a single
        # tape-measure placement serves both hardware protocols.

        # Goal ON the table at the SIDE edge across from the mounted camera,
        # one mug-base-radius before the rail, nudged inward per the operator
        # view. Probe-measured mapping (scripts/probes/probe_goal_map.py):
        # command frame IS the env frame. Marker visualization on, so the
        # target is visible in the viewer and training clips.
        self.commands.object_pose.ranges.pos_z = (OBJECT_REST_Z, OBJECT_REST_Z)
        self.commands.object_pose.ranges.pos_x = (0.285, 0.285)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.debug_vis = True


@configclass
class TrossenMugSlideEnvCfg_PLAY(TrossenMugSlideEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
