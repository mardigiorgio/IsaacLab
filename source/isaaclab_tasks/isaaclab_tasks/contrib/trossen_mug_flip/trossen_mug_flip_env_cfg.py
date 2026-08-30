# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug FLIP: the mug starts upside down at its
tape-measure spot, handle toward the arm, and the policy must right it BY
THE HANDLE at that same spot, then let go and settle.

A standalone task package on the slide's pattern: every manager declared
here, platform pieces (rig scene, physics stack, wiring constants) imported
from the mug-lift package. Style is enforced the slide's way — gates unpay
wrong behavior (shoved, flung, airborne mugs earn nothing) and the two
standing fines target the arm: pressing the mug anywhere but the handle,
and scraping the table.

Sim2real: ONE fixed spawn (the lift/slide tape-measure spot), the flip in
place — start pose and goal pose are both physically measurable on the
real table.
"""

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_X,
    _SPAWN_Y,
    ARM_JOINTS,
    BASE_LINK,
    EE_LINK,
    EE_TCP_OFFSET,
    GRIPPER_JOINT,
    GRIPPER_JOINT_R,
    MUG_RIM_HEIGHT,
    OBJECT_REST_Z,
    TrossenMugLiftPhysicsCfg,
    TrossenMugLiftSceneCfg,
)

from isaaclab_tasks.contrib.trossen_mug_lift.bedrock import apply_reverse_curriculum
from isaaclab_tasks.contrib.trossen_mug_flip.flip_fsm_core import SUCCESS_BONUS
from isaaclab_tasks.contrib.trossen_mug_tree.trossen_mug_tree_env_cfg import ARM_FINISH_POSE

from . import mdp

# Pre-grasp HOVER for the inverted mug's handle, generated and VALIDATED by
# probe_generate_bank.py on this scene (2026-08-28): fingers pitched 60 degrees
# down from the base side, jaws closing along env x across the 11.7 mm handle
# bar, banked AT the grasp point (jaws open around the bar, 0.000 N at spawn):
# a 2 cm hover is fine for a 10 cm mug body, not for a 12 mm bar.
# Existence proof: close -> jaws 5.0/5.0 mm on the bar, 111 N per pad on the
# HANDLE pieces, 0.00 N on the body, mug raised 143 mm by the handle (HELD).
# A horizontal approach is kinematically impossible (joint_3 saturates +1.57).
# Joints are PRE-COMPENSATED by the measured PD gravity sag (q_int at the grasp:
# j1 -0.0315, j2 +0.0296, j3 +0.0129) so the PD-held start puts the fingertips
# mid-bar (env y 0.062, z 0.071), not on the curved lower arm where a pinch slips.
# LIFTED-HELD bank (probe_flip_low_pinch_chain, 2026-08-28): low pinch, shoulder
# -0.30 rad, settled: mug root 10.8 cm above its rest, held 100%.
FLIP_LIFTED_BANK_POSE = {
    "follower_left_joint_0": -0.0001,
    "follower_left_joint_1": 1.4870,
    "follower_left_joint_2": 0.7428,
    "follower_left_joint_3": 0.0595,
    "follower_left_joint_4": 0.0001,
    "follower_left_joint_5": -0.0002,
    "follower_left_left_carriage_joint": 0.0060,
    "follower_left_right_carriage_joint": 0.0060,
}
FLIP_LIFTED_MUG_POSE = (-0.0199, 0.0562, 0.2266, 0.6685, 0.6697, -0.229, 0.2283)  # env frame, (x y z qx qy qz qw)

# ROTATED-held bank (probe_flip_low_pinch_chain, 2026-08-28): low pinch, lifted,
# forearm roll -> 1.4 and wrist roll -> -3.0 over 60 steps: up_cos +0.78..0.82,
# FSM ROTATED 100%, still held (fingers pointing ~40 deg up, hand below the handle).
FLIP_ROTATED_BANK_POSE = {
    "follower_left_joint_0": -0.0001,
    "follower_left_joint_1": 1.4758,
    "follower_left_joint_2": 0.7442,
    "follower_left_joint_3": 1.4397,
    "follower_left_joint_4": -0.0001,
    "follower_left_joint_5": -2.9499,
    "follower_left_left_carriage_joint": 0.0060,
    "follower_left_right_carriage_joint": 0.0060,
}
FLIP_ROTATED_MUG_POSE = (-0.0065, 0.0624, 0.3471, -0.1954, 0.2644, -0.7243, -0.606)

# LOW (rim-end) pinch (probe_generate_bank tcp z 0.0965 + probe_flip_low_pinch_chain,
# 2026-08-28): fingertips ~4 cm above the table on the bar, 60-degree approach.
# Pinched 100%, held after the lift 100% at every squeeze, holds through every
# wrist motion tested; the mid-height pinch (handle_middle) held 17-75%.
# Open-jaw GRASP-POINT rung (2026-08-29): the low-pinch grasp pose with the jaws
# OPEN (0.044) and no squeeze offset -- fingertips at the bar, a plain close
# pinches (probe_flip_bank_stages: 100%). The hover 2 cm back never produced a
# pinch: the policy held still there with the gripper opening. This rung is
# annealed toward home (jaws interpolate open, so no intermediate start squeezes
# the mug) to teach the approach; the squeezed grasp bank cannot be annealed.
FLIP_OPEN_GRASP_BANK_POSE = {
    "follower_left_joint_0": -0.0001,
    "follower_left_joint_1": 1.7416,
    "follower_left_joint_2": 0.7745,
    "follower_left_joint_3": 0.0754,
    "follower_left_joint_4": 0.0000,
    "follower_left_joint_5": -0.0001,
    # jaws 6 mm from the bar (0.012: faces at +-12 mm, bar +-5.85 mm), NOT fully open:
    # the carriage velocity limit (0.0875 m/s) makes a close from 44 mm a 13-step
    # commitment no exploration sample sustains (fsm20: pinch at baseline for 200
    # iterations); from 12 mm a single sampled close reaches the bar.
    "follower_left_left_carriage_joint": 0.0120,
    "follower_left_right_carriage_joint": 0.0120,
}

# Via-point 8 cm above the low-pinch grasp along the same 60-degree approach
# (probe_generate_bank tcp z +0.08, IK 0.2 mm, 2026-08-29): the approach rung's
# anneal anchor. Jaws at 12 mm so nothing squeezes on the way down.
FLIP_VIA_POSE = {
    "follower_left_joint_0": -0.0002,
    "follower_left_joint_1": 1.5908,
    "follower_left_joint_2": 1.0548,
    "follower_left_joint_3": -0.4799,
    "follower_left_joint_4": -0.0095,
    "follower_left_joint_5": -0.1384,
    "follower_left_left_carriage_joint": 0.0120,
    "follower_left_right_carriage_joint": 0.0120,
}

FLIP_GRASP_BANK_POSE = {
    "follower_left_joint_0": -0.0001,
    "follower_left_joint_1": 1.7416,
    "follower_left_joint_2": 0.7745,
    "follower_left_joint_3": 0.0754,
    "follower_left_joint_4": 0.0000,
    "follower_left_joint_5": -0.0001,
    "follower_left_left_carriage_joint": 0.0060,
    "follower_left_right_carriage_joint": 0.0060,
}

# Speed gate [m/s]: the mug pivots and briefly swings during an honest
# handle flip, so the cap sits at the slide's push cap — a fling still
# crosses it immediately.
FLIP_SPEED_MAX = 0.75

# Inverted rest height of the mug ROOT. The mug root origin lies ON its
# bottom plane (collision zmin = 0 in the root frame), so the upright rest
# height IS the table-top height; flipped onto its rim, the root sits one
# full mug height above the table top, plus 1 mm of settle allowance — the
# tape-measure protocol places the mug, it does not press it.
OBJECT_REST_Z_INVERTED = OBJECT_REST_Z + MUG_RIM_HEIGHT + 0.001

# rot is (x, y, z, w): 180 degrees about (1, 1, 0)/sqrt(2) == roll pi (mug
# upside down) THEN yaw +90 — the mug's +X handle ends along env +Y, toward
# the Trossen's base plate, matching the upright tasks' handle-toward-arm
# spawn convention.
_INVERTED_ROT = (0.70710678, 0.70710678, 0.0, 0.0)


@configclass
class FlipSceneCfg(TrossenMugLiftSceneCfg):
    """The shared rig scene with the mug spawned upside down, plus the
    handle/body pad sensor split the flip's rewards read."""

    # Per-pad HANDLE contact: the opposed-handle-pinch gate. The handle
    # pieces are their own collision prims, so the filter is exact.
    pad_left_handle = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_left",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_handle_[0-2]/.*"],
    )
    pad_right_handle = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_right",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_handle_[0-2]/.*"],
    )
    # Pads pressing the mug's BODY (wall/base): in the flip this is the
    # penalized channel — the grasp belongs on the handle. The lift exempts
    # pads from its no-batting rule because body brushes precede its pinch;
    # the flip's pinch is the handle, so body contact is never en route.
    pad_body_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_.*",
        filter_shape_prim_expr=[
            "{ENV_REGEX_NS}/Object/collisions_wall_[0-7]/.*",
            "{ENV_REGEX_NS}/Object/collisions_base/.*",
        ],
    )

    def __post_init__(self):
        super().__post_init__()
        self.object.init_state.pos = [_SPAWN_X, _SPAWN_Y, OBJECT_REST_Z_INVERTED]
        self.object.init_state.rot = list(_INVERTED_ROT)


@configclass
class FlipCommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        # ONE fixed goal: the mug's own spawn spot, upright. The flip happens
        # in place — start and goal are the same tape measurement.
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(_SPAWN_X, _SPAWN_X),
            pos_y=(_SPAWN_Y, _SPAWN_Y),
            pos_z=(OBJECT_REST_Z, OBJECT_REST_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # Online Metrics/success_rate with the evaluator's gates, measured on
        # the OBJECT, orientation as TILT only (a righted mug may rest at any
        # yaw — the handle ends wherever the flip leaves it).
        position_success_threshold=mdp.SUCCESS_POS_THRESHOLD,
        orientation_success_threshold=mdp.SUCCESS_TILT_THRESHOLD,
    )

    def __post_init__(self):
        self.object_pose.class_type = mdp.ObjectPoseSuccessCommand


@configclass
class FlipActionsCfg:
    """The campaign's ONE action space (see the lift ActionsCfg ruling)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[ARM_JOINTS],
        scale={
            "follower_left_joint_[0-2]": 0.5,
            "follower_left_joint_3": 1.0,
            "follower_left_joint_[4-5]": 1.5,
        },
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT, GRIPPER_JOINT_R],
        scale=0.15,
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
    )


@configclass
class FlipObservationsCfg:
    """The slide's observation set plus the mug's orientation — the flip
    policy cannot right an orientation it cannot see."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_world)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))
        # LAST (2026-08-28): the arm action offset in action units, so the policy
        # knows what a zero action holds after a banked start. Appended last so a
        # checkpoint trained without it can be padded (probe_pad_checkpoint).
        action_offset = ObsTerm(func=mdp.arm_action_offset, clip=(-6.0, 6.0))
        # LAST (2026-08-29): the fingertip->handle vector and the two pads'
        # handle contact. Until now the policy saw no end-effector pose and no
        # contact: it could not tell a jaw closed on nothing from a pinch, and
        # committed to the flip 6 cm off the handle (phantom flips from home,
        # from far rung starts, after arrivals; the far-field swings it learned
        # that way then collapsed the lifted-bank start from 100% to 0%).
        # Appended last so a 44-dim checkpoint pads to 49 (probe_pad_checkpoint).
        tip_handle = ObsTerm(
            func=mdp.tip_handle_vector,
            params={"wrist_cfg": SceneEntityCfg("robot", body_names=["follower_left_link_6"])},
        )
        handle_pinch = ObsTerm(func=mdp.handle_pinch_flags)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlipRewardsCfg:
    """The 7-stage FSM economy (2026-08-29 spec): ONE positive-only stage-gated
    progress term + ONE one-shot milestone term + the terminal bonus + small
    interpretable penalties + weak action regularization. No signed PBRS, no
    per-step proximity/contact income, no duplicated event payments
    (flip_by_handle and phantom_flip are replaced by the FSM's grasp gating).
    All metric_* / fsm_* terms are logging-only (1e-3 metric scale)."""

    milestones = RewTerm(func=mdp.fsm_milestones, params={}, weight=1.0)
    progress = RewTerm(func=mdp.fsm_progress, params={}, weight=1.0)
    success_bonus = RewTerm(func=mdp.is_terminated_term, weight=SUCCESS_BONUS, params={"term_keys": ["task_complete"]})
    invalid_regression = RewTerm(func=mdp.fsm_regressed_raw, params={}, weight=-1.0)

    # -------- penalties (small and interpretable; the grasp-validity gate does
    # the real work: body contact yields no credit rather than a big fine)
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-8.0,
        params={"term_keys": ["robot_abnormal", "physics_diverged", "object_out_of_bound"]},
    )
    arm_on_mug_body = RewTerm(
        func=mdp.body_contact, weight=-1.0, params={"sensor_name": "arm_body_contact", "threshold": 1.0}
    )
    pads_on_mug_body = RewTerm(
        func=mdp.body_contact_without_handle, weight=-2.0, params={"sensor_name": "pad_body_contact", "threshold": 1.0}
    )
    table_scrape = RewTerm(
        func=mdp.body_contact, weight=-1.0, params={"sensor_name": "arm_table_contact", "threshold": 1.0}
    )
    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-3e-3)
    action_jerk = RewTerm(func=mdp.arm_action_jerk_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["follower_left_joint_[0-5]"])},
    )

    # -------- logging-only metrics (weight 1.0, values scaled 1e-3; W&B x1000)
    metric_stage = RewTerm(func=mdp.fsm_metric_stage, params={}, weight=1.0)
    metric_regressions = RewTerm(func=mdp.fsm_metric_regressions, params={}, weight=1.0)
    fsm_stage_0_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 0}, weight=1.0)
    fsm_stage_1_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 1}, weight=1.0)
    fsm_stage_2_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 2}, weight=1.0)
    fsm_stage_3_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 3}, weight=1.0)
    fsm_stage_4_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 4}, weight=1.0)
    fsm_stage_5_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 5}, weight=1.0)
    fsm_stage_6_fraction = RewTerm(func=mdp.fsm_stage_fraction, params={"stage": 6}, weight=1.0)
    fsm_transition_0_to_1 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 0}, weight=1.0)
    fsm_transition_1_to_2 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 1}, weight=1.0)
    fsm_transition_2_to_3 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 2}, weight=1.0)
    fsm_transition_3_to_4 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 3}, weight=1.0)
    fsm_transition_4_to_5 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 4}, weight=1.0)
    fsm_transition_5_to_6 = RewTerm(func=mdp.fsm_transition_rate, params={"k": 5}, weight=1.0)
    fsm_valid_handle_grasp_rate = RewTerm(func=mdp.fsm_flag_metric, params={"key": "valid_grasp"}, weight=1.0)
    fsm_drop_after_grasp_rate = RewTerm(func=mdp.fsm_flag_metric, params={"key": "dropped"}, weight=1.0)
    fsm_completion_rate = RewTerm(func=mdp.fsm_flag_metric, params={"key": "success"}, weight=1.0)
    # MUST log 0 forever: completion is impossible without the grasp latch
    fsm_completion_without_valid_handle_grasp = RewTerm(
        func=mdp.fsm_flag_metric, params={"key": "success_no_grasp"}, weight=1.0
    )


@configclass
class FlipTerminationsCfg:
    """The slide's containment set, verbatim."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # POSITIVE termination: flipped by the handle, placed upright, released, arm parked.
    task_complete = DoneTerm(
        func=mdp.flip_fsm,
        params={
            "sensor_name": "pad_object_contact",
            "body_sensor": "pad_body_contact",
            "contact_threshold": 0.5,
            "body_threshold": 1.0,
            "lift_height": 0.05,  # spec 0.05-0.06 above the inverted rest
            "rest_z": OBJECT_REST_Z_INVERTED,
            "rest_z_upright": OBJECT_REST_Z,
            # ASSUMPTIONS to tune: placed z tolerance, calm thresholds, retreat scale
            "placed_z_tol": 0.01,
            "max_speed": 0.05,
            "max_ang_speed": 2.0,
            "reach_std": 0.10,
            "retreat_std": 0.15,
            # the righted mug hangs ~30 cm up (measured); the placement bucket
            # spans the ballistic descent from there
            "descend_range": 0.35,
            "wrist_cfg": SceneEntityCfg("robot", body_names=["follower_left_link_6"]),
        },
    )
    # Truncate un-pinched episodes at 60 steps (2026-08-29): failing far-field
    # episodes ran 240 steps against ~40-80 for a held-bank success and were
    # >80% of every batch, decaying the held skills each time far starts were
    # added (lift bank 100% -> 0%, twice). See mdp.no_pinch_by.
    # no_pinch 60 (2026-08-30): the legacy value. The decay ledger credits this
    # truncation with keeping far-field failure batches from rotting the held
    # skills (fsm30-38); the fsm7m collapse had five confounds and is not
    # evidence against it. Live again now that pinched_ever is published.
    no_pinch = DoneTerm(func=mdp.no_pinch_by, params={"steps": 60}, time_out=True)
    # a latched flip that ends with the mug down and released is over; idling to the
    # 240-step time-out diluted 97% of fsm7b's batches. Unfined truncation: the
    # fsm7e "farming loop" was really the seeded progress paying itself (fixed in
    # flip_fsm_core: first-step best_prog baseline); fining drops (fsm7f) taxed
    # legitimate toss attempts and slowed everything.
    mug_fallen = DoneTerm(func=mdp.mug_fallen, params={"frames": 15, "rest_z_upright": OBJECT_REST_Z}, time_out=True)
    robot_abnormal = DoneTerm(func=mdp.robot_state_abnormal, params={"max_joint_vel": 25.0})
    physics_diverged = DoneTerm(func=mdp.physics_diverged)
    object_out_of_bound = DoneTerm(
        func=mdp.object_off_table,
        params={"x_bound": 0.38, "y_bound": 0.62, "z_bound": 1.0},
    )


@configclass
class FlipEventCfg:
    """Home starts only, fixed inverted mug spawn, zero jitter: the
    tape-measure protocol."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Declared BEFORE the bank event (bedrock checks it by name): the home
    # half's start diversity is part of the recipe. Zero range = tape-measure.
    randomize_arm_start = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
        },
    )
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class FlipCurriculumCfg:
    """No curriculum: the flip trains from home starts throughout."""


@configclass
class TrossenMugFlipEnvCfg(ManagerBasedRLEnvCfg):
    """The flip task, assembled from its own managers on the shared rig scene."""

    scene: FlipSceneCfg = FlipSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: FlipObservationsCfg = FlipObservationsCfg()
    actions: FlipActionsCfg = FlipActionsCfg()
    commands: FlipCommandsCfg = FlipCommandsCfg()
    rewards: FlipRewardsCfg = FlipRewardsCfg()
    terminations: FlipTerminationsCfg = FlipTerminationsCfg()
    events: FlipEventCfg = FlipEventCfg()
    curriculum: FlipCurriculumCfg = FlipCurriculumCfg()

    def validate_config(self):
        """Aim the recording camera at env_0's workspace (see the slide)."""
        import math as _math  # noqa: PLC0415

        n = max(int(self.scene.num_envs), 1)
        num_rows = _math.ceil(n / _math.sqrt(n))
        num_cols = _math.ceil(n / num_rows)
        ox = (num_rows - 1) / 2 * self.scene.env_spacing
        oy = -(num_cols - 1) / 2 * self.scene.env_spacing
        self.sim.default_visualizer_cfg.eye = (ox - 0.02, oy - 1.4, 0.65)
        self.sim.default_visualizer_cfg.lookat = (ox - 0.02, oy + 0.15, 0.02)

    def _set_step(self, subdivisions: int):
        """30 Hz control, ``subdivisions`` uniform physics steps per 1/90 boundary."""
        self.sim.dt = (1.0 / 90.0) / subdivisions
        self.decimation = 3 * subdivisions
        self.sim.render_interval = self.decimation

    def __post_init__(self):
        # Control boundary identical to the slide/lift by construction: the
        # adaptive-vs-fixed comparison holds the control rate across tasks.
        # FIXED arm of the adaptive-vs-fixed comparison: five uniform subdivisions of the
        # family's 1/90 boundary at the same 30 Hz control (dt 1/450, decimation 15). The
        # ADAPTIVE arm keeps the 1/90 boundary (TrossenMugFlipEnvCfg_ADAPTIVE): the ICF error
        # controller's boundary is sim.dt x num_substeps. Measured 2026-08-29: this recipe at
        # 1/450 reaches 100% deterministic true-home success by iteration 500.
        self._set_step(5)
        self.episode_length_s = 8.0  # 5 -> 8 (2026-08-28): pinch-lift-rotate-place-release-retreat, like the hang
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        self.sim.default_visualizer_cfg = NewtonVisualizerCfg(
            headless=True, eye=(-0.02, -0.55, 0.3), lookat=(-0.02, 0.2, 0.1)
        )
        self.sim.physics = TrossenMugLiftPhysicsCfg()

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + BASE_LINK,
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + EE_LINK,
                    name="end_effector",
                    offset=OffsetCfg(pos=list(EE_TCP_OFFSET)),
                ),
            ],
        )
        # BANK STARTS (2026-08-28): half the resets begin at the validated handle
        # pre-grasp hover (FLIP_GRASP_BANK_POSE) and anneal back toward home over
        # 2400 env-steps -- the hang's recipe. No placement DR here, so no Jacobian.
        # NO anneal (end_step 1e9): the pinch window is ~1% of the joint-space
        # home->grasp path, so alpha_min 0.955 at iter 45 already left only ~22%
        # of bank starts pinched (handle_pinch 10 -> 3 per episode, run y92g80kp).
        # The held half stays at alpha=1; the home half carries the approach.
        apply_reverse_curriculum(
            self, bank_pose=FLIP_GRASP_BANK_POSE, bank_fraction=0.5, end_step=1_000_000_000, gripper_offset=-0.05,
            home_offset_pose=FLIP_VIA_POSE,
            # +-0.03 rad on the plain home starts (2026-08-29): the env had no start
            # randomization at all and the exact-home start was a knife edge.
            home_noise=0.03,
        )
        # seed the grasp bank's FSM state (2026-08-30, fsm7o model_600): from home the
        # WALKER delivers the pinch (home episodes end by time_out with pinched_ever
        # true, never no_pinch) -- but the freshly-pinched state was the one held bank
        # never latch-seeded, so it has the same latch-less value gap that idled every
        # other held bank before fsm7i. Same provenance argument: a genuine handle
        # pinch; any righting from it goes through the handle.
        self.events.reset_arm_grasp_bank.params["seed_stage"] = 1
        self.events.reset_arm_grasp_bank.params["seed_latch"] = True
        # SECOND bank (2026-08-28): the LIFTED-HELD state -- arm + mug 11 cm up in
        # the jaws, captured by probe_flip_lifted_state (56% of lifts settle held) --
        # so the rotation is the first discovery from a quarter of the starts.
        # Stacks on the grasp bank (write_home=False): ~25% lifted, ~37% grasp, ~37% home.
        self.events.reset_arm_lift_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_LIFTED_BANK_POSE,
                "bank_fraction": 0.2,
                "seed_stage": 2, "seed_latch": True,  # held seed: righting can only happen through the pinch (fsm7h: latch-less held starts hold still for 240 steps)
                "noise": 0.0,
                "alpha_min": 1.0,
                "gripper_offset": -0.05,  # firm (~45 N): rigid enough for the rotation on the low pinch
                "object_pose": FLIP_LIFTED_MUG_POSE,
                "write_home": False,
                # offset_pose removed (2026-08-29): with the offset OBSERVED (action_offset
                # obs term) per-rung offsets are no longer ambiguous; each rung anchors to
                # its own pose so the hold from a rotated start is a small action there,
                # and the critic carries that state's value over to the self-flips.
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # THIRD bank: the ROTATED-held state (stacks again): ~20% rotated, ~20% lifted, ~30% grasp, ~30% home.
        self.events.reset_arm_rotate_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_ROTATED_BANK_POSE,
                "bank_fraction": 0.1,
                "seed_stage": 2, "seed_latch": True,  # held seed: righting can only happen through the pinch (fsm7h: latch-less held starts hold still for 240 steps)
                "noise": 0.0,
                "alpha_min": 1.0,
                "gripper_offset": -0.05,
                "object_pose": FLIP_ROTATED_MUG_POSE,
                "write_home": False,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # FOURTH bank: the open-jaw GRASP POINT, ANNEALED toward home over 48k env-steps
        # (~2000 iterations) so the approach is learned from progressively farther
        # starts. Stacks last: ~25% open-grasp, ~15% rotated, ~15% lifted, ~22% grasp, ~22% home.
        # NOTE: the anneal counts env steps of the CURRENT run -- it restarts on resume.
        self.events.reset_arm_hover_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_OPEN_GRASP_BANK_POSE,
                "anchor_pose": FLIP_VIA_POSE,  # anneal descends from the via-point, not from home
                "bank_fraction": 0.25,
                "noise": 0.0,
                "alpha_min": 1.0,
                "write_home": False,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Competence-gated (2026-08-29): the clock anneal reached alpha 0.68 while
        # the policy still only pinched from within a few cm of the bar (6% success
        # over alpha 0.68..1 vs 78% at the rung). alpha_min moves 0.01 per 256 rung
        # episodes: down while they succeed > 60%, up when < 30%.
        self.curriculum.grow_hover = CurrTerm(
            func=mdp.anneal_by_competence,
            # 0.6/0.3 -> 0.35/0.15 (2026-08-29): under exploration noise the rung
            # succeeds ~20% in training against 78% deterministically; the gate never opened.
            # 0.35/0.15 -> 0.30/0.12 (2026-08-29): deterministic success from alpha 0.78..1
            # starts is 62% while the stochastic training rate reads 25-29%.
            # 0.30/0.12 -> 0.22/0.10 (2026-08-29): alpha stalls at 0.80 with the stochastic
            # rung success at 0.25-0.28 while the deterministic policy succeeds ~62% from
            # the same starts -- a +-4 mm pinch under exploration noise reads low.
            params={"event_name": "reset_arm_hover_bank", "lower_at": 0.22, "raise_at": 0.10, "step": 0.01, "window": 256},
        )
        self.curriculum.rung_success = CurrTerm(func=mdp.competence_rate, params={"event_name": "reset_arm_hover_bank"})
        # HOME starts (2026-08-29): no fifth rung. The alpha-interpolated
        # home->via rung anchored each start's action offset to the start pose,
        # so every alpha was its own action regime; the policy learned the
        # via-side half (55% at alpha >= 0.68) and 0% below alpha 0.5 while the
        # whole-rung gate still let alpha_min reach 0 (probe_flip_policy_rollout
        # --alpha_max slices, model_14000). Plain home starts instead carry the
        # VIA as their action offset (home_offset_pose on the grasp bank): zero
        # action drives the arm home -> via along the verified collision-free
        # path and the arrival state is the descent rung's own start. Their
        # success is logged as Curriculum/home_success.
        self.curriculum.home_success = CurrTerm(func=mdp.home_success_rate, params={"window": 256})
        # FIFTH bank, re-added with ONE action offset (2026-08-29): starts along
        # the home -> via path, every one anchored to the via (offset_pose), so
        # the rung is a pure state-space curriculum with the same action
        # semantics as the plain home starts (its alpha=0) and the descent rung's
        # via start (its alpha=1). The gate counts only the FRONTIER slice
        # (alpha within 0.15 of alpha_min), not the whole rung, and the rung
        # samples a sliding window [alpha_min, alpha_min + 0.3] (30% of its
        # starts from the full remainder so passed regions keep coverage):
        # over the whole remainder the frontier was 5.6% of the batch and
        # alpha_min sat at 0.47 for 900 iterations (fsm31). 0.3 of envs, taken
        # from the grasp bank (0.5 -> 0.4).
        self.events.reset_arm_home_via_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_VIA_POSE,
                # 0.3 -> 0.25, grasp 0.4 -> 0.5, lift 0.25 -> 0.35 (2026-08-29): with
                # far starts at 49% of envs the held skills decayed even under the
                # 60-step truncation (lift bank 100% -> 29% in 1500 iterations of
                # fsm39). Effective shares now: held 63%, far-field 28%.
                "bank_fraction": 0.25,
                "noise": 0.03,  # a ball around every rung start, not just the home->via line
                "alpha_min": 1.0,
                "alpha_max": 1.0,
                "alpha_tail": 0.3,
                "write_home": False,
                "offset_pose": FLIP_VIA_POSE,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.curriculum.grow_home_via = CurrTerm(
            func=mdp.anneal_by_competence,
            params={"event_name": "reset_arm_home_via_bank", "lower_at": 0.22, "raise_at": 0.10, "step": 0.01, "window": 64, "frontier": 0.15, "span": 0.3},
        )
        self.curriculum.rung_success_home_via = CurrTerm(func=mdp.competence_rate, params={"event_name": "reset_arm_home_via_bank"})
        # Competence-gated reverse curriculum (spec 2026-08-29): phases latch
        # forward on measured completion/grasp rates, never on elapsed steps.
        self.curriculum.stage_gates = CurrTerm(func=mdp.staged_bank_gates, params={"window": 512})
        # FLIP_NO_ANNEAL=1 (2026-08-29): every rung fully open from step 0 -- the
        # whole home->via and via->grasp paths plus the held banks -- and no
        # competence gates. A/B against the gated anneal for convergence speed.
        if os.environ.get("FLIP_NO_ANNEAL", "1") == "1":
            self.events.reset_arm_hover_bank.params["alpha_min"] = 0.0
            self.events.reset_arm_home_via_bank.params["alpha_min"] = 0.0
            self.events.reset_arm_home_via_bank.params["alpha_max"] = 1.0
            self.events.reset_arm_home_via_bank.params["alpha_tail"] = 0.0
            self.curriculum.grow_hover = None
            self.curriculum.grow_home_via = None
        # Banks that WRITE THE MUG (lift, rotate) must run after the arm-only
        # rungs: events stack by selection, and a later arm-only rung inherited
        # the airborne mug of an earlier lift/rotate selection -- 41% of
        # home->via and 40% of descent starts began with the mug falling
        # (probe_bank_overlap, 2026-08-29), so the rung metrics read ~0.6x and
        # the 0.22 gate was effectively ~0.37. Moving them last makes every
        # start self-consistent.
        for name in ("reset_arm_lift_bank", "reset_arm_rotate_bank"):
            term = getattr(self.events, name)
            delattr(self.events, name)
            setattr(self.events, name, term)
        # ROTATION-PATH bank (2026-08-29): held, partially rotated starts sampled
        # from the recorded scripted doorknob (probe_flip_rotpath_dump.py ->
        # rotpath.json: 19 states, up_cos -0.81 .. +0.96, 64/64 held). From
        # scratch the doorknob was the bottleneck: at 500 iterations the policy
        # lifts and lets the mug dangle (j5 never moves) because the 2 rad
        # wrist roll is a 13-sigma mean drift. Stacks LAST (writes the mug).
        import json
        with open(os.path.join(os.path.dirname(__file__), "rotpath.json")) as f:
            rotpath = json.load(f)
        self.events.reset_arm_rotpath_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": rotpath[0]["pose"],
                "trajectory": rotpath,
                "bank_fraction": 0.3,
                "seed_stage": 2, "seed_latch": True,  # held seed (see lift bank note)
                "noise": 0.0,
                "alpha_min": 1.0,
                "gripper_offset": -0.05,
                "write_home": False,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # TOSS bank (2026-08-29): held, mid-air, pre-release states harvested from placed2
        # model_600's own successes (probe_toss_harvest -> tosspath.json). The measured set-down
        # is a timed release ~1 s before touchdown (mug held at up ~0.1, z ~0.2); scripted
        # lowering never lands upright (13 ramps x 4 squeezes: 0-6%). Starting 15% of envs
        # here puts the release decision inside every rollout. Stacks LAST (writes the mug).
        _tosspath = os.path.join(os.path.dirname(__file__), "tosspath.json")
        if os.path.exists(_tosspath):
            with open(_tosspath) as f:
                tosspath = json.load(f)
            if tosspath:
                self.events.reset_arm_toss_bank = EventTerm(
                    func=mdp.reset_arm_reverse_curriculum,
                    mode="reset",
                    params={
                        "pose": tosspath[0]["pose"],
                        "trajectory": tosspath,
                        "bank_fraction": 0.15,
                        "seed_stage": 3,  # pre-release mid-flip; place-bucket progress pays the descent
                        "seed_latch": True,  # provenance: harvested from verified handle-flip successes
                        "noise": 0.0,
                        "alpha_min": 1.0,
                        "gripper_offset": -0.05,
                        "write_home": False,
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                )
        # FLIP_WAYPOINT=1 (2026-08-29): once the arm reaches the via, the action
        # offset advances to the open-jaw grasp pose (zero action = descend along
        # the verified path; the descent rung's starts are anchored there too).
        if os.environ.get("FLIP_WAYPOINT", "1") == "1":
            self.events.approach_waypoint = EventTerm(
                func=mdp.advance_action_offset_waypoint,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=False,
                params={"from_pose": FLIP_VIA_POSE, "to_pose": FLIP_OPEN_GRASP_BANK_POSE, "tol": 0.15, "max_wait": 15},
            )
            self.events.reset_arm_hover_bank.params["offset_pose"] = FLIP_OPEN_GRASP_BANK_POSE
        # FLIP_FAR_PENALTY=1 (2026-08-29): with the waypoint offsets zero action is
        # the approach; bill arm+gripper action beyond 4 cm from the handle so
        # the deterministic mean stays on the nominal path and the jaws stay
        # open until arrival (fast-H regressed 90 -> 66% from home at 500
        # because it learned to close the jaws at step 0).
        if os.environ.get("FLIP_FAR_PENALTY", "1") == "1":
            self.rewards.action_l2_far = RewTerm(
                func=mdp.action_l2_far,
                weight=-0.2,
                params={"far": 0.04, "wrist_cfg": SceneEntityCfg("robot", body_names=["follower_left_link_6"])},
            )
        # FLIP_FAR_PENALTY=1 (2026-08-29): with the waypoint offsets zero action is
        # the approach; bill arm+gripper action beyond 4 cm from the handle so the
        # deterministic mean stays on the nominal path and the jaws stay open
        # until arrival (fast-H: 89% -> 68% from home between 250 and 500 because
        # the mean learned to close the jaws at step 0).
        if os.environ.get("FLIP_FAR_PENALTY", "1") == "1":
            self.rewards.action_l2_far = RewTerm(
                func=mdp.action_l2_far,
                weight=-0.2,
                params={"far": 0.04, "wrist_cfg": SceneEntityCfg("robot", body_names=["follower_left_link_6"])},
            )
        # FLIP_FAR_HEAVY=1 (2026-08-29): the held skills (pinch, lift, doorknob)
        # are learned by iteration 250 with roll sigma 0.4/0.5 even at a small
        # share (fast-D), so the start mix can favour the approach from the
        # start: effective shares home ~30%, home->via 20%, descent 20%,
        # grasp 10%, lift 4%, rotate 4%, rotpath 12%.
        if os.environ.get("FLIP_FAR_HEAVY", "1") == "1":
            # banks select independently and the LAST one owns the env, so a
            # bank's effective share is f x prod(1 - f_later); order: grasp,
            # hover, home_via, lift, rotate, rotpath (probe_bank_overlap verifies).
            self.events.reset_arm_rotpath_bank.params["bank_fraction"] = 0.12   # 12%
            self.events.reset_arm_rotate_bank.params["bank_fraction"] = 0.0455  # 4%
            self.events.reset_arm_lift_bank.params["bank_fraction"] = 0.0476    # 4%
            self.events.reset_arm_home_via_bank.params["bank_fraction"] = 0.25  # 20%
            self.events.reset_arm_hover_bank.params["bank_fraction"] = 0.333    # 20%
            self.events.reset_arm_grasp_bank.params["bank_fraction"] = 0.25     # 10%; plain home 30%


@configclass
class TrossenMugFlipEnvCfg_ADAPTIVE(TrossenMugFlipEnvCfg):
    """The ADAPTIVE arm: one physics step per 1/90 boundary (launch with --solver icf-adaptive)."""

    def __post_init__(self):
        super().__post_init__()
        self._set_step(1)


@configclass
class TrossenMugFlipEnvCfg_PLAY(TrossenMugFlipEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
