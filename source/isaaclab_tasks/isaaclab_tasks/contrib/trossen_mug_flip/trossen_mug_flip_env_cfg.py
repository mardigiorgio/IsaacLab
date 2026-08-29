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

from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_tasks.contrib.trossen_mug_tree.hang_fsm_core import SUCCESS_BONUS
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlipRewardsCfg:
    """The hang's STAGED economy on the flip (2026-08-28): one-shot milestones
    + rest-neutral shaping (hang_fsm_core, via flip_fsm) + one success bonus,
    plus the family's action taxes, blowup fine and the flip's standing body
    fines. No per-step proximity or contact income anywhere: run wt7f4r33 showed
    a hold annuity makes "pinch and sit on the table" the optimum."""
    milestones = RewTerm(func=mdp.fsm_milestones, params={}, weight=1.0)
    shaping = RewTerm(func=mdp.fsm_shaping, params={}, weight=1.0)
    success_bonus = RewTerm(func=mdp.is_terminated_term, weight=SUCCESS_BONUS, params={"term_keys": ["task_complete"]})
    metric_stage = RewTerm(func=mdp.fsm_metric_stage, params={}, weight=1.0)
    metric_regressions = RewTerm(func=mdp.fsm_metric_regressions, params={}, weight=1.0)
    metric_ms_pinch = RewTerm(func=mdp.fsm_metric_ms_grasp, params={}, weight=1.0)
    metric_ms_lift = RewTerm(func=mdp.fsm_metric_ms_lift, params={}, weight=1.0)
    metric_ms_rotate = RewTerm(func=mdp.fsm_metric_ms_rotate, params={}, weight=1.0)
    metric_ms_place = RewTerm(func=mdp.fsm_metric_ms_release, params={}, weight=1.0)
    metric_ms_retreat = RewTerm(func=mdp.fsm_metric_ms_retreat, params={}, weight=1.0)
    flip_by_handle = RewTerm(func=mdp.flip_by_handle_metric, weight=1.0)

    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50,
        params={"term_keys": ["robot_abnormal", "physics_diverged", "object_out_of_bound"]},
    )

    arm_on_mug_body = RewTerm(
        func=mdp.body_contact, weight=-6.0, params={"sensor_name": "arm_body_contact", "threshold": 1.0}
    )

    pads_on_mug_body = RewTerm(
        # threshold 1 -> 5 N (2026-08-28, measured): a legitimate 14 N handle
        # pinch presses the fingertips into the wall at the handle root at
        # 1.4-2.1 N; a body grab at the vendor gripper gain reads ~44 N.
        # Fined only WITHOUT a handle pinch (2026-08-28): a hanging mug presses the
        # fingertips 5-70 N through the handle root; see body_contact_without_handle.
        func=mdp.body_contact_without_handle, weight=-6.0, params={"sensor_name": "pad_body_contact", "threshold": 1.0}
    )

    table_scrape = RewTerm(
        func=mdp.body_contact, weight=-6.0, params={"sensor_name": "arm_table_contact", "threshold": 1.0}
    )

    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-3e-3)
    action_jerk = RewTerm(func=mdp.arm_action_jerk_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["follower_left_joint_[0-5]"])},
    )

@configclass
class FlipTerminationsCfg:
    """The slide's containment set, verbatim."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # POSITIVE termination: flipped by the handle, placed upright, released, arm parked.
    task_complete = DoneTerm(
        func=mdp.flip_fsm,
        params={
            "pose": ARM_FINISH_POSE,
            "joint_tol": 0.5,
            "sensor_name": "pad_object_contact",
            "lift_height": 0.06,
            # 0.7 -> 0.5 (2026-08-28, probe_flip_low_pinch_chain static sweep): the
            # STATIC held up_cos tops out at 0.66 (forearm roll 1.2, wrist roll -3.0,
            # any wrist pitch); every reading above 0.7 was a transient overshoot, and
            # the one pose peaking at 0.81 drops the mug within 2 s. Stable holds at
            # 0.53-0.66 (100% held) clear 0.5 with margin.
            "rotate_min_cos": 0.5,
            "rest_z": OBJECT_REST_Z_INVERTED,
            # ROTATE ratchet 15 -> 40 (2026-08-28, probe_flip_scripted_rotate): from
            # the lifted bank, j3+j5 rotate the held mug -0.63 -> +0.62 (ROTATED in
            # 36%) and the mug drops right after; a partial rotation paid ~3 against
            # a ~7 shaping loss on the drop, so first steps were net-negative.
            # Max pre-completion 55 + 105 + 20 = 180 < SUCCESS_BONUS 200 (asserted).
            "ratchet_w": (10.0, 10.0, 40.0, 15.0, 30.0),
            # "in_hand": flipped by the handle and held upright for hold_frames (1 s)
            # ends the episode with the success bonus. "placed" (set down upright,
            # released, arm parked) is kept as the alternative; measured 2026-08-28
            # to be kinematically out of reach from a single pinch on this rig.
            "success_mode": "in_hand",
            "hold_frames": 30,
            "wrist_cfg": SceneEntityCfg("robot", body_names=["follower_left_link_6"]),
        },
    )
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

    def __post_init__(self):
        # Control boundary identical to the slide/lift by construction: the
        # adaptive-vs-fixed comparison holds the control rate across tasks.
        # 7 s episodes: approach + handle pinch + flip + release + settle is
        # a longer program than a push.
        self.decimation = 3
        self.episode_length_s = 8.0  # 5 -> 8 (2026-08-28): pinch-lift-rotate-place-release-retreat, like the hang
        self.sim.dt = 1 / 90
        self.sim.render_interval = self.decimation
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
            self, bank_pose=FLIP_GRASP_BANK_POSE, bank_fraction=0.5, end_step=1_000_000_000, gripper_offset=-0.05
        )
        # SECOND bank (2026-08-28): the LIFTED-HELD state -- arm + mug 11 cm up in
        # the jaws, captured by probe_flip_lifted_state (56% of lifts settle held) --
        # so the rotation is the first discovery from a quarter of the starts.
        # Stacks on the grasp bank (write_home=False): ~25% lifted, ~37% grasp, ~37% home.
        self.events.reset_arm_lift_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_LIFTED_BANK_POSE,
                "bank_fraction": 0.25,
                "noise": 0.0,
                "alpha_min": 1.0,
                "gripper_offset": -0.05,  # firm (~45 N): rigid enough for the rotation on the low pinch
                "object_pose": FLIP_LIFTED_MUG_POSE,
                "write_home": False,
                "offset_pose": FLIP_GRASP_BANK_POSE,  # one action semantics across rungs
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # THIRD bank: the ROTATED-held state (stacks again): ~20% rotated, ~20% lifted, ~30% grasp, ~30% home.
        self.events.reset_arm_rotate_bank = EventTerm(
            func=mdp.reset_arm_reverse_curriculum,
            mode="reset",
            params={
                "pose": FLIP_ROTATED_BANK_POSE,
                "bank_fraction": 0.2,
                "noise": 0.0,
                "alpha_min": 1.0,
                "gripper_offset": -0.05,
                "object_pose": FLIP_ROTATED_MUG_POSE,
                "write_home": False,
                "offset_pose": FLIP_GRASP_BANK_POSE,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class TrossenMugFlipEnvCfg_PLAY(TrossenMugFlipEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
