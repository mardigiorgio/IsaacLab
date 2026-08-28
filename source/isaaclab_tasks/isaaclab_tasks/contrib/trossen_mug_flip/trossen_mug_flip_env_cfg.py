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

from . import mdp

# Speed gate [m/s]: the mug pivots and briefly swings during an honest
# handle flip, so the cap sits at the slide's push cap — a fling still
# crosses it immediately.
FLIP_SPEED_MAX = 0.75

# Inverted rest height of the mug ROOT: upright the root rides OBJECT_REST_Z
# above the table with the rim MUG_RIM_HEIGHT above the base plane, so
# resting on the rim puts the root at (rim height - upright rest) + 1 mm of
# settle allowance. The spawn drops that millimeter on the first frames —
# the tape-measure protocol places the mug, it does not press it.
# slab top (0.020) + rim height: inverted, the ROOT (bottom center) is at the
# top, a full mug height above the rim that touches the slab. The earlier
# "rim - upright rest" (0.077) spawned the mug 4 cm INSIDE the slab; hull
# contact tolerated it, the raw collision mesh ejects it through the table
# (measured 2026-08-28: settled z -17 m in 83% of envs). TRI's own
# mug_top_center__z_down frame confirms 0.097 above the resting plane.
OBJECT_REST_Z_INVERTED = 0.020 + MUG_RIM_HEIGHT + 0.001

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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlipRewardsCfg:
    """The flip's reward set, on the slide's expected-value design: gates
    unpay wrong styles (shoved, flung, airborne mugs earn nothing), fines
    price the two behaviors that are never en route to a handle flip."""

    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50,
        params={"term_keys": ["robot_abnormal", "physics_diverged"]},
    )

    # Reach toward the mug: for an inverted mug the handle is the near side.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.5)
    # Flip ratchet: pays per 0.05 of NEW episode-best up-cosine, only while
    # the handle is held in an opposed pinch and the mug is calm — rocking it
    # back and forth or shoving it over with a link earns nothing, ever.
    flip_progress = RewTerm(
        func=mdp.upright_progress,
        params={"min_improvement": 0.05, "max_speed": FLIP_SPEED_MAX},
        weight=5.0,
    )
    # Arrival annuity: STRICTLY upright, at the spot, on the table, calm.
    upright_at_goal = RewTerm(
        func=mdp.upright_at_goal,
        params={"std": 0.05, "max_speed": FLIP_SPEED_MAX, "command_name": "object_pose"},
        weight=16.0,
    )
    # Post-flip retreat: the arrival gates times arm stillness times pads-off
    # — "let go and settle" is itself the paid behavior.
    arm_retreated = RewTerm(
        func=mdp.arm_retreated_after_flip,
        params={"std": 0.05, "vel_std": 4.0, "max_speed": FLIP_SPEED_MAX, "command_name": "object_pose"},
        weight=2.0,
    )

    # The two standing fines: non-handle mug contact (arm links on the body
    # via the lift's no-batting sensor, pads on the body via the flip's own
    # split sensor) and table scraping. Heavy by order: each is priced above
    # the flip ratchet's per-event payment.
    arm_on_mug_body = RewTerm(
        func=mdp.body_contact, weight=-6.0, params={"sensor_name": "arm_body_contact", "threshold": 1.0}
    )
    pads_on_mug_body = RewTerm(
        func=mdp.body_contact, weight=-6.0, params={"sensor_name": "pad_body_contact", "threshold": 1.0}
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
        self.episode_length_s = 7.0
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


@configclass
class TrossenMugFlipEnvCfg_PLAY(TrossenMugFlipEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
