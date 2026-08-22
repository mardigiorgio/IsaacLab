# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug SLIDE: push the mug from its tape-measure spawn
to a commanded position on the table WITHOUT tipping it.

A standalone task package, isaaclab-style: every manager is declared in this
file; nothing is inherited from the lift task, so no lift retune can leak into
the slide's frozen slidev1 recipe. The PLATFORM pieces — rig scene, physics
stack, wiring constants — are imported from the mug-lift package the way
isaaclab tasks import robots from ``isaaclab_assets``: they define the shared
Stationary AI setup, not either task's behavior, and the adaptive-vs-fixed
comparison requires both tasks to share them exactly.

The grasp machinery is absent by construction: pushing is discoverable from
plain reach shaping, and the physics of interest is sustained frictional
sliding — the regime where fixed-step contact distorts normal forces and tips
the mug.
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

from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    ARM_JOINTS,
    BASE_LINK,
    EE_LINK,
    EE_TCP_OFFSET,
    GRIPPER_JOINT,
    GRIPPER_JOINT_R,
    OBJECT_REST_Z,
    TrossenMugLiftPhysicsCfg,
    TrossenMugLiftSceneCfg,
)

from . import mdp

# Push speed cap [m/s]: below the lift task's carry cap — a controlled push
# stays under it, a smack exceeds it immediately. Loose enough that contact
# transients do not zero honest pushes.
PUSH_SPEED_MAX = 0.75


@configclass
class SlideCommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(5.0, 5.0),
        # Marker visualization on, so the target is visible in the viewer and
        # training clips.
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Goal ON the table at the SIDE edge across from the mounted
            # camera, one mug-base-radius before the rail, nudged inward per
            # the operator view. Probe-measured mapping
            # (scripts/probes/probe_goal_map.py): command frame IS the env
            # frame. ONE fixed goal, never resampled elsewhere — the hardware
            # protocol is a single tape-measured target.
            pos_x=(0.285, 0.285),
            pos_y=(0.0, 0.0),
            pos_z=(OBJECT_REST_Z, OBJECT_REST_Z),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SlideActionsCfg:
    """The slide's action set, pinned verbatim to the slidev1 recipe.

    The lift retunes its actions for pinch survival (smaller arm scale,
    position-controlled gripper); the slide's trained ladder used arm scale
    0.25 and the binary gripper, so the frozen recipe is declared here."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[ARM_JOINTS], scale=0.25, use_default_offset=True, clip={".*": (-6.0, 6.0)}
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT, GRIPPER_JOINT_R],
        open_command_expr={GRIPPER_JOINT: 0.044, GRIPPER_JOINT_R: 0.044},
        close_command_expr={GRIPPER_JOINT: 0.0, GRIPPER_JOINT_R: 0.0},
    )


@configclass
class SlideObservationsCfg:
    """The slide's observation set, pinned verbatim to the slidev1 recipe
    (relative-to-default joint angles, as its trained baselines observed)."""

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
    # Post-delivery rest goal. After arrival nothing constrained the ARM: the
    # mug annuity pays regardless of what the arm does, and the action
    # penalties are orders below it, so a finished arm wanders on residual
    # exploration. This pays the arrival annuity's own gates times arm
    # stillness -- income exists only on top of a completed, settled delivery,
    # so it cannot fund hovering short of one, and at weight 2 it stays far
    # below the arrival term (16), so a calmer arm is never worth
    # surrendering the delivery itself.
    arm_settled = RewTerm(
        func=mdp.arm_settled_at_goal,
        weight=2.0,
        params={
            "std": 0.05,
            "vel_std": 0.5,
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
            "command_name": "object_pose",
        },
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
class TrossenMugSlideEnvCfg(ManagerBasedRLEnvCfg):
    """The slide task, assembled from its own managers on the shared rig scene.

    ONE mug position across lift and slide, deliberately: a single
    tape-measure placement serves both hardware protocols."""

    scene: TrossenMugLiftSceneCfg = TrossenMugLiftSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: SlideObservationsCfg = SlideObservationsCfg()
    actions: SlideActionsCfg = SlideActionsCfg()
    commands: SlideCommandsCfg = SlideCommandsCfg()
    rewards: SlideRewardsCfg = SlideRewardsCfg()
    terminations: SlideTerminationsCfg = SlideTerminationsCfg()
    events: SlideEventCfg = SlideEventCfg()
    curriculum: SlideCurriculumCfg = SlideCurriculumCfg()

    def validate_config(self):
        """Aim the recording camera at env_0's workspace.

        The camera frames env_0 in WORLD coordinates, env_0's grid origin
        depends on the final num_envs, and CLI overrides land after
        construction — only this hook sees the real count. Mirrors
        ``cloner.grid_transforms`` for index 0 (ii=jj=0)."""
        import math as _math  # noqa: PLC0415

        n = max(int(self.scene.num_envs), 1)
        num_rows = _math.ceil(n / _math.sqrt(n))
        num_cols = _math.ceil(n / num_rows)
        ox = (num_rows - 1) / 2 * self.scene.env_spacing
        oy = -(num_cols - 1) / 2 * self.scene.env_spacing
        self.sim.default_visualizer_cfg.eye = (ox - 0.02, oy - 1.4, 0.65)
        self.sim.default_visualizer_cfg.lookat = (ox - 0.02, oy + 0.15, 0.02)

    def __post_init__(self):
        # Exact 30 Hz control (decimation 3 x dt); dt is the closest integer-
        # decimation solution to a 10 ms physics step at that exact rate.
        # 5 s episodes. Identical to the lift's boundary by construction: the
        # adaptive-vs-fixed comparison holds the control rate across tasks.
        self.decimation = 3
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 90
        self.sim.render_interval = self.decimation
        # Recorded video camera: FRONT view facing the Trossen (the arm faces
        # -Y; the camera sits beyond the mug on -Y looking back at the arm),
        # framing env_0's workspace instead of the whole grid.
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        self.sim.default_visualizer_cfg = NewtonVisualizerCfg(
            headless=True, eye=(-0.02, -0.55, 0.3), lookat=(-0.02, 0.2, 0.1)
        )
        self.sim.physics = TrossenMugLiftPhysicsCfg()

        # EE frame sensor at the true grasp point (finger midpoint).
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
class TrossenMugSlideEnvCfg_PLAY(TrossenMugSlideEnvCfg):
    """Evaluation variant: no observation noise, fixed tape-measure spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
