# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug SLIDE: push the mug from its tape-measure spawn
to the commanded target WITHOUT tipping it, at the PACE of a MOVING goal.

The commanded goal starts ON the mug's spawn and glides to the final
tape-measured target at ``GOAL_SPEED``: tracking income exists only near the
current goal, so the mug must follow the pace — a controlled, sustained
frictional slide. A fixed goal let the policy pick the push speed, and the
trained solution was an unphysical shove; the pace is also the harder regime
for fixed step, which distorts sustained frictional contact.

A standalone task package, isaaclab-style: every manager is declared in this
file; nothing is inherited from the lift task. The PLATFORM pieces — rig
scene, physics stack, wiring constants — are imported from the mug-lift
package the way isaaclab tasks import robots from ``isaaclab_assets``: they
define the shared Stationary AI setup, not either task's behavior, and the
adaptive-vs-fixed comparison requires both tasks to share them exactly.

The grasp machinery is absent by construction: pushing is discoverable from
plain reach shaping.
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
    _SPAWN_X,
    _SPAWN_Y,
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

from isaaclab_tasks.contrib.trossen_mug_lift.mdp import SUCCESS_POS_THRESHOLD, SUCCESS_TILT_THRESHOLD

from . import mdp

# Push speed cap [m/s]: below the lift task's carry cap — a controlled push
# stays under it, a smack exceeds it immediately. Loose enough that contact
# transients do not zero honest pushes.
PUSH_SPEED_MAX = 0.75

# The tape-measured final target: ON the table at the SIDE edge across from
# the mounted camera, one mug-base-radius before the rail, nudged inward per
# the operator view. Probe-measured mapping (scripts/probes/probe_goal_map.py):
# command frame IS the env frame. The hardware protocol is a single
# tape-measured target.
FINAL_GOAL = (0.285, 0.0)
# Goal glide speed [m/s], Marco-confirmed: 0.305 m of travel in ~1.5 s,
# inside the 6 s episode with reach and settle margin.
GOAL_SPEED = 0.20


@configclass
class SlideCommandsCfg:
    object_pose = mdp.MovingPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(1.0e9, 1.0e9),
        # Marker visualization on, so the target is visible in the viewer and
        # training clips.
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
        # Online Metrics/success_rate against the FINAL target, the shared
        # campaign gates, tilt-only (the mug may spin about z while sliding).
        position_success_threshold=SUCCESS_POS_THRESHOLD,
        orientation_success_threshold=SUCCESS_TILT_THRESHOLD,
    )


@configclass
class SlideActionsCfg:
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

    # The one fine that exists targets a different exploit class: with
    # negative early returns, an unpenalized termination is a paid exit,
    # and the divergence termination in particular taught the policy to
    # crush the mug until the contact solve broke. The gates cannot
    # unpay an exit; only a fine prices it.
    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50,
        params={"term_keys": ["robot_abnormal", "physics_diverged"]},
    )

    # Reach toward the mug root: for a push, low on the wall IS the right
    # approach point.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.5)
    # Pacing income: pays near the CURRENT (gliding) goal, upright (gate at
    # ~53 degrees — a pushed mug rocks well past 30), on the table, at push
    # speed. A parked mug's income vanishes as the goal walks away, so the
    # moving goal itself closes the parked-mug exploit the old fixed-goal
    # recipe needed a progress ratchet for. std 0.08 ~= one goal-second of
    # lag tolerance.
    # TIGHT gaussian on the moving goal, by ruling: the tanh tail at std
    # 0.08 paid a parked mug for half the traverse and parking mid-path
    # maxed reward. exp(-(d/0.04)^2) pays 0.37 at 4 cm, 0.02 at 8 cm,
    # nothing at 12 cm — only tracking collects.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_on_table,
        params={
            "std": 0.04,
            "kernel": "gaussian",
            "min_up_cos": 0.6,
            "max_speed": PUSH_SPEED_MAX,
            "command_name": "object_pose",
        },
        weight=5.0,
    )
    # There is NO final-anchored income, by ruling: any term that pays for
    # being at the endpoint pays MORE for arriving early, which the
    # expected-value arithmetic showed makes whack-it-across the optimum.
    # Tracking the CURRENT goal is the entire objective.
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
        # 6 s episodes: ~3.8 s of goal travel plus reach and settle margin.
        # Identical control boundary to the lift by construction: the
        # adaptive-vs-fixed comparison holds the control rate across tasks.
        self.decimation = 3
        self.episode_length_s = 6.0
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
