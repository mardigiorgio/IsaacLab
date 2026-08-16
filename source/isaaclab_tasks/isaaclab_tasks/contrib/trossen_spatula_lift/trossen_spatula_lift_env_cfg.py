# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI spatula-lift: classic Franka-cube-lift shaping on the official rig.

The thin-object task for the adaptive-vs-fixed solver comparison: the LBM wooden flat
spatula (66 g, 30.0 x 7.0 x 5.1 cm -- the same asset as the G1 spatula task) on the
official Trossen Stationary AI rig, single active LEFT arm.

GRASP GEOMETRY (measured from the official USD, decides the task): the gripper's
carriage travel is 0 -> 0.044 m with a hard lower limit, giving a CLOSED finger gap of
4.83 cm. The ~2.2 cm handle is therefore ungraspable -- the fingers close past it. The
BLADE is 6.98 cm wide, so gripping ACROSS the blade gives 2.15 cm of squeeze:
dimensionally the same pinch the Stationary AI cube task performs on its 5.4 cm cube.
The task is lift-by-the-blade -- finger pads clamping a millimeters-thick wooden plate
resting on a rigid tabletop, which is precisely the stiff thin-object contact regime
where fixed-step integration artifacts are largest.

Reward shaping is the classic Franka cube lift, unchanged in structure and weights:
reach (std 0.1, w 1) / lift (w 15) / goal-track (w 16, std 0.3) / fine track (w 5,
std 0.05) / action-rate and joint-velocity penalties. Observations are the reference
single policy group (proprioception + object position + target + last action).

Robot wiring constants (EE link, TCP offset, joints, gripper commands, goal and reset
bands) are the Stationary AI cube task's measured values, reused verbatim.
"""

from __future__ import annotations

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import STATIONARY_AI_CFG

# ---------------------------------------------------------------------------- constants
# Measured Stationary AI wiring (see the cube task for the derivations).
ARM_JOINTS = "follower_left_joint_[0-5]"
GRIPPER_JOINT = "follower_left_left_carriage_joint"
# The right carriage tracks the left 1:1 in joint coordinates (PhysX-measured); command
# it explicitly so the gripper geometry is identical under every solver.
GRIPPER_JOINT_R = "follower_left_right_carriage_joint"
EE_LINK = "follower_left_link_6"
# link_6 -> finger midpoint along link_6 local x: the true grasp point (the
# ee_gripper_link offset of 0.1561 lands ~7 cm past the fingers).
EE_TCP_OFFSET = (0.087, 0.0, 0.0)
BASE_LINK = "follower_left_base_link"

# The manipulated object: the LBM Inomata white mug (convert_mug.py authors the
# USD from the banked lbm_src glTF; collision is pre-split into near-convex
# pieces so per-prim hulling preserves the rim and the handle opening).
OBJECT_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "usd", "mug_inomata_white.usd")

# ---- Real-world-reproducible placement (see REAL_SETUP.md) -------------------------
# Physical reference: the LEFT arm's base plate center at tabletop level. Its env-frame
# position was measured in sim (geometry survey): (-0.020, 0.4575). The spatula's
# nominal pose is defined RELATIVE to that landmark so the identical setup can be
# reproduced at the rig with a tape measure:
#   blade center on the base plate's centerline (lateral offset 0),
#   33.0 cm forward of the base plate center (toward the opposite arm),
#   lying flat, handle pointing to the operator's left, blade edge facing the arm.
BASE_PLATE_ENV = (-0.020, 0.4575)
OBJECT_FORWARD_M = 0.450
OBJECT_LATERAL_M = 0.0
_SPAWN_X = BASE_PLATE_ENV[0] + OBJECT_LATERAL_M
_SPAWN_Y = BASE_PLATE_ENV[1] - OBJECT_FORWARD_M
# Rig tabletop slab top is z=0.02; the mug's body frame origin is its bottom
# center, so the root rests essentially at the slab top.
OBJECT_REST_Z = 0.021
# Root rest z is ~0.020; 0.08 demands unambiguous lift-off.
LIFT_HEIGHT = 0.08


# ---------------------------------------------------------------------------- physics
def _mjwarp_solver_cfg() -> MJWarpSolverCfg:
    # The upstream core/lift (Franka pick-up) MJWarp stack: every OPTION below
    # matches isaaclab_tasks.core.lift.lift_env_cfg PhysicsCfg.newton_mjwarp.
    # The CAPS are scene-sized, not ported: this dual-arm rig's resting
    # constraint demand alone exceeds the reference scene's njmax (the solver
    # asks for 310-330 rows/world with everything at rest), and grasp-phase
    # demand is far higher. njmax/nconmax carry this task's measured budgets.
    return MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=4096,
        nconmax=400,
        impratio=1.0,
        cone="pyramidal",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        use_mujoco_contacts=False,
        ccd_iterations=35,
    )


def _newton_collision_cfg() -> NewtonCollisionPipelineCfg:
    # Upstream core/lift arena, plus a triangle-pair cap sized for this scene:
    # the pair cap is GLOBAL across worlds and overflow drops mesh contacts
    # silently. Doubled with the move to 4096 envs (pooled demand scales with
    # world count; the 2048-env demand already grazed the previous cap).
    return NewtonCollisionPipelineCfg(rigid_contact_max=8_000_000, max_triangle_pairs=192_000_000)


@configclass
class TrossenSpatulaLiftPhysicsCfg(PresetCfg):
    """Newton (MuJoCo-Warp) is the DEFAULT: the experiment is Newton-fixed
    (``--solver mujoco``, preset ``newton_mjwarp`` == ``default``) vs Newton-adaptive
    (``--solver mujoco-adaptive physics=newton_mjwarp_adaptive``). PhysX remains
    reachable via ``physics=physx`` as a debugging escape hatch only.

    The default IS the fixed tier: a bare ``--solver mujoco`` run must never land on
    the 1-substep boundary, where mj dt 0.01 sinks the resting blade into the
    tabletop (rest-probe measured) and goes non-finite on first grasp (NaN at
    ~iter 38, 8192 envs).
    """

    # The FIXED arm: upstream core/lift NewtonCfg verbatim (2 substeps of the
    # outer step, default shape cfg — no margin or mesh-exclusion authoring).
    default: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    # The ADAPTIVE arm: 1 substep keeps the full outer boundary -- choosing dt
    # inside it is the adaptive solver's job. Guarded against fixed-solver use
    # in :meth:`TrossenSpatulaLiftEnvCfg.validate_config`.
    newton_mjwarp_adaptive: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_patch_count=2**20,
        gpu_total_aggregate_pairs_capacity=2**23,
        gpu_found_lost_aggregate_pairs_capacity=2**26,
    )


# ---------------------------------------------------------------------------- scene
@configclass
class TrossenSpatulaLiftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = STATIONARY_AI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # rot is (x, y, z, w): identity = (0, 0, 0, 1). The wxyz-habit [1, 0, 0, 0]
        # is a 180-degree roll here and buries the handle grip in the tabletop.
        # rot is (x, y, z, w): +90-degree yaw points the mug's +X handle along
        # env +Y, toward the Trossen's base plate.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[_SPAWN_X, _SPAWN_Y, OBJECT_REST_Z], rot=[0.0, 0.0, 0.70710678, 0.70710678]
        ),
        # Plain upstream-style spawn: default materials and shapes, contact model
        # entirely from the shared physics preset. (The rigid_props velocity and
        # depenetration caps and iteration counts are PhysX-only; Newton has no
        # consumer for them.)
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
        ),
    )

    # Finger pads pressing the mug's HANDLE pieces: the grasp-the-handle signal.
    pad_handle_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_carriage_.*",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_handle_[0-2]/.*"],
    )
    # NON-GRIPPER left-arm links pressing the mug's BODY (wall sectors + base):
    # the no-batting rule. The finger pads are deliberately excluded — pads
    # brushing the body is a normal part of a grasp attempt, and penalizing it
    # teaches the policy to avoid the mug entirely; only shoving with the arm
    # itself costs.
    arm_body_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_link_.*",
        filter_shape_prim_expr=[
            "{ENV_REGEX_NS}/Object/collisions_wall_[0-7]/.*",
            "{ENV_REGEX_NS}/Object/collisions_base/.*",
        ],
    )

    # Static stand-in for the rig's tabletop collider, which is DISABLED in the
    # task USD overlay: the tabletop link belongs to the robot articulation and
    # enabled_self_collisions=False (required so the finger-assembly hulls do
    # not jam the gripper) filters every arm-tabletop pair — fingers could sink
    # through the slab and scoop the object from below. As a separate static
    # asset, finger-table and object-table are external pairs for every body.
    # Pose/dims match the overlay's collision_box (slab top at z = 0.02).
    table_guard: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableGuard",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.0055, 0.0, 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.749, 1.2192, 0.02),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    ee_frame: FrameTransformerCfg | None = None  # built in __post_init__ (marker cfg copy)

    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------- mdp cfgs
@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # The cube task's measured reachable band: out in front of the left arm.
            pos_x=(-0.12, 0.12),
            pos_y=(-0.10, 0.05),
            # z raised from (0.08, 0.25): goals barely off the slab let the
            # policy score while hovering at table height — carry targets now
            # sit clearly in the air.
            pos_z=(0.15, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    # clip at 6 sigma of the initial policy: harmless for any sane policy, and it
    # bounds the action-rate penalty against the last_action feedback runaway (see
    # the clipped actions observation term).
    # scale 0.25 (was 0.5): the commanded PD-target jump per step is a hard
    # kinematic speed bound the policy cannot learn around — halved after the
    # slow-mo review showed approach speeds no real rig should attempt.
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[ARM_JOINTS], scale=0.25, use_default_offset=True, clip={".*": (-6.0, 6.0)}
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT, GRIPPER_JOINT_R],
        # BOTH carriages actively driven to the same target -- symmetric close, like
        # the real hardware. This replaces the earlier right-carriage weld (a
        # ghost-world-era workaround for the solver-sensitive passive mimic): with
        # the mimic defused in the task layer, two position-driven carriages need no
        # coupling constraint at all, and the grasp stays centered like the real
        # gripper's.
        open_command_expr={GRIPPER_JOINT: 0.044, GRIPPER_JOINT_R: 0.044},
        close_command_expr={GRIPPER_JOINT: 0.0, GRIPPER_JOINT_R: 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Reference Franka lift observation set: proprio + object + target + action."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # The raw last action feeds back into the policy input; unclipped, the loop
        # goes exponentially unstable once the network's feedback gain crosses 1
        # (measured: |action| 1.3e6 within one rollout, then NaN in the PPO update).
        # The clip bounds the loop; 6 sigma never binds for a healthy policy.
        actions = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """The classic Franka cube-lift reward, verbatim: reach / lift / goal-track /
    fine track / action penalties. minimal_height re-based for this object."""

    # FRANKA-FAITHFUL (Marco, 2026-08-13 ~21:00): the classic terms verbatim
    # and NOTHING else. The anti-throw mechanism is the classic economics
    # itself — no failure terminations, so a batted mug wastes the episode's
    # remaining income budget instead of buying a fresh one. The speed recipe
    # (action scale 0.25 + the -5e-1 penalty ramp) stays: the arm's motion is
    # stable and rig-plausible with it.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": LIFT_HEIGHT}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": LIFT_HEIGHT, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": LIFT_HEIGHT, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """The classic Franka-lift penalty ramp, with a 5x steeper target: the
    classic -1e-1 endpoint converged to slam-speed motion (reward 163 run),
    so the converged optimum needs a steeper smoothness price."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -5e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-1, "num_steps": 10000}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    # Off the table is irrecoverable: end the episode instead of letting the policy
    # farm shaping next to a dead object. Bounds sit just outside the slab footprint.
    object_off_table = DoneTerm(func=mdp.object_off_table, params={"x_bound": 0.40, "y_bound": 0.63, "z_bound": 1.0})
    # MJWarp drops the authored body velocity clamps, so the speed bound lives
    # here (Newton migration guide): 20 m/s / 200 rad/s is far above any fling a
    # 0.7 m arm can impart, so this fires only on diverging contact events —
    # its trigger rate is an experiment metric, shared by every arm.
    object_speeding = DoneTerm(
        func=mdp.object_speed_exceeds,
        params={"max_linear_speed": 20.0, "max_angular_speed": 200.0},
    )
    # Robot-side containment twin of object_speeding: 25 rad/s is far beyond
    # any commanded motion, so this fires only on constraint blowups through
    # the arm (which feed joint state straight into the observations).
    robot_abnormal = DoneTerm(func=mdp.robot_state_abnormal, params={"max_joint_vel": 25.0})
    # Solver-level containment valve: the adaptive solver latches a world
    # ``diverged`` when it refuses to commit a step (NaN state, or a SAP inner
    # solve still failing at the dt floor). The world then holds its last
    # committed FINITE state while its clock skips to the boundary, so the
    # state-space valves above cannot see it; the latch is the only signal.
    # Terminate so the env resets the world instead of feeding frozen,
    # action-independent transitions to the learner until time_out.
    physics_diverged = DoneTerm(func=mdp.physics_diverged)
    # FRANKA-FAITHFUL termination set, deliberately (Marco, 2026-08-13): no
    # knocked/tipped/dropped-after-lift terms. Under the classic economics a
    # failed mug wastes the episode's remaining income budget — that waste IS
    # the anti-batting price, and fumbled grasp attempts get to retry within
    # the episode instead of being amputated by a reset. The wall-time cost
    # of simulating dead states to time_out is the accepted trade. Failure
    # semantics (knocked/tipped) live in the eval judgment via video and the
    # metrics, not in the episode structure.


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # ZERO spawn jitter, deliberately: the real-world protocol is a
            # tape-measured FIXED placement (REAL_SETUP.md), so the experiment is
            # lift-from-THE-spot. Prove pickup first; add randomization only if the
            # rig demands robustness. Also makes exploration far cheaper.
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# ---------------------------------------------------------------------------- env cfg
@configclass
class TrossenSpatulaLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrossenSpatulaLiftSceneCfg = TrossenSpatulaLiftSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def validate_config(self):
        """Reject the fixed MJWarp solver on a 1-substep boundary.

        Also aims the recording camera here rather than in ``__post_init__``:
        the camera frames env_0's workspace in WORLD coordinates, env_0's grid
        origin depends on the final num_envs, and CLI overrides land after
        construction — only this hook sees the real count. Mirrors
        ``cloner.grid_transforms`` for index 0 (x from rows negated, y from
        cols; env_0 is ii=jj=0).
        """
        import math as _math  # noqa: PLC0415

        n = max(int(self.scene.num_envs), 1)
        num_rows = _math.ceil(n / _math.sqrt(n))
        num_cols = _math.ceil(n / num_rows)
        ox = (num_rows - 1) / 2 * self.scene.env_spacing
        oy = -(num_cols - 1) / 2 * self.scene.env_spacing
        self.sim.default_visualizer_cfg.eye = (ox - 0.02, oy - 1.4, 0.65)
        self.sim.default_visualizer_cfg.lookat = (ox - 0.02, oy + 0.15, 0.02)
        self._validate_solver_substeps()

    def _validate_solver_substeps(self):
        """Reject the fixed MJWarp solver on a 1-substep boundary.

        mj dt 0.01 sinks the resting blade into the tabletop and goes non-finite on
        first grasp; only the adaptive solver may own the full 0.01 boundary.
        """
        solver_cfg = getattr(self.sim.physics, "solver_cfg", None)
        if solver_cfg is None:
            return
        num_substeps = getattr(self.sim.physics, "num_substeps", 1)
        if not getattr(solver_cfg, "adaptive", False) and num_substeps < 2:
            raise ValueError(
                "The spatula task requires num_substeps >= 2 (mj dt <= 0.005) under the fixed"
                " MJWarp solver: dt 0.01 sinks the resting blade into the tabletop and goes"
                " non-finite on first grasp. Use the default/newton_mjwarp preset, or pair"
                " physics=newton_mjwarp_adaptive with --solver mujoco-adaptive."
            )

    def __post_init__(self):
        # Upstream core/lift timing, verbatim: 120 Hz outer step, decimation 4
        # -> 30 Hz control; with the preset's 2 substeps the fixed tiers run
        # mj dt ~4.2 ms. 5 s episodes.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # Recorded video camera: FRONT view facing the Trossen (the arm faces
        # -Y; the camera sits beyond the mug on -Y looking back at the arm),
        # framing env_0's workspace instead of the whole grid.
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        self.sim.default_visualizer_cfg = NewtonVisualizerCfg(
            headless=True, eye=(-0.02, -0.55, 0.3), lookat=(-0.02, 0.2, 0.1)
        )
        self.sim.physics = TrossenSpatulaLiftPhysicsCfg()

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
class TrossenSpatulaLiftEnvCfg_PLAY(TrossenSpatulaLiftEnvCfg):
    """Evaluation variant: no observation noise and ZERO spawn jitter, so every episode
    starts from the exact tape-measure pose in REAL_SETUP.md -- paired sim/real trials
    evaluate the same initial condition."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
