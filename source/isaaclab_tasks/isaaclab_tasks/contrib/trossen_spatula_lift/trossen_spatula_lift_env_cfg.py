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
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

SPATULA_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "usd", "thimma_wood_natural_flat_spatula.usd")

# ---- Real-world-reproducible placement (see REAL_SETUP.md) -------------------------
# Physical reference: the LEFT arm's base plate center at tabletop level. Its env-frame
# position was measured in sim (geometry survey): (-0.020, 0.4575). The spatula's
# nominal pose is defined RELATIVE to that landmark so the identical setup can be
# reproduced at the rig with a tape measure:
#   blade center on the base plate's centerline (lateral offset 0),
#   33.0 cm forward of the base plate center (toward the opposite arm),
#   lying flat, handle pointing to the operator's left, blade edge facing the arm.
BASE_PLATE_ENV = (-0.020, 0.4575)
SPATULA_FORWARD_M = 0.330
SPATULA_LATERAL_M = 0.0
_SPAWN_X = BASE_PLATE_ENV[0] + SPATULA_LATERAL_M
_SPAWN_Y = BASE_PLATE_ENV[1] - SPATULA_FORWARD_M
# Rig tabletop slab top is z=0.02; the blade rests just above it.
SPATULA_REST_Z = 0.025
# Root rest z is ~0.023; 0.08 demands unambiguous lift-off while tolerating the
# handle-down tilt of a blade grip.
LIFT_HEIGHT = 0.08

# Constraint-row cap sized for exact-mesh GRASP-phase demand: hulled-era 400 and
# a pre-grasp-sized 600 both overflow once policies engage (logged peak 1292
# rows/world), and MuJoCo silently DROPS overflow rows — the spatula loses its
# support, falls through the table, and the unconstrained state cascades to NaN.
# 1600 covers the logged peak with headroom (G1's hand-dense scene validates
# 1200). Memory: 1600 fits at 4096 worlds (the proven caps-A/B config) but hits
# the 3.11 dense-layout wall at 8192 — size num_envs accordingly.
_NEWTON_NJMAX = 1600
_NEWTON_NCONMAX = 200


# ---------------------------------------------------------------------------- physics
def _mjwarp_solver_cfg() -> MJWarpSolverCfg:
    # Newton CollisionPipeline contacts, injected into MJWarp: the manipulated
    # object must collide as its exact meshes (sim-to-real fidelity of the thin
    # blade), and MuJoCo's internal pipeline convexifies every mesh. Both arms
    # consume the same injected contacts, keeping the contact model out of the
    # fixed-vs-adaptive comparison.
    return MJWarpSolverCfg(
        njmax=_NEWTON_NJMAX,
        nconmax=_NEWTON_NCONMAX,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
        use_mujoco_contacts=False,
    )


def _newton_collision_cfg() -> NewtonCollisionPipelineCfg:
    # rigid_contact_max pinned explicitly: the auto-estimator can size the arena
    # below MuJoCo's demand (nconmax x nworld), which breaks graph capture and
    # corrupts the eager fallback. 2M covers nconmax=200 at 8192 worlds.
    # max_triangle_pairs is a GLOBAL cap across all worlds: raw-mesh narrowphase
    # candidates scale with world count, and overflow silently drops mesh
    # contacts (the spatula falls through the table). Logged demand at 8192
    # worlds is 5.1M pre-grasp against the 1M default; 24M leaves grasp-phase
    # headroom.
    return NewtonCollisionPipelineCfg(
        broad_phase="explicit",
        rigid_contact_max=2_000_000,
        max_triangle_pairs=24_000_000,
    )


# The spatula's colliders stay RAW meshes (no convexification of the manipulated
# object); every other collider keeps the default simplification.
_MESH_EXCLUDE = [r".*/Object/collisions_.*"]


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

    # The FIXED arm: 2 substeps -> mj dt 0.005, the cheapest fixed step that holds
    # both the resting blade and the blade-squeeze contact.
    default: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        # 1 mm shape margin (2 mm per pair): the thin-shell runway from the G1
        # spatula scene — force ramps in before true penetration of the raw meshes.
        default_shape_cfg=NewtonShapeCfg(margin=0.001),
        simplify_meshes_exclude=_MESH_EXCLUDE,
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        # 1 mm shape margin (2 mm per pair): the thin-shell runway from the G1
        # spatula scene — force ramps in before true penetration of the raw meshes.
        default_shape_cfg=NewtonShapeCfg(margin=0.001),
        simplify_meshes_exclude=_MESH_EXCLUDE,
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    # The ADAPTIVE arm: 1 substep keeps the full 0.01 boundary -- choosing dt inside
    # it is the adaptive solver's job. Guarded against fixed-solver use in
    # :meth:`TrossenSpatulaLiftEnvCfg.validate_config`.
    newton_mjwarp_adaptive: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        # 1 mm shape margin (2 mm per pair): the thin-shell runway from the G1
        # spatula scene — force ramps in before true penetration of the raw meshes.
        default_shape_cfg=NewtonShapeCfg(margin=0.001),
        simplify_meshes_exclude=_MESH_EXCLUDE,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[_SPAWN_X, _SPAWN_Y, SPATULA_REST_Z], rot=[0, 0, 0, 1]),
        # Thin-object contact authoring ported from the G1 spatula scene (same
        # spatula asset, same lab-table dimensions): compliant contact on the two
        # raw collision meshes so the force ramp is survivable at the march's
        # step sizes, and a low depenetration cap so contact violations bleed
        # out instead of catapulting the 66 g object.
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=SPATULA_USD_PATH,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=0.5,
                disable_gravity=False,
            ),
            # Sized to the CLAMP BOUNDARY of the slower arm: mjwarp clamps solref
            # timeconst to 2*dt, so any stiffer material is silently softened for
            # the fixed arm (dt 5 ms -> timeconst 10 ms -> ke ~9% of nominal) while
            # the adaptive arm's ~1 ms steps feel it in full — an asymmetric
            # material between the two arms of the experiment. timeconst 10 ms
            # (ke ~11k / kd ~210) is what BOTH arms experience identically.
            compliant_contact_stiffness=11000.0,
            compliant_contact_damping=210.0,
            physics_material_prim_path=["collisions_blade/mesh", "collisions_handle/mesh"],
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
            pos_z=(0.08, 0.25),
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
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[ARM_JOINTS], scale=0.5, use_default_offset=True, clip={".*": (-6.0, 6.0)}
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
    """The classic Franka cube-lift terms, weights unchanged; only minimal_height is
    re-based for the spatula's rest height."""

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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    # Off the table is irrecoverable: end the episode instead of letting the policy
    # farm shaping next to a dead object. Bounds sit just outside the slab footprint.
    object_off_table = DoneTerm(func=mdp.object_off_table, params={"x_bound": 0.40, "y_bound": 0.63})


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
    curriculum = None

    def validate_config(self):
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
        # Reference lift timing: 100 Hz physics, decimation 2 -> 50 Hz control, 5 s episodes.
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
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
