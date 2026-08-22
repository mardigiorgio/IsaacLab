# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug LIFT: pick the mug up from its tape-measure
spawn and carry it to a fixed goal 23 cm above the table.

The stiff-contact task for the adaptive-vs-fixed solver comparison: the LBM
Inomata mug (plastic, hull-decomposed collision) on the official Trossen
Stationary AI rig, single active LEFT arm, parallel-jaw rim pinch.

Structure: the validated core-lift reward family (grasp-gated progress
ratchet toward the aerial goal, success bonus, contact terms) with the
operator's cylinder-surface reach kernel and straddle pair gate; touch,
object pose and 5-step history in the observations; reverse-curriculum
starts anchored on the operator's teleop-authored pre-grasp (exact
teleports, grasped subset at the measured clamp seat, randomized arm starts
for the home half).

Robot wiring constants (EE link, TCP offset, joints, gripper commands) are
the Stationary AI rig's measured values; every placement constant is
tape-measure-reproducible per REAL_SETUP.md.
"""

from __future__ import annotations

import math
import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
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
from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import STATIONARY_AI_CFG

# ---------------------------------------------------------------------------- constants
# Measured Stationary AI wiring (see the cube task for the derivations).
ARM_JOINTS = "follower_left_joint_[0-5]"
GRIPPER_JOINT = "follower_left_left_carriage_joint"
# The right carriage tracks the left 1:1 in joint coordinates; command it explicitly
# so the gripper geometry is identical under every solver.
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
# Physical reference: the LEFT arm's base plate center at tabletop level, at env-frame
# (-0.020, 0.4575). The spatula's nominal pose is defined RELATIVE to that landmark so
# the identical setup can be reproduced at the rig with a tape measure:
#   blade center on the base plate's centerline (lateral offset 0),
#   33.0 cm forward of the base plate center (toward the opposite arm),
#   lying flat, handle pointing to the operator's left, blade edge facing the arm.
BASE_PLATE_ENV = (-0.020, 0.4575)
# Table lengthwise midpoint: halfway between the short edges, under the
# table camera — reachable for teleop and a one-measurement placement.
OBJECT_FORWARD_M = 0.4575
OBJECT_LATERAL_M = 0.0
_SPAWN_X = BASE_PLATE_ENV[0] + OBJECT_LATERAL_M
_SPAWN_Y = BASE_PLATE_ENV[1] - OBJECT_FORWARD_M
# Rig tabletop slab top is z=0.02; the mug's body frame origin is its bottom
# center, so the root rests essentially at the slab top.
OBJECT_REST_Z = 0.021
# Root rest z is ~0.020; 0.08 demands unambiguous lift-off.
LIFT_HEIGHT = 0.08
# Airborne income pays only below this mug speed [m/s]: a deliberate carry
# stays well under it, a fling exceeds it several-fold.
CARRY_SPEED_MAX = 0.75

# Pre-grasp reset pose: the operator's teleop-authored rim-pinch straddle —
# the only pose to pass the DYNAMIC ladder (scripted close seats at 3.2 mm
# with 3.6 N and lifts 100%; the takeoff policy trained on it lifts 50/50
# from home starts, all straddle grips). Fingers-down generation was probed
# and is kinematically infeasible at this reach (frame-clear closing sweep
# vs joint-fold limits — see probe_generate_pregrasp.py). Under DR this
# vector is only the NOMINAL: every bank start is re-solved per env by
# BANK_POSE_XY_JACOBIAN for its actual mug placement.
GRASP_BANK_POSE = {
    "follower_left_joint_0": 0.042,
    "follower_left_joint_1": 1.978,
    "follower_left_joint_2": 1.586,
    "follower_left_joint_3": -0.753,
    "follower_left_joint_4": 0.000,
    "follower_left_joint_5": -0.043,
    "follower_left_left_carriage_joint": 0.021,
    "follower_left_right_carriage_joint": 0.021,
}

# Joint response translating the banked pre-grasp with a planar mug shift:
# damped pseudo-inverse of the position Jacobian AT the bank pose, (x, y)
# columns. Derived and verified by scripts/probes/probe_bank_jacobian.py
# against the vendor MuJoCo model (0.1-0.2 mm error over a 1 cm shift);
# re-run that probe whenever GRASP_BANK_POSE changes.
BANK_POSE_XY_JACOBIAN = [
    [+2.486126, +0.104479],
    [+0.162484, -3.866392],
    [+0.230619, -5.487691],
    [-0.068135, +1.621299],
    [+1.026885, +0.043155],
    [+2.264140, +0.095150],
]

# Rim circle of the mug in its body frame, the one-wall pinch target for the
# reach term. Height is the authored MUG_HEIGHT in assets/convert_mug.py
# (mesh-asserted there); the radius is the mug body radius, bounded above by
# that script's handle-classification threshold HANDLE_X_MIN = 0.043.
MUG_RIM_HEIGHT = 0.097
MUG_RIM_RADIUS = 0.040

# Arm configuration hovering just short of the mug on a grasping approach
# (200 ms before first pad contact): the grasp-bank reset pose. A hover, not
# the contact pose itself, so reset noise cannot spawn the pads inside the mug
# and the initial home-ward PD pull retreats along a collision-free path.


# ---------------------------------------------------------------------------- physics
def _mjwarp_solver_cfg() -> MJWarpSolverCfg:
    # The upstream core/lift (Franka pick-up) MJWarp stack: every OPTION below
    # matches isaaclab_tasks.core.lift.lift_env_cfg PhysicsCfg.newton_mjwarp.
    # The CAPS are scene-sized, not ported: this dual-arm rig's resting
    # constraint demand alone exceeds the reference scene's njmax, and
    # grasp-phase demand is higher again, so njmax/nconmax are sized for this
    # scene rather than inherited.
    return MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=4096,
        nconmax=400,
        # Friction-to-normal constraint impedance, and the cone it is enforced
        # over. At impratio 1 the tangential direction is no stiffer than the
        # normal one, so a loaded contact slides before the normal constraint
        # yields -- a gripper cannot hold against gravity. The vendor-curated
        # MuJoCo model of this arm (mujoco_menagerie trossen_wxai, and the ALOHA
        # model built on the same hardware) runs impratio 10 over an elliptic
        # cone; a pyramidal cone would make the friction limit direction-
        # dependent, which the higher impratio would then amplify.
        impratio=10.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        use_mujoco_contacts=False,
        ccd_iterations=35,
        sap_contact_tau_d=6.6e-4,
    )


def _newton_collision_cfg() -> NewtonCollisionPipelineCfg:
    # Upstream core/lift arena, plus a triangle-pair cap sized for this scene:
    # the pair cap is GLOBAL across worlds and overflow drops mesh contacts
    # silently. Doubled with the move to 4096 envs (pooled demand scales with
    # world count; the 2048-env demand already grazed the previous cap).
    return NewtonCollisionPipelineCfg(rigid_contact_max=8_000_000, max_triangle_pairs=192_000_000)


# Contact material from the LBM asset's own drake:proximity_properties, which
# reach neither solver: SAP reads no Drake properties and no mjc: attributes,
# so both arms were running Newton's shape defaults (mu 1.0, ke 2.5e3).
#
# mu 0.2 is authored directly in the SDF, and belongs to the MUG ALONE -- it is
# applied as that asset's own material, not here. Carrying it on the shared
# shape cfg would put ceramic's coefficient on the rubber finger pads and the
# tabletop as well, since every shape without its own material takes this value.
#
# ke follows Drake's own conversion k = A * E / H from the compliant-hydroelastic
# modulus E = 1e8 Pa, evaluated on this asset's collision hulls. Stiffness
# combines in series, so the softer body of a pair sets it; the mug's value is
# authored here because every pair it takes part in is mug-limited.
#
# kd is not free: on MuJoCo-Warp (ke, kd) convert to solref as dampratio =
# (kd/2)*sqrt(1/ke), so holding dampratio at 1.0 across the stiffness change
# requires kd = 2*sqrt(ke). Leaving kd at its default would collapse the MuJoCo
# arm's contact damping and make the engine comparison meaningless.
_CONTACT_KE = 4.6e7
_CONTACT_KD = 2.0 * math.sqrt(_CONTACT_KE)


def _newton_shape_cfg() -> NewtonShapeCfg:
    # The FALLBACK material, for shapes that carry no material of their own.
    #
    # mu stays at Newton's ShapeConfig default of 1.0, which is also what the
    # vendor-curated MuJoCo model of this arm authors on every collider
    # (mujoco_menagerie trossen_wxai, friction="1 5e-3 5e-4"; its torsional and
    # rolling terms already equal Newton's defaults). Bodies that need a
    # different coefficient -- the mug, the tabletop -- declare it on their own
    # spawn, so this value must not be repurposed to carry any one of them.
    #
    # gap is a contact GENERATION threshold here, not MuJoCo's inactive-contact
    # band: Newton's narrow phase emits a pair while its separation is below the
    # summed gap and drops it above. Zero would mean no contact is ever created
    # before interpenetration. Sized as a lookahead band: the adaptive solver's
    # step floor (boundary/8 = 1.39 ms) at a 0.1 m/s closing speed advances
    # 0.14 mm per accepted step, so 3 mm covers ~20 steps of approach; only a
    # body crossing the band faster than ~2 m/s per floored step could
    # generate its first pair already in contact. The candidate triangle-pair
    # count the narrow phase scans grows with this band, and it dominates the
    # adaptive march's cost.
    return NewtonShapeCfg(ke=_CONTACT_KE, kd=_CONTACT_KD, gap=0.003)


@configclass
class TrossenMugLiftPhysicsCfg(PresetCfg):
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
        default_shape_cfg=_newton_shape_cfg(),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        default_shape_cfg=_newton_shape_cfg(),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    # The ADAPTIVE arm: 1 substep keeps the full outer boundary -- choosing dt
    # inside it is the adaptive solver's job. Guarded against fixed-solver use
    # in :meth:`TrossenMugLiftEnvCfg.validate_config`.
    newton_mjwarp_adaptive: NewtonCfg = NewtonCfg(
        solver_cfg=_mjwarp_solver_cfg(),
        collision_cfg=_newton_collision_cfg(),
        default_shape_cfg=_newton_shape_cfg(),
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
class TrossenMugLiftSceneCfg(InteractiveSceneCfg):
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
        # (The rigid_props velocity and depenetration caps and iteration counts
        # are PhysX-only; Newton has no consumer for them.)
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            activate_contact_sensors=True,
            collision_props=[
                sim_utils.schemas.UsdPhysicsCollisionCfg(),
                # "none" = raw triangle meshes; MUG_COLLISION=hull hulls each
                # pre-split near-convex piece instead — the asset's authored
                # decomposition exists precisely so per-prim hulling preserves
                # the rim and the handle opening, and hull-hull narrow phase
                # replaces the per-triangle contact kernel. Any adoption of the
                # hull path must revalidate rest penetration and the grip
                # census, on BOTH solver arms together.
                sim_utils.schemas.UsdPhysicsMeshCollisionCfg(
                    mesh_approximation_name=(
                        "convexHull" if os.environ.get("MUG_COLLISION", "mesh") == "hull" else "none"
                    )
                ),
                # See the rig's collision_props: the contact response time is
                # authored rather than derived from ke/kd, which would place it
                # far below what a step of sim.dt can represent.
                MujocoCollisionCfg(solref=(0.02, 1.0)),
            ],
            # TRI's authored ceramic coefficient, from this asset's own
            # drake:mu_static/mu_dynamic. Scoped to the mug so it governs only
            # the pairs the mug takes part in.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.2,
                dynamic_friction=0.2,
                restitution=0.0,
            ),
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

    # Finger PAD pressing the mug, anywhere on it (handle or body pieces).
    # gripper_left/right are a separate body, fixed-jointed onto the carriage,
    # sitting distal to it (toward the object) -- live-verified: during a
    # settled hold the carriage reads exactly 0 while gripper reads sustained
    # 40-70 N; the carriage only sees the incidental transient as an object
    # falls past it, not an actual held grasp.
    pad_object_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_.*",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_.*/.*"],
    )
    # Per-pad views of the same contacts: upstream's success_reward gates on
    # named per-finger sensors (thumb + fingers); on a parallel jaw the left
    # pad stands in for the thumb and the right pad for the finger set.
    pad_left_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_left",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_.*/.*"],
    )
    pad_right_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_gripper_right",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Object/collisions_.*/.*"],
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
    # Any robot body pressing the tabletop slab: scraping the table is never
    # part of a correct pick or push.
    arm_table_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/follower_left_.*",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/TableGuard.*"],
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
            # The tabletop is wood. Without its own material it takes the shared
            # fallback, which is the rig's coefficient rather than anything
            # chosen for a table surface.
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6, restitution=0.0),
                # This slab spawns from a shape cfg, whose collision fragment
                # takes no mjc: authoring, so its contact response time comes
                # from ke/kd. Newton's own stiffness defaults are the pair that
                # converts to MuJoCo's default solref (0.02 s, ratio 1) -- the
                # same response time the rig and the mug author directly.
                NewtonMaterialCfg(contact_stiffness=2.5e3, contact_damping=100.0),
            ],
        ),
    )

    ee_frame: FrameTransformerCfg | None = None  # built in __post_init__ (marker cfg copy)

    # The ground is VISUAL-ONLY. No dynamic body can touch it inside a live
    # episode -- off-table termination fires at tabletop level, ~0.75 m above
    # the floor, and the caged arm cannot reach below the slab -- while a
    # colliding infinite plane defeats BVH pruning, so every geometry query
    # would test the rig frame's full-resolution mesh against it. Newton's
    # importer honors collisionEnabled=False and imports the grid as a
    # non-colliding visual shape.
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
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
        # Marker visualization on, matching the slide: the carry target is
        # visible in the viewer and training clips.
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # ONE fixed carry target, never resampled to a different point —
            # the hardware protocol is a single tape-measured goal, and the
            # demo policy should train against exactly that.
            pos_x=(0.0, 0.0),
            pos_y=(-0.03, -0.03),
            pos_z=(0.25, 0.25),
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
    # scale 0.1 (was 0.25): a rim pinch dies under commanded jitter that a
    # push shrugs off — at init std ~1 the policy commands +/-scale rad of
    # target jitter per step, and grasped starts measured seated clamps
    # breaking within 1-2 steps at 0.25. The slide pins its own 0.25 (its
    # recipe is frozen with trained baselines).
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[ARM_JOINTS], scale=0.1, use_default_offset=True, clip={".*": (-6.0, 6.0)}
    )
    # POSITION-controlled carriages with the same hold-at-offset machinery as
    # the arm, NOT the binary open/close term: a binary gripper cannot HOLD a
    # clamp under exploration — zero/positive action is a full open, the sign
    # flips every few steps at init, and P(close held for an episode) is
    # 0.5^150. With a default offset, zero action KEEPS the current grip, and
    # the bank event retargets the offset so grasped starts hold their seat.
    # Both carriages actively driven -- symmetric close, like the real
    # hardware (the mimic weld stays defused in the task layer).
    # scale 0.008: sampled jitter must stay inside the clamp's measured seat
    # (~3.2 mm) or a held grip breaks between decisions — at 0.02 x std 0.5
    # the seat still dies in a few steps. Full travel stays reachable in a
    # handful of saturated steps.
    # DISCOVERY CONSTRAINT (the entropy arithmetic): the commanded carriage is
    # offset + scale * a with the offset held at the episode's start pose, so
    # closing is discoverable only if the full 17.5 mm open-to-seat travel lies
    # within ~1 sigma of the policy's initial exploration. At init_std 0.5 that
    # requires scale >= ~0.035; at 0.008 the seat is a 4.4-sigma action, a
    # sustained close is ~1e-53 per window, and the fingers never move off the
    # reset value — no reward term can rescue an action the policy cannot
    # sample. PPO anneals its own std once contact income exists, converting
    # discovered brushes into holds; the wide scale only has to make the first
    # contact reachable.
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT, GRIPPER_JOINT_R],
        scale=0.05,
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Reference Franka lift observation set: proprio + object + target + action."""

        # ABSOLUTE joint angles, deliberately: the grasp-bank reset retargets
        # each env's default_joint_pos at its start pose (so zero action holds
        # it), which makes relative-to-default readings alias — one physical
        # pose would observe differently across envs.
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # TOUCH: net per-pad contact force from the mug, the upstream franka
        # lift's fingertip-force channel. A pinch policy without it must find
        # and hold a millimeter-tolerance clamp blind. Clip bounds transient
        # spikes so the obs scale stays sane for the network.
        contact = ObsTerm(
            func=mdp.pad_contact_forces,
            params={"sensor_name": "pad_object_contact"},
            clip=(-20.0, 20.0),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # Upstream observes the object quaternion; a rim pinch cares about
        # tilt, and the policy cannot infer it from position alone.
        object_orientation = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": True}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # The raw last action feeds back into the policy input; unclipped, the loop
        # goes exponentially unstable once the network's feedback gain crosses 1,
        # which ends in a NaN in the PPO update. The clip bounds the loop; 6 sigma
        # does not bind for a healthy policy.
        actions = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # Upstream's proprio group carries 5 steps of history: contact
            # events and closing motions are multi-step phenomena.
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """The VALIDATED core-lift reward structure (isaaclab_tasks.core.lift
    RewardsCfg plus the franka config's contact term), ported in form and
    weight. Note upstream has NO lift term: height income IS the grasp-gated
    progress ratchet toward the aerial goal, plus success at it — our fixed
    goal already sits 23 cm up, so the same mechanism forces the same lift.

    Adaptations, each forced by hardware or task shape and nothing else:
    - fingers_to_object / the ratchet / good_finger_contact run on this
      rig's pad bodies and pad_object_contact sensor (their per-fingertip
      sensor plumbing does not exist on a parallel jaw); kernel forms and
      weights are theirs.
    - orientation_tracking is omitted and success's rotation factor is
      neutralized (rot_std 1e6): the command is position-only and a mug's
      yaw is free.
    """

    # -0.001, not upstream's -0.01: their action tax is affordable because
    # start diversity does the discovering; from our fixed home a 50-step
    # approach at -0.01 costs ~5x what the reach kernel pays back (hand
    # arithmetic in the session log), making the statue the optimum.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)

    # weight 1.0 (classic cube-lift reach), not the dexsuite 0.05: with no
    # aerial-spawn machinery, reach-driven discovery must be profitable.
    # Weight 3.0: at 1.0 the depth gradient is negligible — a rim-tip pinch
    # and a deep grip differ by ~0.11 on the kernel, dwarfed by the contact
    # income, so grips settle at the shallowest hold (rim plucking). Tripling
    # makes depth worth trading gripper jitter for. The known failure mode of
    # a strong root pull is the both-fingers-inside attractor; the opposed-
    # grasp gate (10% income without a two-sided hold) is the guard, and the
    # early videos are the check.
    fingers_to_object = RewTerm(
        func=mdp.fingers_to_object,
        params={
            "std": 0.4,
            "sensor_name": "pad_object_contact",
            "contact_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", body_names="follower_left_gripper_.*"),
        },
        weight=3.0,
    )

    # Progress pays once per min_improvement of NEW best object-to-goal
    # distance, only while held in an opposed grasp: ground given back cannot
    # be re-earned, a batted flight moves nothing.
    position_tracking = RewTerm(
        func=mdp.position_command_progress,
        weight=5.0,
        params={
            "min_improvement": 0.0025,
            "command_name": "object_pose",
            "sensor_name": "pad_object_contact",
            "contact_threshold": 0.01,
        },
    )

    # Upstream's success semantics (pose match while grasped) in the
    # parallel-jaw form: their function hard-requires per-fingertip sensor
    # shapes this gripper does not have.
    success = RewTerm(
        func=mdp.success_at_goal,
        weight=10,
        params={
            "command_name": "object_pose",
            "pos_std": 0.05,
            "sensor_name": "pad_object_contact",
            "contact_threshold": 0.01,
        },
    )

    good_finger_contact = RewTerm(
        func=mdp.mug_grasped,
        params={"sensor_name": "pad_object_contact", "threshold": 0.01},
        weight=0.75,
    )

    contact_count = RewTerm(
        func=mdp.pad_contact_count,
        params={"sensor_name": "pad_object_contact", "threshold": 0.01},
        weight=0.1,
    )

    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-50, params={"term_keys": ["robot_abnormal"]})


@configclass
class CurriculumCfg:
    """Reverse-curriculum anneal: bank starts begin AT the pre-grasp and the
    start distribution grows back toward home over the first 100 iterations
    (2.4k env-steps at 24 steps/iter), per Florensa reverse curriculum
    generation.

    The horizon is set by two constraints. It must end AFTER close-discovery
    saturates (the wide-scale gripper finds sustained contact inside ~40
    iterations, so the anchor has served its purpose by then), and as soon
    after as possible: while the anneal runs, every band of newly retreated
    starts injects fresh unsolved episodes, and consolidation only begins
    once the start distribution goes stationary at alpha 0 — a long horizon
    spends most of the run paying that moving-target tax."""

    grow_approach = CurrTerm(
        func=mdp.anneal_reverse_curriculum,
        params={"start_step": 0, "end_step": 2_400, "event_name": "reset_arm_grasp_bank"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 25 rad/s is far beyond any commanded motion, so this fires only on
    # constraint blowups through the arm (which feed joint state straight
    # into the observations).
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
            # lift-from-THE-spot, and sim must match it.
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # Scatter every reset's arm joints; declared BEFORE the bank event, which
    # overwrites its own subset with exact poses — so the net effect is wide
    # start-state coverage on the home half only. Mug-independent, hence
    # composes with placement/yaw DR unchanged.
    randomize_arm_start = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.6, 0.6),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
        },
    )
    # Reverse-curriculum starts (Florensa): bank episodes begin at an
    # interpolation between home and the teleop pre-grasp; the curriculum
    # lowers alpha_min from 1 to 0, growing the start distribution back
    # along the approach path as training proceeds.
    reset_arm_grasp_bank = EventTerm(
        func=mdp.reset_arm_reverse_curriculum,
        mode="reset",
        params={
            "pose": GRASP_BANK_POSE,
            "bank_fraction": 0.5,
            # Bank poses are applied EXACTLY: under DR the per-env Jacobian
            # re-solve does the adapting, so jitter here only degrades the
            # seeded states (mm-scale clearances).
            "noise": 0.0,
            "alpha_min": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------------------------------------------------------------------------- env cfg
@configclass
class TrossenMugLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrossenMugLiftSceneCfg = TrossenMugLiftSceneCfg(num_envs=8192, env_spacing=2.5)
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

    def __post_init__(self):
        # Exact 30 Hz control (decimation 3 x dt); dt is the closest integer-
        # decimation solution to a 10 ms physics step at that exact rate
        # (decimation 4 -> 8.33 ms is the other candidate; 11.11 ms is closer
        # to 10 ms). 5 s episodes.
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
class TrossenMugLiftEnvCfg_PLAY(TrossenMugLiftEnvCfg):
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
        # Eval is the rig protocol: every episode approaches from home.
        self.events.reset_arm_grasp_bank.params["bank_fraction"] = 0.0
