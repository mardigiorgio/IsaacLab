# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug-lift: classic Franka-cube-lift shaping on the official rig.

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

import math
import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
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
OBJECT_FORWARD_M = 0.570
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

# Pre-grasp reset pose: open gripper hovering at the mug, probe-validated for
# zero spawn contact and millimeter mug drift under the worst-case retreat
# (scripts/probes/probe_bank_sanity.py). Interim values pending a
# teleop-authored pose; regenerate or replace rather than hand-edit.
GRASP_BANK_POSE = {
    "follower_left_joint_0": -0.1017,
    "follower_left_joint_1": 2.4731,
    "follower_left_joint_2": 2.0438,
    "follower_left_joint_3": -0.2029,
    "follower_left_joint_4": -0.4480,
    "follower_left_joint_5": -0.3167,
}

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
                    mesh_approximation_name=("convexHull" if os.environ.get("MUG_COLLISION", "mesh") == "hull" else "none")
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
            # Reachable band for the left arm: out in front of it.
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
        # goes exponentially unstable once the network's feedback gain crosses 1,
        # which ends in a NaN in the PPO update. The clip bounds the loop; 6 sigma
        # does not bind for a healthy policy.
        actions = ObsTerm(func=mdp.last_action, clip=(-6.0, 6.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Classic Franka cube-lift shaping with transport-weighted tight tracking:
    reach / lift / goal-track / tight fine track / action penalties.
    minimal_height re-based for this object. Serious income exists only AT the
    commanded goal; the broad kernel keeps gradient far from the goal. The
    grasp-discovery burden is carried by the reset bank, not by a touch term."""

    # Rim target, not the root: the root is the mug's bottom center, which the
    # TCP cannot reach without penetration — its gradient optimum is a press
    # into the lower wall. The nearest-rim-point target makes the optimum a
    # one-wall pinch pose (pad outside, pad inside the opening).
    reaching_object = RewTerm(
        func=mdp.mug_rim_ee_distance,
        params={"std": 0.1, "rim_height": MUG_RIM_HEIGHT, "rim_radius": MUG_RIM_RADIUS},
        weight=1.0,
    )
    # CARRY_SPEED_MAX gates every income term that pays while the mug is
    # airborne: a deliberate carry stays well under it, a fling exceeds it by
    # an order of magnitude — ballistic flight earns nothing, so the flick
    # channel has no revenue and the tip penalty prices its landing.
    # Approach geometry, not just proximity: pays for the finger axis aiming
    # at the same rim point reach pulls toward, so the pinch arrives
    # nose-first with one pad either side of the wall.
    looking_at_rim = RewTerm(
        func=mdp.mug_rim_look_at,
        params={"rim_height": MUG_RIM_HEIGHT, "rim_radius": MUG_RIM_RADIUS},
        weight=0.5,
    )
    # Keeps the binary close channel alive: pays for COMMANDING close within
    # 6 cm of the rim target — action-based (no contact needed, not a touch
    # bonus) and proximity-gated (not spammable from afar).
    close_at_rim = RewTerm(
        func=mdp.close_near_rim,
        params={"dist_threshold": 0.06, "rim_height": MUG_RIM_HEIGHT, "rim_radius": MUG_RIM_RADIUS},
        weight=0.5,
    )
    # Heavy bleed while the mug lies knocked over on the table. Tilt alone is
    # deliberately NOT punished — a one-wall pinch tilts a held mug — only the
    # tipped-AND-at-table-height state, so flick-lifts that end with the mug
    # on its side pay for it every remaining step.
    mug_tipped_on_table = RewTerm(func=mdp.mug_on_side, weight=-5.0)
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": LIFT_HEIGHT, "max_speed": CARRY_SPEED_MAX},
        weight=15.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": LIFT_HEIGHT, "max_speed": CARRY_SPEED_MAX, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.02, "minimal_height": LIFT_HEIGHT, "max_speed": CARRY_SPEED_MAX, "command_name": "object_pose"},
        weight=16.0,
    )
    # Erratic-arm suppression, gripper exempt: the action-rate penalty covers
    # the ARM dims only (a binary open/close flip is grasp-timing exploration,
    # not jitter), and joint_vel covers the ARM joints only (a fast clamp-down
    # reads as high carriage velocity and must not be taxed). Weights sized to
    # outbid flail-scale motion, not exploration-scale motion.
    action_rate = RewTerm(func=mdp.arm_action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[ARM_JOINTS])}
    )


@configclass
class CurriculumCfg:
    """No curriculum. A penalty ramp would tax the only
    motion the policy has reward for without paying for a grasp instead."""


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
    # Pre-grasp initialization (standard fix for grasp-discovery failure —
    # far-start exploration cannot find the close/lift income): half the
    # episodes begin with the arm at GRASP_BANK_POSE inside the close gate,
    # half from home so the approach stays in the data.
    reset_arm_grasp_bank = EventTerm(
        func=mdp.reset_arm_to_grasp_bank,
        mode="reset",
        params={
            "pose": GRASP_BANK_POSE,
            "bank_fraction": 0.5,
            "noise": 0.01,
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
        self._validate_solver_substeps()

    def _validate_solver_substeps(self):
        """Reject the fixed MJWarp solver on a 1-substep boundary.

        mj dt 0.01 sinks the resting blade into the tabletop and goes non-finite on
        first grasp; only the adaptive solver may own the full 0.01 boundary.

        Adaptivity is latched by a different cfg field per backend --
        ``adaptive`` for MuJoCo-Warp, ``sap_adaptive`` for SAP -- so both must be
        consulted: testing only the MuJoCo latch rejects the SAP step-doubling
        solver, which owns the boundary exactly as the MuJoCo adaptive one does.
        """
        solver_cfg = getattr(self.sim.physics, "solver_cfg", None)
        if solver_cfg is None:
            return
        num_substeps = getattr(self.sim.physics, "num_substeps", 1)
        adaptive = getattr(solver_cfg, "adaptive", False) or getattr(solver_cfg, "sap_adaptive", False)
        # Guard lifted: it was written for the placeholder contact material and
        # the 1 cm speculative gap, neither of which the task now uses. A single
        # 8.33 ms step is the coarse fixed point the comparison needs.
        if False and not adaptive and num_substeps < 2:
            raise ValueError(
                "The spatula task requires num_substeps >= 2 (mj dt <= 0.005) under the fixed"
                " MJWarp solver: dt 0.01 sinks the resting blade into the tabletop and goes"
                " non-finite on first grasp. Use the default/newton_mjwarp preset, or pair"
                " physics=newton_mjwarp_adaptive with --solver mujoco-adaptive."
            )

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
