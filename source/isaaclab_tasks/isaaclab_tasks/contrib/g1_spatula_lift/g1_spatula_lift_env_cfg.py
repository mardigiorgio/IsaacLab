# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 spatula pick-up env at the lab table.

A fixed-base G1 stands at the lab table with an LBM wooden spatula on the
tabletop. Task: pick the spatula up BY THE HANDLE with the right TriHand and
hold it at the carry point — the hold income pays continuously to timeout
(no success termination). Touching the blade ends the episode. Resets jitter
the right arm/hand joints and the spatula pose for exploration diversity.
"""

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.hydra import PresetCfg

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT, LAB_TABLE_WIDTH, lab_table_cfgs
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from . import mdp

##
# Constants
##

ACTUATED_JOINT_NAMES = [
    "right_shoulder_.*_joint",
    "right_elbow_joint",
    "right_wrist_.*_joint",
    "right_hand_.*_joint",
]
"""Observed joints: right arm + right TriHand fingers (14 dims). The waist is
FROZEN (held at defaults like the legs): the waist-enabled variant let the
policy drape the torso/elbow onto the tabletop instead of reaching.
Arm AND fingers are actuated in JOINT space via
``RelativeJointPositionActionCfg`` (see :class:`ActionsCfg`) — bounded joint
deltas, not task-space IK. The waist, left arm and legs are inert, held at
their defaults by the ``hold_inert_joints`` event."""

SPATULA_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "usd", "thimma_wood_natural_flat_spatula.usd")
"""Converted LBM spatula (see ``assets/convert_assets.py``): body frame = the
Drake SDF frame (X along the length, blade x < 0.057, handle to x = 0.247),
with SEPARATE handle and blade collision prims — the blade prim is what the
blade-contact termination senses."""

HANDLE_GRASP_OFFSET_B = (0.15, 0.0, 0.027)
"""Reach target on the handle centerline, spatula BODY frame [m] (the SDF's
own ``handle_center_aligned`` frame)."""

HANDLE_SEGMENT_P0_B = (0.07, 0.0, 0.02)
"""Handle centerline segment start, spatula BODY frame [m] (privileged grasp geometry)."""

HANDLE_SEGMENT_P1_B = (0.23, 0.0, 0.041)
"""Handle centerline segment end, spatula BODY frame [m]."""

PALM_GRASP_RADIUS = 0.120
"""Palm link within this distance [m] of the handle segment = handle under the
palm. Calibrated to the TriHand's real geometry (probe-measured): with the
hand seated on the handle in the pronated grip the palm link ORIGIN floors at
~0.111 m — the finger knuckles and pads bottom out first."""

THUMB_PRESS_RADIUS = 0.075
"""Thumb tip link within this distance [m] of the handle segment = thumb pad
on the handle's near face. The thumb_2 link ORIGIN sits at the knuckle and
floors at ~0.064 m with the pad seated on the handle (probe-measured in the
pronated vertical-curl grip)."""

FINGER_CONTACT_RADIUS = 0.025
"""Index/middle tip link within this distance [m] of the handle segment
counts as touching it (their knuckle origins DO thread the handle's sides)."""

FINGERTIP_BODY_NAMES = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
"""TriHand fingertip links for the privileged grasp signal."""

BLADE_INWARD_DIR_W = (-1.0, 0.0, 0.0)
"""Use-ready aim direction (world frame): blade toward the robot's midline,
as the spatula rests at spawn."""

GRASP_MAP_PATH = os.path.join(os.path.dirname(__file__), "grasp_map.pt")
"""Curated reset stages authored by ``assets/author_grasp_map.py`` (missing
file → nominal resets only)."""

GRASP_MAP_STAGE_PROBS = (1.0, 0.0)
"""Reset mixture: index 0 = nominal ready-pose start, index 1 = the caged
stage. Caged starts DISABLED (user call): starting in contact with the 66 g
spatula under exploration noise batters it airborne every episode. The map
stays authored/validated — flip to (0.6, 0.4) to re-enable."""

G1_TOE_REACH = 0.142
"""Forward extent of the G1 toe tips from the pelvis origin [m]."""

TOE_STANDOFF = 0.02
"""Gap between the toe tips and the table-edge plane [m]."""

ROBOT_STAND_POS = (0.0, -(LAB_TABLE_WIDTH / 2 + G1_TOE_REACH + TOE_STANDOFF), 0.85)
"""Fixed-base pelvis position [m] at the long table edge, facing +Y. The real
rig is GANTRY-mounted (user), so the base height is a free parameter: 0.85
gives the shoulder ~30 cm of clearance over the 83 cm tabletop — at the
standing 0.75 the forearm fouls the table edge before the fingers reach the
handle (324 searched poses all stalled on that contact). A standing policy
driving legs + waist comes later; this stage is fixed-base."""

PELVIS_TO_SPATULA_Y = 0.366
"""Pelvis -> spatula-spawn Y offset [m]: the ready-pose hand-over-handle
alignment is authored relative to the pelvis, so the spawn must track
ROBOT_STAND_POS, not the table center. 0.366 puts the spatula ~20 cm inside
the table's near edge (user call: deeper than the original 0.3058 bake)."""

SPATULA_REST_TABLE_OFFSET = 0.0134
"""Settled spatula root height above the tabletop [m] (probe ``--mode settle``,
83 cm table, 1 mm Newton contact margin)."""

SPATULA_SPAWN_POS = (
    0.0958,
    ROBOT_STAND_POS[1] + PELVIS_TO_SPATULA_Y,
    LAB_TABLE_HEIGHT + SPATULA_REST_TABLE_OFFSET,
)
"""Spatula spawn, env frame [m]: flat on the tabletop, handle along +X
threading the pronated vertical-curl hand's thumb/finger aperture. Derived
from LAB_TABLE_HEIGHT and ROBOT_STAND_POS so a table remeasure cannot strand
the spawn inside (or off) the tabletop; only the offsets are probe-baked."""

SPATULA_SPAWN_QUAT = (-0.0002, 0.1011, 0.0025, 0.9949)
"""Spawn orientation (x, y, z, w) — THIS FORK'S CONVENTION, not wxyz: the
probe-baked at-rest attitude (slight pitch; the flat handle hovers a few mm
above the tabletop)."""

REST_SPATULA_Z = SPATULA_SPAWN_POS[2]
"""Spatula root height at rest, env frame [m]. Lifting is measured from here."""

LIFT_SUCCESS_Z = LAB_TABLE_HEIGHT + 0.20
"""Carry-point height [m]: the hold income peaks with the spatula root here
(~19 cm of lift from rest)."""

CARRY_POINT = (SPATULA_SPAWN_POS[0], SPATULA_SPAWN_POS[1], LIFT_SUCCESS_Z)
"""Fixed carry target, env frame [m]: straight up from the spawn. The task
income is a continuous kernel on distance to this point while grasped —
there is no success termination to game (core-lift/dexsuite invariant)."""

CONTACT_GATE_N = 0.1
"""Opposed-contact gate force [N] (dexsuite recipe): thumb AND one opposing
digit pressing the handle above this zeroes-or-enables all task income."""

CONTACT_COUNT_N = 0.01
"""Per-digit touch threshold [N] for the ungated contact-count bootstrap."""

BLADE_FORCE_THRESHOLD = 1.0
"""Hand-on-blade force [N] that ends the episode — feather grazes forgiven."""

DEFAULT_ARM_JOINT_POS = {
    # left arm STRAIGHT DOWN by the side, inert. G1 elbow convention
    # (probe-measured): elbow 0 = forearm horizontal FORWARD, positive folds
    # DOWN — hanging the forearm needs elbow ~+1.45; modest splay clears the
    # thigh
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.28,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.45,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # right arm: hovering over the table, hand over the spatula handle in the
    # PRONATED vertical-curl grip (wrist roll -1.87: knuckles up, fingers
    # hooking down over the handle, thumb opposing on the near face —
    # frame-verified; the +1.87 roll is supinated, fingers curl UP).
    # Probe-tuned as a set around the roll (author_grasp_map.py --view)
    # right arm: the ORIGINAL palm-down claw hover (probe-verified
    # palm->grasp 0.0836): palm facing the tabletop, OPEN fingers pointing
    # down at the handle — the claw approach in the user's TriHand photos.
    # Fingers are OPEN by default (no right-hand joint entries = zeros):
    # the pick is approach -> press digits to the table around the handle ->
    # close the claw horizontally -> lever up
    # raised for the 83 cm table (the 73 cm-table pose put the forearm inside
    # the taller slab at reset): more shoulder lift, straighter elbow, roll
    # tucked in. Wrist pitch is re-tuned WITH the flatter forearm: negative
    # pitch is EXTENSION (hand tips back), and the old -0.90 only read as a
    # claw because the folded elbow pointed the forearm steeply down — on the
    # raised arm it faced the palm at the ceiling (frame-verified). -0.20
    # restores palm straight down over the handle: palm arrow world-z -0.98,
    # palm->grasp 0.074 m, fingers level pointing across the handle
    # right arm: the VERIFIED CLAW CAGE, from assets/straddle_search.py.
    # Wrist roll is the axis that decides this task and every earlier pose had
    # it wrong: the old hover used +0.30 and author_grasp_map hard-codes -1.87,
    # while every pose that actually cages the handle sits near -1.05..-1.20.
    # Measured at this pose (tips vs the handle centerline, local-y):
    #   thumb_2  -0.0362 m   index_1  +0.0573 m   middle_1  +0.0245 m
    # i.e. thumb on one side, index AND middle on the other — the hardware
    # grasp (user-replicated on a pen: digits down on the table straddling the
    # object, then close and rake it in). Tips sit 1.2-3.1 cm above the
    # tabletop, so the hand starts around the handle without touching it.
    "right_shoulder_pitch_joint": -0.638,
    "right_shoulder_roll_joint": -0.145,
    "right_shoulder_yaw_joint": -0.099,
    "right_elbow_joint": 1.103,
    "right_wrist_roll_joint": -1.200,
    "right_wrist_pitch_joint": 0.500,
    "right_wrist_yaw_joint": -0.200,
    # thumb ABDUCTED into opposition and its two flexion joints open, so the
    # policy's job is to close them. Index/middle stay at 0 (open) — they have
    # no abduction DOF, so their curl is the whole closing motion on that side
    # thumb_1/thumb_2 sit ON their lower limits in the searched pose
    # ([-1.047, 0.724] and [-1.745, 0.0]); nudged just inside, because a
    # default exactly at (or a rounding step past) a bound fails validation
    "right_hand_thumb_0_joint": 0.628,
    "right_hand_thumb_1_joint": -1.040,
    "right_hand_thumb_2_joint": -1.740,
}
"""Default pose [rad]: left arm down by the side, right hand in the verified
claw cage around the handle (``assets/straddle_search.py``)."""

_TABLE = lab_table_cfgs("{ENV_REGEX_NS}/LabTable")


##
# Scene definition
##


@configclass
class G1SpatulaLiftSceneCfg(InteractiveSceneCfg):
    """Ground + lab table + fixed-base G1 + LBM spatula + blade contact sensor."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    table_top = _TABLE["table_top"]
    table_leg_0 = _TABLE["table_leg_0"]
    table_leg_1 = _TABLE["table_leg_1"]
    table_leg_2 = _TABLE["table_leg_2"]
    table_leg_3 = _TABLE["table_leg_3"]

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    spatula = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Spatula",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=SPATULA_USD_PATH,
            activate_contact_sensors=True,
            # PhysX-preset knobs ONLY: the Newton importer never reads
            # physxRigidBody attributes (its levers are the compliant material
            # below and the preset's MJWarpSolverCfg/NewtonShapeCfg)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=0.5,
            ),
            # CONTACT STIFFNESS: research-validated pinch regime — ManipTrans
            # grasp tuning uses solref (0.003 s, 1.0), i.e. ke ~1.1e5 /
            # kd ~667 through this fork's ke/kd -> solref conversion. History:
            # stage-1 softness (625/50, 0.04 s) let PPO discover the pick via
            # a penetration weld (~300 N pinch on a 0.65 N object, autopsy-
            # measured, run jvf7cph4); 2500/100 (0.02 s) was stage 2. The
            # scripted-pick existence proof is the gate for this value: a
            # scripted close-and-lift must hold with <= 10x object weight and
            # release the spatula below 0.5 m/s
            compliant_contact_stiffness=111000.0,
            compliant_contact_damping=667.0,
            physics_material_prim_path=["collisions_blade/mesh", "collisions_handle/mesh"],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SPATULA_SPAWN_POS, rot=SPATULA_SPAWN_QUAT),
    )

    # any right-hand link pressing the BLADE collision prim: the no-blade rule
    hand_blade_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/.*",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Spatula/collisions_blade/.*"],
    )

    # fingertip pads pressing the HANDLE prim: the force-closure grasp signal
    # (ManiSkill G1 pick recipe — grasp = thumb force AND opposing finger force)
    thumb_handle_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_thumb_[12]_link",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Spatula/collisions_handle/.*"],
    )
    index_handle_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_index_.*_link",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Spatula/collisions_handle/.*"],
    )
    middle_handle_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_middle_.*_link",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Spatula/collisions_handle/.*"],
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    ManiSkill G1 pick recipe (``pd_joint_delta_pos``): tightly BOUNDED joint
    deltas — |arm| <= 0.2 rad, |finger| <= 0.5 rad per step — on the right arm
    + hand only (waist frozen). Tight bounds + firm PD are what keep the
    proven implementation's exploration precise instead of flailing."""

    upper_body = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_.*_joint",
            "right_elbow_joint",
            "right_wrist_.*_joint",
            "right_hand_.*_joint",
        ],
        scale={
            "right_shoulder_.*_joint": 0.2,
            "right_elbow_joint": 0.2,
            "right_wrist_.*_joint": 0.2,
            "right_hand_.*_joint": 0.8,
        },
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Teacher-student layout: ``policy`` is the TEACHER set — deployable
    proprioception/object state PLUS privileged sim-only signals (contact
    forces, cage geometry, object dynamics, task phase). ``student`` is the
    deployable subset the eventual distilled policy is allowed to see; it is
    computed every step but unused by PPO, so distillation can read both
    groups from the same rollouts.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """TEACHER observations: deployable set + privileged sim-only state."""

        # --- deployable (mirrored in StudentCfg) ---
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        spatula_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        spatula_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        palm_to_handle = ObsTerm(
            func=mdp.palm_to_handle_vector,
            params={
                "grasp_offset_b": HANDLE_GRASP_OFFSET_B,
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        # --- privileged (teacher-only, stripped at distillation) ---
        # the exact forces the grasp reward and lift/success gates threshold on
        digit_forces = ObsTerm(func=mdp.digit_handle_forces)
        blade_force = ObsTerm(func=mdp.blade_contact_force)
        # straddle state: per-tip distance to the handle line + which face
        fingertip_cage = ObsTerm(
            func=mdp.fingertip_cage_geometry,
            params={
                "handle_p0_b": HANDLE_SEGMENT_P0_B,
                "handle_p1_b": HANDLE_SEGMENT_P1_B,
                "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES),
            },
        )
        spatula_velocity = ObsTerm(func=mdp.object_velocity_in_robot_root_frame)
        handle_frame = ObsTerm(
            func=mdp.handle_frame_in_robot_root_frame,
            params={
                "grasp_offset_b": HANDLE_GRASP_OFFSET_B,
                "handle_p0_b": HANDLE_SEGMENT_P0_B,
                "handle_p1_b": HANDLE_SEGMENT_P1_B,
            },
        )
        palm_aim = ObsTerm(
            func=mdp.palm_aim_obs,
            params={
                "grasp_offset_b": HANDLE_GRASP_OFFSET_B,
                "palm_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            },
        )
        lift_progress = ObsTerm(
            func=mdp.lift_progress_obs,
            params={"rest_height": REST_SPATULA_Z, "target_height": LIFT_SUCCESS_Z},
        )
        time_left = ObsTerm(func=mdp.time_remaining)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentCfg(ObsGroup):
        """Deployable observations for the distilled student (unused by PPO).

        Proprioception + object pose (a tracker/estimator provides it on the
        real system) + last action. Corruption stays off here — noise models
        belong to the distillation run, not the teacher rollouts.
        """

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        spatula_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        spatula_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        palm_to_handle = ObsTerm(
            func=mdp.palm_to_handle_vector,
            params={
                "grasp_offset_b": HANDLE_GRASP_OFFSET_B,
                "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    student: StudentCfg = StudentCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # non-actioned joints (left arm/hand, legs, feet, full waist) get
    # their PD TARGETS written to the defaults — in this fork untargeted
    # joints otherwise drive to the ZERO pose, not init_state.joint_pos
    hold_inert_joints = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",
                    "left_shoulder_.*_joint",
                    "left_elbow_joint",
                    "left_wrist_.*_joint",
                    "left_hand_.*_joint",
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                ],
            )
        },
    )
    # reset diversity (the exemplar invariant we were missing): with a single
    # deterministic start, PPO converges onto the first local optimum of the
    # pre-contact gradient and nothing ever perturbs it out. Franka lift
    # jitters the cube +-10/25 cm; dexsuite randomizes all joints +-0.5 rad
    randomize_right_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # +-0.06 rad, scaled DOWN for the claw-cage ready pose. The old
            # +-0.15 was sized for a hover 7 cm clear of the handle; from the
            # cage the fingertips sit 1.2-3.3 cm off the object, and 0.15 rad
            # at the elbow is ~4.5 cm of fingertip travel, so the jitter drove
            # the digits into the handle and displaced the spatula >5 cm at
            # reset (caught by test_spatula_settles_on_tabletop). Diversity
            # still matters — if this proves too narrow, the better answer is a
            # reset MIXTURE (hover start + cage start) rather than more jitter
            "position_range": (-0.06, 0.06),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_.*_joint",
                    "right_elbow_joint",
                    "right_wrist_.*_joint",
                    "right_hand_.*_joint",
                ],
            ),
        },
    )
    randomize_spatula_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("spatula"),
        },
    )
    # caged starts: teleport a sampled share of resets into the authored
    # fingers-around-the-handle pose. Declared LAST so the teleport survives
    # the default resets above (event terms run in declaration order); no-op
    # until grasp_map.pt is authored
    reset_grasp_map = EventTerm(
        func=mdp.reset_from_grasp_map,
        mode="reset",
        params={"map_path": GRASP_MAP_PATH, "stage_probs": GRASP_MAP_STAGE_PROBS},
    )


@configclass
class RewardsCfg:
    """The published lift recipe, UNGATED (claw-grasp rebuild).

    Every prior version gated reach/lift/track on ``_opposed_handle_contact``
    — thumb force AND an opposing digit's force, which is a PINCH signature.
    The intended grip is a claw: fingertips go down to the tabletop on either
    side of the handle and rake it into the hand, with the table as the
    opposing surface. That grip may never produce the pinch signature, so
    gating on it made every task term unpayable. Fingertip-on-table contact is
    explicitly legal here: it is the grasp mechanism. Handle preference is kept
    by pointing reach and track at the handle grasp point, not by contact
    plumbing.
    """

    # breadcrumb: worst of palm + tips to the grasp point, std 0.4 so there is
    # gradient across the whole workspace and no absorbing far pose
    reach = RewTerm(
        func=mdp.fingers_to_handle,
        params={
            "std": 0.4,
            "grasp_offset_b": HANDLE_GRASP_OFFSET_B,
            "bodies_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link", *FINGERTIP_BODY_NAMES]),
        },
        weight=1.0,
    )
    # pure height indicator: 5/60 = 0.083/step for every step off the table
    lift = RewTerm(
        func=mdp.object_lifted,
        params={"rest_height": REST_SPATULA_Z, "minimal_offset": 0.03},
        weight=5.0,
    )
    # THE task income: coarse + fine kernels on distance to the carry point,
    # paid every step to timeout. There is no success termination to game
    track = RewTerm(
        func=mdp.track_carry_point,
        params={
            "carry_point": CARRY_POINT,
            "stds": (0.30, 0.05),
            "weights": (0.5, 0.5),
            "rest_height": REST_SPATULA_Z,
            "minimal_offset": 0.03,
        },
        weight=10.0,
    )
    # start at franka-scale ~zero: pre-contact these penalties are the ONLY
    # gradient finger dimensions ever sample (zero income, guaranteed tax),
    # which measurably drove finger actions and noise to zero in run
    # 2026-08-01_18-03-39. The curriculum brings real penalties in after
    # competence — never author fear from step 0
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.0001)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2_clamped,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        weight=-0.0001,
    )
    # -6/60 = -0.1 once: the no-blade rule's real cost is losing the income
    # stream; the explicit penalty is a tiebreaker, not a wall
    blade_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "blade_contact"}, weight=-6.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # NO success termination (core-lift/dexsuite invariant): success is the
    # hold_at_point flow paying to timeout, so there is no terminal-bonus
    # economics to game and no almost-done attractor

    # the no-blade rule: pressing the blade ends the episode
    blade_contact = DoneTerm(
        func=mdp.blade_contact,
        params={"threshold": BLADE_FORCE_THRESHOLD},
    )

    # spatula off the table or batted out of reach
    spatula_dropped = DoneTerm(
        func=mdp.object_out_of_bound,
        params={
            "in_bound_range": {
                "x": (-0.7, 0.7),
                # just past the table's long edges: batted off the near/far
                # side = dropped (the z floor catches actual falls)
                "y": (-(LAB_TABLE_WIDTH / 2 + 0.045), LAB_TABLE_WIDTH / 2 + 0.045),
                "z": (LAB_TABLE_HEIGHT - 0.09, 2.0),
            }
        },
    )

    # abnormal-state guard (dexsuite convention): any watched joint above 2x
    # its velocity limit or a non-finite state; fingers excluded (their tiny
    # links spike legitimately on contact)
    robot_exploded = DoneTerm(
        func=mdp.robot_or_object_state_invalid,
        params={
            "vel_limit_factor": 2.0,
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
            ),
        },
    )


@configclass
class CurriculumCfg:
    """Penalty ramp AFTER competence (the core-lift pattern: regularization is
    curricula'd, task rewards never are).

    ``modify_reward_weight`` is a one-shot STEP, not a ramp, and ``num_steps``
    counts ``common_step_counter`` — control steps, not iterations. At
    ``num_steps_per_env=24`` the old 15000 landed at iteration 625, which is
    42% of the new 1500-iteration budget. 6000 puts it at iteration 250, right
    at the first decision point.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": 6000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_l2", "weight": -0.02, "num_steps": 6000},
    )


##
# Physics presets
##


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets."""

    default = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
        friction_correlation_distance=0.00625,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=1200,
            nconmax=512,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            # 50/10: measured throughput lever; dexsuite's 100/15 is conservative
            iterations=50,
            ls_iterations=10,
            ls_parallel=False,
            use_mujoco_contacts=False,
            # fixed-step: the adaptive solver is an ~11x tax on this scene
            # (855 vs 9700 steps/s at 1024 envs, measured)
            adaptive=False,
            ccd_iterations=35,
            sap_solver_iterations=64,
        ),
        # sized for 4096+ envs: rigid_contact_max >= nconmax x nworlds
        collision_cfg=NewtonCollisionPipelineCfg(
            rigid_contact_max=8_000_000,
            max_triangle_pairs=16_000_000,
        ),
        # the spatula keeps its RAW handle/blade triangle colliders (the
        # asset's own low-res collision surface); everything else hulls
        simplify_meshes=True,
        simplify_meshes_exclude=[".*/Spatula/collisions.*"],
        # 1 mm contact margin (thin-shell runway): force ramps in before true
        # penetration of the 7 mm-thin raw spatula meshes; margins sum per
        # pair, so resting contacts stand off ~2 mm
        default_shape_cfg=NewtonShapeCfg(margin=0.001),
        # 1 substep: +54% throughput measured at 4096 envs (10.9k -> 16.9k
        # env-steps/s). The 2-substep setting predated the compliant spatula
        # contacts (solref 0.04 s >> the 1/240 step), which are what actually
        # tamed the finger/table impact pops
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


##
# Environment configuration
##


@configclass
class G1SpatulaLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 spatula pick-up environment."""

    viewer: ViewerCfg = ViewerCfg(eye=(1.4, -1.0, 1.3), lookat=(0.25, -0.15, 0.85), origin_type="env")
    scene: G1SpatulaLiftSceneCfg = G1SpatulaLiftSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 120 Hz physics, decimation 2 -> 60 Hz control; picking something up
        # is a two-second task
        # 240 Hz physics (decimation 4 keeps 60 Hz control): the light thin
        # spatula pops airborne from finger/table impacts at 120 Hz —
        # halving the step stabilizes the contact dynamics the lever relies on
        self.decimation = 4
        # 5 s (300 steps at 60 Hz): the hold income needs runway — the value
        # of a grasp is the flow it pays until timeout (proven lifts run 5-6 s)
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        # fixed base: the G1 stands bolted at the table (the established
        # convention for this robot's manipulation tasks; the real robot's
        # factory controller owns balance)
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.pos = ROBOT_STAND_POS
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **DEFAULT_ARM_JOINT_POS,
        }
        # pin the leg pose with stiff implicit PD (legs are static scenery
        # with the root fixed; the asset's DC-motor gains would sag).
        # waist is FROZEN (held at defaults by hold_inert_joints) with firm
        # PD: policy-actuated waist converged to torso-on-table draping
        self.scene.robot.actuators["waist"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=400.0,
            damping=40.0,
            armature=0.03,
            effort_limit_sim=60.0,
        )
        self.scene.robot.actuators["legs"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            stiffness=400.0,
            damping=40.0,
            armature=0.03,
        )
        self.scene.robot.actuators["feet"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=100.0,
            damping=10.0,
            armature=0.03,
        )
        # hardware-plausible effort bounds (the asset's defaults are far higher)
        self.scene.robot.actuators["arms"].effort_limit = None
        self.scene.robot.actuators["arms"].effort_limit_sim = 60.0
        self.scene.robot.actuators["arms"].damping = 30.0
        self.scene.robot.actuators["hands"].effort_limit = None
        self.scene.robot.actuators["hands"].effort_limit_sim = 5.0
        # firm finger tracking (ManiSkill pick recipe uses stiff PD): the
        # stock 20/2 gains ratchet-sag under gravity and barely press
        self.scene.robot.actuators["hands"].stiffness = 100.0
        self.scene.robot.actuators["hands"].damping = 10.0
