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

PALM_GRASP_RADIUS = 0.150
"""Palm link within this distance [m] of the handle segment = handle under the
palm. RECALIBRATED TO THE CLAW (measured at the authored pregrasp reset over
512 envs): the palm link origin sits at 0.101 m. The previous 0.120 was
calibrated for the PRONATED VERTICAL-CURL grip, a different hand pose."""

THUMB_PRESS_RADIUS = 0.100
"""Thumb tip link within this distance [m] of the handle segment = thumb pad
on the handle's far face. RECALIBRATED TO THE CLAW: measured 0.082 m at the
pregrasp reset. The previous 0.075 was below the measured value, so the
predicate could never fire on the claw."""

FINGER_CONTACT_RADIUS = 0.100
"""Index/middle tip link within this distance [m] of the handle segment
counts as caging it. RECALIBRATED TO THE CLAW: measured 0.090 m (index) and
0.086 m (middle) at the pregrasp reset. The previous 0.025 was a leftover from
the pinch recipe and failed by 3.6x, which is why :func:`mdp.grasp_handle`
measured 0.00% firing at every radius tested until this correction.
Verified: at 0.150 / 0.100 / 0.100 the predicate fires on 100% of pregrasp
resets and 0% of hover resets."""

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

ROBOT_STAND_POS = (0.0, -(LAB_TABLE_WIDTH / 2 + G1_TOE_REACH + TOE_STANDOFF), 0.75)
"""Fixed-base pelvis position [m] at the long table edge, facing +Y.

0.75 is the G1's natural standing pelvis height, so the feet rest ON the floor.
This was 0.85 for a gantry-mounted rig, which left the robot visibly floating
10 cm in the air. The original note claimed 0.85 was needed because "at the
standing 0.75 the forearm fouls the table edge before the fingers reach the
handle"; if that reach problem returns, fix it by moving the robot back in Y or
re-authoring the arm pose, not by hanging the robot in mid-air.
"""

PELVIS_TO_SPATULA_Y = 0.366
"""Pelvis -> spatula-spawn Y offset [m]: the ready-pose hand-over-handle
alignment is authored relative to the pelvis, so the spawn must track
ROBOT_STAND_POS, not the table center. 0.366 puts the spatula ~20 cm inside
the table's near edge (user call: deeper than the original 0.3058 bake)."""

SPATULA_REST_TABLE_OFFSET = 0.0134
"""Settled spatula root height above the tabletop [m] (probe ``--mode settle``,
83 cm table, 1 mm Newton contact margin)."""

SPATULA_SPAWN_POS = (0.03975, -0.12014, 0.84354)
"""Spatula spawn, env frame [m]. AUTHORED IN THE GUI (user, pose_lab session
2026-08-05, saved in the ``pregrasp.pt`` stage): the settled pose the pregrasp
was authored against — handle between the digits of
:data:`PREGRASP_JOINT_POS`, thumb on one side, index+middle on the other.
Sits +3 cm along +Y of the pre-session spawn (away from the robot, which
faces +Y). Absolute rather than derived from ROBOT_STAND_POS because it was
placed against the posed hand, so it must move only if that pose does.
Tabletop is LAB_TABLE_HEIGHT (0.83)."""

SPATULA_SPAWN_QUAT = (-0.00024, 0.10116, 0.0025, 0.99487)
"""Spawn orientation (x, y, z, w) — THIS FORK'S CONVENTION, not wxyz: the
at-rest attitude saved with the same pose_lab stage as
:data:`SPATULA_SPAWN_POS` (slight pitch; the flat handle hovers a few mm
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

TABLE_PRESS_FORCE_N = 1.0
"""Per-fingertip tabletop force [N] at which :func:`mdp.fingertip_table_press`
saturates. ~1.5x the spatula's own 0.65 N weight, so a resting graze does not
read as a press. Deliberately LOW against the measured contacts: a scripted
descent onto the rigid (non-compliant) tabletop reads 1.5-375 N per tip, so
saturating at 1 N means a gentle press earns exactly as much as hammering —
the clamp is what stops "press harder" from paying more."""

BLADE_FORCE_THRESHOLD = 1.0
"""Hand-on-blade force [N] that ends the episode — feather grazes forgiven."""

DEFAULT_ARM_JOINT_POS = {
    # left arm straight down by the side, inert. G1 elbow convention: elbow 0
    # is forearm horizontal FORWARD and positive folds DOWN, so hanging the
    # forearm needs a positive elbow; the shoulder roll splays enough to clear
    # the thigh.
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.28,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.45,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # right arm: palm-down hover over the spatula handle, clear of the blade.
    # Wrist pitch is negative = EXTENSION (hand tips back); it must be set
    # against the shoulder/elbow lift so the palm faces the tabletop rather
    # than the ceiling. Fingers are open by default (no right-hand entries =
    # zeros).
    # This is the reach-start half of the reset mixture: the hand hovers clear
    # of the spatula and the policy has to bring it down itself. The other half
    # starts already in PREGRASP_JOINT_POS (see EventCfg.reset_pregrasp).
    "right_shoulder_pitch_joint": -0.60,
    "right_shoulder_roll_joint": -0.15,
    "right_shoulder_yaw_joint": -0.10,
    "right_elbow_joint": 0.70,
    "right_wrist_roll_joint": 0.30,
    "right_wrist_pitch_joint": -0.20,
    "right_wrist_yaw_joint": -0.30,
}
"""Default pose [rad]: left arm down by the side, right hand hovering over the
handle with the fingers open (no right-hand entries = zeros = open)."""

DEFAULT_LEG_JOINT_POS = {
    # Symmetric flat-foot stance sized to the FIXED 0.75 m pelvis: hip and
    # ankle at -knee/2 keeps the sole level and the foot directly under the
    # pelvis. The knee angle must be the straightest stance whose reach still
    # matches the pelvis height — a straighter one penetrates the floor at
    # reset, and the contact buckles the legs into a hips-pitched-back crouch.
    # Same REGEX keys as the asset's own init_state, so the dict merge in
    # __post_init__ REPLACES those entries — explicit joint names alongside
    # the patterns would double-match and fail articulation init.
    ".*_hip_pitch_joint": -0.375,
    ".*_knee_joint": 0.75,
    ".*_ankle_pitch_joint": -0.375,
}
"""Leg stance [rad] for the fixed-base stand: feet flat on the floor under the
pelvis, knees bent forward only as much as the 0.75 m pelvis height demands."""

PREGRASP_JOINT_POS = {
    # Fingers straddling the handle with the tips down at the tabletop — thumb
    # one side, index+middle the other. This is the claw an instant before it
    # closes, so an env starting here only has to learn close+lift. Must stay
    # consistent with SPATULA_SPAWN_POS/QUAT or the hand starts posed for a
    # spatula that is not there.
    # Kept as a plain dict, NOT a grasp_map .pt: *.pt is routed through git-lfs
    # in this repo and arrives unsmudged on some machines, which would silently
    # disable the stage.
    "right_shoulder_pitch_joint": -0.7513,
    "right_shoulder_roll_joint": -0.3003,
    "right_shoulder_yaw_joint": -0.0563,
    "right_elbow_joint": 0.7068,
    "right_wrist_roll_joint": -1.2449,
    "right_wrist_pitch_joint": 0.0090,
    "right_wrist_yaw_joint": 0.4248,
    "right_hand_index_0_joint": 0.0003,
    "right_hand_index_1_joint": 1.7450,
    "right_hand_middle_0_joint": 0.0003,
    "right_hand_middle_1_joint": 1.7450,
    "right_hand_thumb_0_joint": -0.0005,
    "right_hand_thumb_1_joint": 0.7246,
    "right_hand_thumb_2_joint": -1.6386,
}
"""Pregrasp pose [rad] for half the resets (see :data:`PREGRASP_RESET_PROB`)."""

PREGRASP_RESET_PROB = 0.5
"""Share of resets that start in :data:`PREGRASP_JOINT_POS`. The rest start at
the hover so the reach is still learned and the pregrasp is not the only state
the policy ever sees."""

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
            # Contact stiffness for the pinch. Too soft and PPO can discover
            # the pick as a penetration weld instead of a grasp, so the gate
            # on this value is the scripted-pick existence proof: a scripted
            # close-and-lift must hold with <= 10x object weight and release
            # the spatula below 0.5 m/s.
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

    # the three FINGERTIPS pressing the TABLETOP: the claw's opposing surface.
    # Sensed at SHAPE level (one collision mesh per tip) because no single
    # body-name glob selects exactly thumb_2 + index_1 + middle_1 — and the tip
    # set must be exact: a palm-or-any-link version of this signal is farmed by
    # resting the back of the hand on the table. The tabletop is a static
    # AssetBaseCfg with no rigid body, so the filter is shape-level too.
    fingertip_table_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_.*_link",
        sensor_shape_prim_expr=[
            f"{{ENV_REGEX_NS}}/Robot/right_hand/{name}/collisions/.*/mesh" for name in FINGERTIP_BODY_NAMES
        ],
        filter_shape_prim_expr=["{ENV_REGEX_NS}/LabTable/top/.*"],
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
    # Both reset randomizations are off. Spatula jitter in particular is
    # incompatible with the pregrasp half of the mixture below: moving the
    # object leaves those envs posed for a spatula that is no longer there.
    # Reset diversity comes from the pregrasp/hover mixture instead of noise.
    # The mixture: half the envs start in the authored pregrasp (digits already
    # around the spatula, so only close+lift is left to learn), half at the
    # hover so the reach is still learned. Declared LAST so the teleport
    # survives the default resets (event terms run in declaration order).
    reset_pregrasp = EventTerm(
        func=mdp.reset_to_joint_pose,
        mode="reset",
        params={"joint_pose": PREGRASP_JOINT_POS, "probability": PREGRASP_RESET_PROB},
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
    # Pure height indicator. Bare altitude must be worth no more than being
    # near the handle: the reason to lift is that crossing this gate unlocks
    # `track`. Weighting it above that inverts the ordering — a ballistic hop
    # on an unpossessed object out-earns hovering by the handle to timeout, and
    # batting the spatula off the table becomes the best-paying action in the
    # MDP. The invariant this weight has to satisfy is hold >> hover > fling.
    lift = RewTerm(
        func=mdp.object_lifted_in_cage,
        params={
            "rest_height": REST_SPATULA_Z,
            "minimal_offset": 0.03,
            "handle_p0_b": HANDLE_SEGMENT_P0_B,
            "handle_p1_b": HANDLE_SEGMENT_P1_B,
            "palm_radius": PALM_GRASP_RADIUS,
            "thumb_radius": THUMB_PRESS_RADIUS,
            "contact_radius": FINGER_CONTACT_RADIUS,
            "palm_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES),
        },
        weight=1.0,
    )
    # THE task income: coarse + fine kernels on distance to the carry point,
    # paid every step to timeout. There is no success termination to game
    track = RewTerm(
        func=mdp.track_carry_point_in_cage,
        params={
            "carry_point": CARRY_POINT,
            # coarse std 0.30 -> 0.15: at 0.30 a spatula sailing past at 1.3 m
            # still collected 0.024/step, MORE than lift now pays at all
            "stds": (0.15, 0.05),
            "weights": (0.5, 0.5),
            "rest_height": REST_SPATULA_Z,
            "minimal_offset": 0.03,
            "handle_p0_b": HANDLE_SEGMENT_P0_B,
            "handle_p1_b": HANDLE_SEGMENT_P1_B,
            "palm_radius": PALM_GRASP_RADIUS,
            "thumb_radius": THUMB_PRESS_RADIUS,
            "contact_radius": FINGER_CONTACT_RADIUS,
            "palm_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES),
        },
        weight=10.0,
    )
    # Intermediate rung between `reach` (dense) and `lift`+`track` (which need
    # the cage AND height simultaneously). Without something in between, the
    # only income is `reach` and the policy can park short of the handle and
    # farm it forever. This pays for the cage ALONE, ungated by height, and its
    # rate must exceed the rate `reach` pays at its converged distance so that
    # closing the claw beats hovering and carrying beats closing. It has to be
    # payable from step 1 on the pregrasp half of the batch, so the radii must
    # admit the authored pregrasp pose.
    cage = RewTerm(
        func=mdp.grasp_handle,
        params={
            "handle_p0_b": HANDLE_SEGMENT_P0_B,
            "handle_p1_b": HANDLE_SEGMENT_P1_B,
            "palm_radius": PALM_GRASP_RADIUS,
            "thumb_radius": THUMB_PRESS_RADIUS,
            "contact_radius": FINGER_CONTACT_RADIUS,
            "palm_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "fingertips_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES),
        },
        weight=2.0,
    )
    # Pressing the fingertips down into the table is what scoops a thin flat
    # object into the palm, with the tabletop as the opposing surface. At the
    # authored pregrasp the tips are unloaded, so nothing else in the MDP asks
    # them down.
    # Force-based, not height-based: the tip link origins sit at the knuckles,
    # so a tip pad genuinely loaded on the slab still reads several cm of
    # link-origin altitude — a height kernel calibrated on link origins cannot
    # separate pressing from hovering.
    # Self-extinguishing: multiplied by (1 - lifted), so it pays only while the
    # spatula is still down and exactly 0.0 once it is up. Its weight must stay
    # far below `lift` and `track` so camping on it is a loss, which is what
    # keeps it a rung and not an attractor.
    table_press = RewTerm(
        func=mdp.fingertip_table_press,
        params={
            "force_scale": TABLE_PRESS_FORCE_N,
            "rest_height": REST_SPATULA_Z,
            "minimal_offset": 0.03,
        },
        weight=0.5,
    )
    # Start at franka-scale ~zero: pre-contact these penalties are the only
    # gradient the finger dimensions ever sample (zero income, guaranteed tax),
    # so a meaningful weight here drives finger actions and exploration noise
    # to zero before contact is ever made. The curriculum raises them after
    # competence.
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

    The step TARGETS are sized in value units, not in isolation. The old
    -0.05/-0.02 pair moved the per-step penalty from -2.05e-4 to -0.0214, i.e.
    -0.0214 x the ~50-step effective horizon = -1.08 against a total return
    scale of 1.40 — 77% of the whole value scale delivered between two
    consecutive iterations. Measured in run 2026-08-03_17-56-16: mean_reward
    fell 1.403 -> 0.706 across the step (predicted -0.714 from the penalty
    delta alone, i.e. ~100% attributable) while gross income actually ROSE,
    and mean_episode_length went monotone down from 49.9 to 18.2 thereafter.
    The term's own precondition — regularization AFTER competence — was
    measurably false at iteration 250. These targets keep the step at ~3% of
    value scale so the run stays readable.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.009, "num_steps": 6000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_l2", "weight": -0.0004, "num_steps": 6000},
    )


##
# Physics presets
##


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets."""

    # Newton IS the default: the task's contact recipe (shape-filtered
    # force_matrix_w through every mdp consumer, raw spatula triangle
    # colliders on a dynamic body) cannot be expressed under PhysX, so a
    # PhysX preset can only ever fail at env construction.
    default = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=1200,
            nconmax=512,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            # Iteration budget: lower than dexsuite's 100/15, which is
            # conservative for this scene's contact count.
            iterations=50,
            ls_iterations=10,
            ls_parallel=False,
            use_mujoco_contacts=False,
            # fixed-step: the adaptive solver's step-doubling costs far more
            # throughput than this scene's contact difficulty justifies
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
        # 1 substep: the spatula contacts are compliant enough (solref well
        # above the step) to tame the finger/table impact pops on their own, so
        # a second substep buys nothing but throughput cost.
        num_substeps=1,
        debug_mode=False,
    )
    newton_mjwarp = default


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
        # 5 s (300 steps at 60 Hz): the hold income needs runway — the value of
        # a grasp is the flow it pays until timeout, so the episode must be
        # long enough for a completed lift to out-earn any transient.
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
            **DEFAULT_LEG_JOINT_POS,
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
