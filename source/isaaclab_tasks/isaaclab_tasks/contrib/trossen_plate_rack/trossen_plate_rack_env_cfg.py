# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI PLATE PICK: lift a plate out of a dishrack slot.

The thin-shell stiff-contact task for the adaptive-vs-fixed comparison — the
CENIC paper's own hardest regime (dishrack clutter) as a learnable
manipulation task: a 4 mm ceramic shell standing between 6 mm wire tines,
where a coarse fixed step tunnels or ejects and only resolved contact can
thread the slot.

Everything transfers from the validated mug lift: the rig, the physics stack,
the action/observation/command/termination structure, the reward economics.
The deltas are the scene (plate in rack replaces mug on table), the reach
target (the plate's rim circle — the orientation-agnostic rim kernel finds
the exposed top arc of a standing plate unchanged), and the start
distribution (home scatter only: no teleop pre-grasp exists for the plate
yet, and the wide-gripper discovery arithmetic that found the mug pinch
applies to the thinner plate rim a fortiori)."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_Y,
    TrossenMugLiftEnvCfg,
    TrossenMugLiftSceneCfg,
)

from . import mdp
from .assets import DISHRACK_USD_PATH, PLATE_USD_PATH

# Contact representation, fixed by decision (no switches): every TRI
# model collides as its RAW authored triangle mesh -- plate and rack alike
# (fitted slabs looked nothing like the visual at contact level, and the
# scene's whole role is thin-geometry fidelity). Only the rig keeps
# primitive/hull colliders, frame and table included.

# Rim circle of the TRI ikea_dinera plate in its body frame, measured by
# assets/convert_plate_rack.py from the source mesh (R=0.0981, rim z=0.0178):
# the pinch target ring sits just inside the outer lip so the nearest-rim-
# point kernel pulls the TCP onto the graspable edge, not the outer face.
PLATE_RIM_HEIGHT = 0.016
PLATE_RIM_RADIUS = 0.095
# A plate standing in a slot rests its bottom rim on the rack's base tray
# (tabletop 0.02 + tray height 0.032), disc center one radius up.
PLATE_STAND_Z = 0.02 + 0.032 + 0.098
# Center-most slot of the sweet_home wireframe, from the converter's slot
# census: slot center x = -0.017 in the rack frame.
SLOT_X_OFFSET = -0.017


@configclass
class PlateRackSceneCfg(TrossenMugLiftSceneCfg):
    """The rig scene with the mug replaced by a plate standing in a dishrack."""

    # Kinematic wire rack, centered on the tape-measure spot the mug used.
    rack: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        # 50 mm toward the arm vs the mug spot: the wrist-constrained Y-closing
        # plate grasp measured 10-25 mm past dexterous reach at the mug
        # placement; the plate protocol authors its own tape point.
        init_state=AssetBaseCfg.InitialStateCfg(
            # the spawn previously sat ~0.15 m from the fixed goal, inside
            # its success kernel's skirt -- the episode began half-solved.
            # -0.10: a nudge right of the original -0.18 tape point, rack
            # fully on the slab, clear of the camera center line
            # +0.05 further along +Y moves it away from the front recording
            # camera (which sits beyond the table on -Y) so the rack does
            # not dominate the video frame.
            pos=(-0.08, _SPAWN_Y + 0.08, 0.02),
            # Yaw returned to identity (a further 180 from the rail-row-
            # toward-the-arm arrangement, ordered after pose-lab
            # inspection): slots still run along Y, rail row away from the
            # arm. Wrist exploration (widened action scale on joints 3-5)
            # compensates the ~24-degree Y-closing misalignment.
            # rot is (x, y, z, w).
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=DISHRACK_USD_PATH.replace("dishrack.usd", "dishrack_mesh.usd")
        ),
    )

    def __post_init__(self):
        # Plate standing vertically in the center slot: body Z (the disc
        # axis) rotated onto env +X by a +90-degree pitch, so the plate
        # plane spans Y-Z and the slot tines bracket it along X.
        self.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                # TRI's own plate-in-rack pose: the PutSpatulaInUtensilCrock
                # FromDryingRack scenario welds the plate to a slot rail with
                # X_PC t=[0.00922, 0.00230, 0.02007], rpy=[84.55, 0, 90] deg;
                # center rail (plate_slot_rail_4) chosen, frames mapped into
                # this rack USD by vertex registration of the wireframe mesh
                # (rotation Rz(-90), translation (0, 0, 0.032), residual
                # 2e-6 m). TRI holds this pose with a weld; here it lives
                # under real contact, so a small settle slide off the
                # authored pose is expected and TRI-faithful. Verify with
                # the settle probe (in-slot, finite) after any rack change.
                pos=[-0.1798, 0.0943, 0.1488],
                # rot is (x, y, z, w).
                rot=[0.6727, 0.0, 0.0, 0.7399],
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=PLATE_USD_PATH,
                activate_contact_sensors=True,
                collision_props=[
                    sim_utils.schemas.UsdPhysicsCollisionCfg(),
                    # Per-piece convex hulls: the plate's 25 collision pieces
                    # approximate as hulls (arm likewise, via the rig USD);
                    # only the dishrack collides as a raw triangle mesh.
                    sim_utils.schemas.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
                    self.object.spawn.collision_props[2],  # rig-matched MujocoCollisionCfg(solref)
                ],
                # TRI's authored coefficient for THIS plate (its SDF's
                # drake:mu_static/mu_dynamic = 0.3) -- NOT the mug's 0.2,
                # which the earlier revision recycled.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.3,
                    dynamic_friction=0.3,
                    restitution=0.0,
                ),
                rigid_props=self.object.spawn.rigid_props,
            ),
        )
        # The mug's sensor filters name its piece set; the plate's pieces are
        # collisions_base / _wall_[0-11] / _rim_[0-11].
        self.arm_body_contact.filter_shape_prim_expr = ["{ENV_REGEX_NS}/Object/collisions_.*/.*"]


@configclass
class PlateEventCfg:
    """Fixed starts: exact plate spawn, arm at home/ready, no grasp bank.

    No arm scatter, deliberately: at this rack placement the mug task's
    +/-0.6 rad scatter samples poses that spawn the gripper inside the
    rack basket. Exploration is the policy's own noise plus the widened
    wrist action scale."""

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


# UNWIRED and UNVALIDATED for the current scene: this pose was generated
# on a pre-rotation rack and its existence proof is void. Kept only as a
# starting seed for the pose lab / a future probe_generate_bank rerun;
# nothing reads it at runtime.
PLATE_BANK_POSE = {
    "follower_left_joint_0": 0.1759,
    "follower_left_joint_1": 2.3190,
    "follower_left_joint_2": 1.5155,
    "follower_left_joint_3": 1.5698,
    "follower_left_joint_4": 1.5698,
    "follower_left_joint_5": 0.8560,
    "follower_left_left_carriage_joint": 0.0440,
    "follower_left_right_carriage_joint": 0.0440,
}


@configclass
class PlateCurriculumCfg:
    """Empty: bootstrap mode trains from home scatter, no curriculum."""


@configclass
class TrossenPlatePickEnvCfg(TrossenMugLiftEnvCfg):
    scene: PlateRackSceneCfg = PlateRackSceneCfg(num_envs=8192, env_spacing=2.5)
    events: PlateEventCfg = PlateEventCfg()
    curriculum: PlateCurriculumCfg = PlateCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Goal on the table's RIGHT (+X): with the rack at the left edge the
        # commanded carry point is ~0.3 m from the spawn, outside the success
        # kernel's skirt -- the transport has to be earned.
        self.commands.object_pose.ranges.pos_x = (0.18, 0.18)
        # Reach shapes toward the plate's rim CIRCLE, not its root: the root
        # (disc center) sits inside the rack behind the tines, and a root pull
        # drags the fingers into the wires. The rim kernel reads the live
        # orientation, so it tracks the exposed top arc as the plate leans.
        self.rewards.fingers_to_object = RewTerm(
            func=mdp.mug_rim_ee_distance,
            params={"std": 0.2, "rim_height": PLATE_RIM_HEIGHT, "rim_radius": PLATE_RIM_RADIUS},
            weight=3.0,
        )
        # FULL articulation, ordered: the commandable set covers every
        # joint's whole range (scale 0.5 x clip 6 = +/-3 rad, clamped to
        # the hardware limits downstream), so any grasp posture the arm
        # can physically reach is expressible. The mug tasks' 0.1 was a
        # PD-transient guard tuned for a straight-in pinch; the plate
        # grasp needs free orientation search and pays the transient
        # cost knowingly.
        self.actions.arm_action.scale = {
            # Shoulders position the wrist; the pre-grasp search itself is a
            # wrist-orientation problem, so the distal axes get the widest
            # band: at policy std 1.5 the doorknob axes (4-5) sample ~2.2 rad
            # of commanded rotation per step — real turning motions, not
            # dithering. PD transients accepted knowingly (see above).
            "follower_left_joint_[0-2]": 0.5,
            "follower_left_joint_3": 1.0,
            "follower_left_joint_[4-5]": 1.5,
        }
        # Finger discovery: the lift's gripper scale guards a settled pinch;
        # the plate first needs fingers that actually open and close during
        # search. 0.15 commands the full 0.044 m stroke well inside 1 sigma.
        self.actions.gripper_action.scale = 0.15
        # Online success in the logger (Metrics/success_rate): the command
        # measures the OBJECT against the commanded pose with the committed
        # evaluator's gates. Logging-only — no reward or dynamics change.
        self.commands.object_pose.class_type = mdp.ObjectPoseSuccessCommand
        self.commands.object_pose.position_success_threshold = mdp.SUCCESS_POS_THRESHOLD
        self.commands.object_pose.orientation_success_threshold = mdp.SUCCESS_ORI_THRESHOLD
        # BOOTSTRAP MODE: no bank. The generated pose above predates the
        # rack reorientation and its proof was invalidated (contaminated by
        # an interpenetrating spawn); regeneration on the final scene has not
        # yet produced a pose that passes both gates. Home scatter plus the
        # wide-gripper discovery scale is the recipe's bootstrap mode -- the
        # mug's close was discovered exactly this way before its bank
        # existed. Wire apply_reverse_curriculum back in only with a pose
        # that passes the clearance AND hold gates on THIS scene.


@configclass
class TrossenPlatePickEnvCfg_PLAY(TrossenPlatePickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
