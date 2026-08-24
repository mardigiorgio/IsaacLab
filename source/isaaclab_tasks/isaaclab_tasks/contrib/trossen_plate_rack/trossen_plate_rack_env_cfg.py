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

# Contact-representation switches for the mesh-speed program: hull/slab
# contact is the trained default; PLATE_COLLISION=mesh and
# RACK_COLLISION=mesh run TRI's raw collision meshes instead.
_PLATE_MESH = os.environ.get("PLATE_COLLISION", "hull") == "mesh"
_RACK_MESH = os.environ.get("RACK_COLLISION", "slabs") == "mesh"

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
            # Table-LEFT placement (screen-left of the front camera = -X):
            # the spawn previously sat ~0.15 m from the fixed goal, inside
            # its success kernel's skirt -- the episode began half-solved.
            # -0.24 pulls the rack out of the front camera's center frame
            # +0.05 further along +Y moves it away from the front recording
            # camera (which sits beyond the table on -Y) so the rack does
            # not dominate the video frame.
            pos=(-0.24, _SPAWN_Y + 0.10, 0.02),
            # Yaw returned to identity (a further 180 from the rail-row-
            # toward-the-arm arrangement, ordered after pose-lab
            # inspection): slots still run along Y, rail row away from the
            # arm. Wrist exploration (widened action scale on joints 3-5)
            # compensates the ~24-degree Y-closing misalignment.
            # rot is (x, y, z, w).
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=DISHRACK_USD_PATH.replace("dishrack.usd", "dishrack_mesh.usd") if _RACK_MESH else DISHRACK_USD_PATH
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
                pos=[-0.3398, 0.1143, 0.1488],
                # rot is (x, y, z, w).
                rot=[0.6727, 0.0, 0.0, 0.7399],
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=PLATE_USD_PATH,
                activate_contact_sensors=True,
                collision_props=[
                    sim_utils.schemas.UsdPhysicsCollisionCfg(),
                    # Hulls unconditionally: the pre-split pieces exist for
                    # exactly this, and the raw-mesh path is retired.
                    sim_utils.schemas.UsdPhysicsMeshCollisionCfg(
                        mesh_approximation_name="none" if _PLATE_MESH else "convexHull"
                    ),
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
    """Home starts only: exact plate spawn, scattered arm, no grasp bank."""

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
    randomize_arm_start = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.6, 0.6),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
        },
    )


# IK pre-grasp hover for the settled plate's rim apex, generated and
# VALIDATED by probe_generate_bank (2026-08-23): damped-least-squares IK to
# 8.8 mm of the hover point, then the scripted close-and-raise existence
# proof -- the pinch lifted the plate 398 mm and held. Placement-tracking
# Jacobian omitted: the task is fixed-placement, and the perturbed-placement
# re-solves did not converge (regenerate before any placement DR).
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
    """Replaced by the bedrock anneal in __post_init__ (bank pose exists)."""


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
        self.commands.object_pose.ranges.pos_x = (0.12, 0.12)
        # Reach shapes toward the plate's rim CIRCLE, not its root: the root
        # (disc center) sits inside the rack behind the tines, and a root pull
        # drags the fingers into the wires. The rim kernel reads the live
        # orientation, so it tracks the exposed top arc as the plate leans.
        self.rewards.fingers_to_object = RewTerm(
            func=mdp.mug_rim_ee_distance,
            params={"std": 0.2, "rim_height": PLATE_RIM_HEIGHT, "rim_radius": PLATE_RIM_RADIUS},
            weight=3.0,
        )
        # Wrist exploration: the Y-running slots put the plate grasp ~24
        # degrees off the wrist's natural closing plane, so discovery has to
        # search orientation, not just position. Tripling the wrist joints'
        # action scale widens the per-step orientation search (init_std 0.5
        # x scale 0.3 = +/-0.15 rad commanded jitter) while the proximal
        # joints keep the seat-preserving 0.1.
        self.actions.arm_action.scale = {
            "follower_left_joint_[0-2]": 0.1,
            "follower_left_joint_[3-5]": 0.3,
        }
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
