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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_X,
    _SPAWN_Y,
    TrossenMugLiftEnvCfg,
    TrossenMugLiftSceneCfg,
)

from . import mdp
from .assets import DISHRACK_USD_PATH, PLATE_USD_PATH

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_SPAWN_X, _SPAWN_Y, 0.02)),
        spawn=sim_utils.UsdFileCfg(usd_path=DISHRACK_USD_PATH),
    )

    def __post_init__(self):
        # Plate standing vertically in the center slot: body Z (the disc
        # axis) rotated onto env +X by a +90-degree pitch, so the plate
        # plane spans Y-Z and the slot tines bracket it along X.
        self.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[_SPAWN_X + SLOT_X_OFFSET, _SPAWN_Y, PLATE_STAND_Z],
                # rot is (x, y, z, w): pitched 84.55 degrees about Y -- the
                # lbm_eval scenarios' AUTHORED resting lean of this plate in
                # this rack (X_PC of the plate_slot_rail weld), 5.45 degrees
                # off vertical against the wires. Spawning exactly vertical
                # puts the plate on an unstable equilibrium instead of its
                # settled pose. TRI's weld height lands the center at 0.1488
                # over the table, agreeing with PLATE_STAND_Z to ~1 mm.
                rot=[0.0, 0.67260, 0.0, 0.74000],
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=PLATE_USD_PATH,
                activate_contact_sensors=True,
                collision_props=[
                    sim_utils.schemas.UsdPhysicsCollisionCfg(),
                    # Hulls unconditionally: the pre-split pieces exist for
                    # exactly this, and the raw-mesh path is retired.
                    sim_utils.schemas.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
                    self.object.spawn.collision_props[2],  # rig-matched MujocoCollisionCfg(solref)
                ],
                physics_material=self.object.spawn.physics_material,
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


@configclass
class PlateCurriculumCfg:
    """No curriculum: there is no bank event to anneal."""


@configclass
class TrossenPlatePickEnvCfg(TrossenMugLiftEnvCfg):
    scene: PlateRackSceneCfg = PlateRackSceneCfg(num_envs=8192, env_spacing=2.5)
    events: PlateEventCfg = PlateEventCfg()
    curriculum: PlateCurriculumCfg = PlateCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Reach shapes toward the plate's rim CIRCLE, not its root: the root
        # (disc center) sits inside the rack behind the tines, and a root pull
        # drags the fingers into the wires. The rim kernel reads the live
        # orientation, so it tracks the exposed top arc as the plate leans.
        self.rewards.fingers_to_object = RewTerm(
            func=mdp.mug_rim_ee_distance,
            params={"std": 0.2, "rim_height": PLATE_RIM_HEIGHT, "rim_radius": PLATE_RIM_RADIUS},
            weight=3.0,
        )


@configclass
class TrossenPlatePickEnvCfg_PLAY(TrossenPlatePickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
