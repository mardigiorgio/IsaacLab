# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI MUG INTO DISHRACK: pick the mug off the table by the
rim, invert it, and set it rim-down onto the dishrack's wire lattice in its
open mug bay, then let go and return home.

TRI's ``MugInDishRack_1`` start state (the upside-down variant) on the single
arm. The rack is the plate scene's (TRI sweet_home, raw wire mesh); the mug
and its spawn are the lift's; the economy is the mug hang's staged FSM
(hang_fsm_core) with the rack's own "placed" predicate.

GOAL POSE: measured, not TRI's range center -- an inverted mug rests stably
on the lattice at only ~3 of 16 cells in TRI's bay range (the rim must
straddle wires on both sides); settle sweep 2026-08-28 found the best rest at
TRI wireframe (-0.017, 0.055), 8 mm drift, up_z -1.00.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab_tasks.contrib.trossen_mug_lift.bedrock import apply_reverse_curriculum
from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_X,
    _SPAWN_Y,
    BANK_POSE_XY_JACOBIAN,
    GRASP_BANK_POSE,
    TerminationsCfg,
    TrossenMugLiftEnvCfg,
    TrossenMugLiftSceneCfg,
)
from isaaclab_tasks.contrib.trossen_mug_tree.trossen_mug_tree_env_cfg import ARM_FINISH_POSE, HangRewardsCfg
from isaaclab_tasks.contrib.trossen_plate_rack.assets import DISHRACK_USD_PATH

from . import mdp

# ============================================================================ USER-SET
# Rack placement: the plate scene's (TRI's -90 yaw already baked into the USD).
RACK_POS = (-0.08, _SPAWN_Y + 0.08, 0.02)
# Measured stable inverted rest on the lattice (env frame): root position and
# orientation from the settle sweep; the goal the policy must match.
GOAL_POSE_ENV = ((-0.0168, 0.0975, 0.1719), (-3.1297, 0.0033, 1.3314))
# The "in the bay" band around it: half-width in xy the rest survives, root z band.
BAY_HALF_XY = 0.03
BAY_Z = (0.155, 0.19)
# ============================================================================


@configclass
class MugRackSceneCfg(TrossenMugLiftSceneCfg):
    """The rig scene plus the plate scene's rack; the mug at its tape spot (inherited)."""

    rack: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=DISHRACK_USD_PATH.replace("dishrack.usd", "dishrack_mesh.usd")),
    )
    # Mug body touching the rack's wire/base meshes: the FSM's support signal.
    object_rack_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/Rack/collisions_.*/.*"],
    )


@configclass
class RackEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
    )
    randomize_arm_start = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]")},
    )


@configclass
class RackTerminationsCfg(TerminationsCfg):
    task_complete = DoneTerm(
        func=mdp.rack_fsm,
        params={
            "pose": ARM_FINISH_POSE,
            "joint_tol": 0.5,
            "bay_center_env": GOAL_POSE_ENV[0],
            "bay_half_xy": BAY_HALF_XY,
            "z_range": BAY_Z,
            "sensor_name": "pad_object_contact",
            "rack_sensor_name": "object_rack_contact",
        },
    )


@configclass
class RackCurriculumCfg:
    """Bedrock attaches the bank anneal."""


@configclass
class TrossenMugRackPlaceEnvCfg(TrossenMugLiftEnvCfg):
    scene: MugRackSceneCfg = MugRackSceneCfg(num_envs=2000, env_spacing=2.5)
    rewards: HangRewardsCfg = HangRewardsCfg()
    events: RackEventCfg = RackEventCfg()
    terminations: RackTerminationsCfg = RackTerminationsCfg()
    curriculum: RackCurriculumCfg = RackCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The hang's physics and control rate (5 subdivisions of 1/90 at 30 Hz), 8 s episodes.
        self.sim.dt = 1 / 450
        self.decimation = 15
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        # The hang's action space: wrist search for the inversion.
        self.actions.arm_action.scale = {"follower_left_joint_[0-2]": 0.5, "follower_left_joint_[3-5]": 1.0}
        # Command = the measured goal pose (drawn by the marker; the lab's --object-at-goal drops the mug there).
        (gx, gy, gz), (gr, gp, gyaw) = GOAL_POSE_ENV
        rg = self.commands.object_pose.ranges
        rg.pos_x, rg.pos_y, rg.pos_z = (gx, gx), (gy, gy), (gz, gz)
        rg.roll, rg.pitch, rg.yaw = (gr, gr), (gp, gp), (gyaw, gyaw)
        # The hang's staged rewards read env._fsm, which rack_fsm publishes.
        self.rewards.milestones.func = mdp.fsm_milestones
        self.rewards.shaping.func = mdp.fsm_shaping
        # Bank starts: the lift's validated pre-grasp (same mug, same spot).
        apply_reverse_curriculum(
            self, bank_pose=GRASP_BANK_POSE, bank_xy_jacobian=BANK_POSE_XY_JACOBIAN,
            nominal_object_xy=(_SPAWN_X, _SPAWN_Y), bank_fraction=0.5, end_step=2_400,
        )


@configclass
class TrossenMugRackPlaceEnvCfg_PLAY(TrossenMugRackPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.reset_arm_grasp_bank.params["bank_fraction"] = 0.0
