# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skeleton G1 pick-cube env at the real lab table.

Rewards here are PLACEHOLDERS to make the env trainable end-to-end; reward
design is intentionally left to the user (see the task board's PPO item).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT, lab_table_cfgs
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.utils.hydra import PresetCfg

from . import mdp

##
# Constants
##

UPPER_BODY_JOINT_NAMES = [
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hand_.*_joint",
]
"""Actuated joint set: waist, both arms, both TriHand hands."""

CUBE_REST_Z = LAB_TABLE_HEIGHT + 0.02
"""DexCube (4 cm) resting center height on the tabletop [m]."""

LIFT_SUCCESS_HEIGHT = LAB_TABLE_HEIGHT + 0.15
"""Cube center height that counts as a successful lift [m]."""

_TABLE = lab_table_cfgs("{ENV_REGEX_NS}/LabTable")


##
# Scene
##


@configclass
class G1PickCubeSceneCfg(InteractiveSceneCfg):
    """Ground + lab table + fixed-base G1 + DexCube."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # lab table (five static parts from the shared factory)
    table_top = _TABLE["table_top"]
    table_leg_0 = _TABLE["table_leg_0"]
    table_leg_1 = _TABLE["table_leg_1"]
    table_leg_2 = _TABLE["table_leg_2"]
    table_leg_3 = _TABLE["table_leg_3"]

    # robot: fixed-base G1 standing at the long edge (-Y side), facing the table (+Y)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.05, CUBE_REST_Z)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    upper_body = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=UPPER_BODY_JOINT_NAMES, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
        )
        cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.10)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    # ------------------------------------------------------------------
    # PLACEHOLDER reward — intentionally simple. Reward design is owned by
    # the user (task board: "Pick up cube rl policy (PPO)").
    # ------------------------------------------------------------------
    reaching = RewTerm(func=mdp.ee_to_cube_distance_reward, params={"std": 0.2}, weight=1.0)
    lifted = RewTerm(func=mdp.cube_lifted, params={"minimum_height": LIFT_SUCCESS_HEIGHT}, weight=10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.cube_lifted, params={"minimum_height": LIFT_SUCCESS_HEIGHT})


##
# Physics presets — copied from the stack task's tuned block (contact
# authoring rationale documented there; keep in sync deliberately, not
# by abstraction: presets are per-task tunables).
##


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for the G1 pick-cube task."""

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
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=35,
            sap_solver_iterations=64,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        # Stiff contact so the TriHand fingers stall on the cube instead of
        # closing through it (see the stack task preset for the full rationale).
        default_shape_cfg=NewtonShapeCfg(ke=1e6, kd=2000),
        num_substeps=2,
        debug_mode=False,
    )
    physx = default


##
# Env cfg
##


@configclass
class G1PickCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Skeleton G1 pick-cube environment at the lab table."""

    scene: G1PickCubeSceneCfg = G1PickCubeSceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        # fixed base: the G1 stands at the table, no locomotion in this task
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # place the robot at the long edge (-Y), facing the table (+Y, yaw +90deg)
        self.scene.robot.init_state.pos = (0.0, -0.55, 0.75)
        self.scene.robot.init_state.rot = (0.7071, 0.0, 0.0, 0.7071)
