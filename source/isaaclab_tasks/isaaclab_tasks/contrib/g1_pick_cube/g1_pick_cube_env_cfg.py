# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 pick-up-cube env at the lab table.

A fixed-base G1 stands at the lab table with a cube at a fixed pose on the
tabletop. Task: pick the cube up with the right TriHand and bring it to a
fixed hold point in front of the chest. Reaching the hold point ends the
episode successfully. Nothing is randomized.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.hydra import PresetCfg

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT, LAB_TABLE_WIDTH, lab_table_cfgs
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from . import mdp

##
# Constants
##

ACTUATED_JOINT_NAMES = [
    "waist_.*_joint",
    "right_shoulder_.*_joint",
    "right_elbow_joint",
    "right_wrist_.*_joint",
    "right_hand_.*_joint",
]
"""Policy-actuated joints: full waist (yaw+roll+pitch) + right arm + right
TriHand fingers (17 dims). The left arm and the legs are inert, held at their
defaults by the ``hold_inert_joints`` event."""

CUBE_EDGE = 0.057
"""Cube edge length [m] (a Rubik's cube). The DexCube asset is natively 6 cm
and is scaled to this."""

CUBE_REST_Z = LAB_TABLE_HEIGHT + CUBE_EDGE / 2
"""Cube center height at rest on the tabletop [m]."""

CUBE_SPAWN_POS = (0.0, 0.0, CUBE_REST_Z)
"""Cube spawn, env frame [m]: resting at the table center in front of the
robot. No randomization."""

HOLD_TARGET_OFFSET = (0.20, 0.0, 0.20)
"""Hold target for the cube, ROBOT ROOT frame [m]: in front of the chest
(~20 cm forward, 20 cm up from the pelvis)."""

SUCCESS_RADIUS = 0.10
"""Success: cube center within this distance [m] of the hold target."""

LIFT_TARGET_Z = LAB_TABLE_HEIGHT + 0.15
"""Height [m] toward which the dense lifting term pays."""

G1_TOE_REACH = 0.142
"""Forward extent of the G1 toe tips from the pelvis origin [m]."""

TOE_STANDOFF = 0.02
"""Gap between the toe tips and the table-edge plane [m]."""

ROBOT_STAND_POS = (0.0, -(LAB_TABLE_WIDTH / 2 + G1_TOE_REACH + TOE_STANDOFF), 0.75)
"""Fixed-base pelvis position [m]: standing at the long table edge, facing the table (+Y)."""

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
    # right arm hovering over the table, palm down
    "right_shoulder_pitch_joint": -0.35,
    "right_shoulder_roll_joint": -0.30,
    "right_shoulder_yaw_joint": -0.10,
    "right_elbow_joint": 1.05,
    "right_wrist_roll_joint": 0.30,
    "right_wrist_pitch_joint": -0.90,
    "right_wrist_yaw_joint": -0.30,
}
"""Default arm pose [rad]: left down by the side, right hovering over the table."""

_TABLE = lab_table_cfgs("{ENV_REGEX_NS}/LabTable")


##
# Scene definition
##


@configclass
class G1PickCubeSceneCfg(InteractiveSceneCfg):
    """Ground + lab table + fixed-base G1 + cube."""

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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(CUBE_EDGE / 0.06, CUBE_EDGE / 0.06, CUBE_EDGE / 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_SPAWN_POS),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # relative joint-position deltas, scale 0.1 rad/step; raw actions clipped
    # (unclipped Gaussian exploration against the stiff arm PD destabilizes
    # the solver)
    upper_body = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINT_NAMES,
        scale=0.1,
        clip={".*": (-3.0, 3.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)},
        )
        cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        cube_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        palm_to_cube = ObsTerm(
            func=mdp.palm_to_object_vector,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"])},
        )
        cube_to_target = ObsTerm(
            func=mdp.object_to_target_vector,
            params={"target_offset": HOLD_TARGET_OFFSET},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # non-actioned joints (left arm/hand, legs, feet) get their PD TARGETS
    # written to the defaults — in this fork untargeted joints otherwise
    # drive to the ZERO pose, not init_state.joint_pos
    hold_inert_joints = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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


@configclass
class RewardsCfg:
    """Minimal pick-and-hold shaping: reach → close → lift → carry to the target."""

    reaching = RewTerm(
        func=mdp.palm_to_object_distance_reward,
        params={
            "std": 0.3,
            "robot_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
        },
        weight=1.0,
    )
    # SceneEntityCfgs must be passed via params: managers only resolve
    # body/joint names for cfgs found there (function defaults stay
    # unresolved and silently select every body)
    grasp_fingers = RewTerm(
        func=mdp.fingers_closed_near_object,
        params={
            "distance_threshold": 0.10,
            "palm_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
            "fingers_cfg": SceneEntityCfg("robot", joint_names=["right_hand_.*_joint"]),
        },
        weight=0.5,
    )
    lifting = RewTerm(
        func=mdp.object_lift_progress,
        params={"rest_height": CUBE_REST_Z, "target_height": LIFT_TARGET_Z},
        weight=2.0,
    )
    goal_tracking = RewTerm(
        func=mdp.object_to_target_distance_reward,
        params={"std": 0.25, "target_offset": HOLD_TARGET_OFFSET},
        weight=2.0,
    )
    # a success termination's bonus must exceed the dense income it cuts off,
    # or the policy hovers just short of success and farms shaping forever
    success_bonus = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "success"}, weight=50.0)
    dropped_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "cube_dropped"}, weight=-5.0)
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # pick-and-hold complete: cube at the chest target
    success = DoneTerm(
        func=mdp.object_near_target,
        params={"threshold": SUCCESS_RADIUS, "target_offset": HOLD_TARGET_OFFSET},
    )

    # cube off the table or batted out of reach
    cube_dropped = DoneTerm(
        func=mdp.object_out_of_bound,
        params={
            "in_bound_range": {
                "x": (-0.7, 0.7),
                "y": (-0.35, 0.35),
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
            iterations=50,
            ls_iterations=10,
            ls_parallel=False,
            use_mujoco_contacts=False,
            # fixed-step: the adaptive solver is a large tax on contact-rich
            # scenes (measured on the sibling spatula task)
            adaptive=False,
            ccd_iterations=35,
            sap_solver_iterations=64,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            rigid_contact_max=8_000_000,
            max_triangle_pairs=16_000_000,
        ),
        simplify_meshes=True,
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    physx = default


##
# Environment configuration
##


@configclass
class G1PickCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 pick-up-cube environment."""

    viewer: ViewerCfg = ViewerCfg(eye=(1.4, -1.0, 1.3), lookat=(0.0, -0.1, 0.85), origin_type="env")
    scene: G1PickCubeSceneCfg = G1PickCubeSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # 120 Hz physics, decimation 2 -> 60 Hz control; picking something up
        # is a two-second task
        self.decimation = 2
        self.episode_length_s = 3.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        # fixed base: the G1 stands bolted at the table (the real robot's
        # factory controller owns balance)
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.pos = ROBOT_STAND_POS
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **DEFAULT_ARM_JOINT_POS,
        }
        # pin the leg pose with stiff implicit PD (legs are static scenery
        # with the root fixed; the asset's DC-motor gains would sag).
        # full waist is policy-actuated with moderate PD and an effort cap
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
