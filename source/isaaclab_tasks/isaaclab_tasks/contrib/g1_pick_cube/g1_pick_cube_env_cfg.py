# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 pick-cube env at the real lab table.

The G1 stands fixed-base at the long edge of the (hip-height, 0.734 m) lab
table, toes aligned to the table-edge plane. A Rubik's-cube-sized DexCube
spawns ANYWHERE on the tabletop; the objective is always the same: pick it
up and hold it at a fixed point in front of the chest
(:data:`HOLD_TARGET_OFFSET`, robot-relative). Spawns beyond the fixed-base
reach envelope (|x| beyond roughly 0.5 m) are deliberately included, and the
robot's placement is jittered slightly around its nominal stance.

Rewards are staged: coarse + fine palm reaching, finger closing gated to
palm-near-cube, dense lift progress, goal tracking toward the hold point,
and a hold bonus near it. There is deliberately no success termination —
terminating while paying dense proximity rewards teaches hovering short of
the objective; holding the cube at the target pays instead.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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

from isaaclab_tasks.utils.hydra import PresetCfg

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT, LAB_TABLE_WIDTH, lab_table_cfgs
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

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

CUBE_EDGE = 0.057
"""Cube edge length [m]: a standard Rubik's cube (57 mm)."""

CUBE_REST_Z = LAB_TABLE_HEIGHT + CUBE_EDGE / 2
"""DexCube resting center height on the tabletop [m].

The DexCube asset is natively 6 cm; it is scaled to :data:`CUBE_EDGE` on
spawn (see :attr:`G1PickCubeSceneCfg.cube`).
"""

LIFT_SUCCESS_HEIGHT = LAB_TABLE_HEIGHT + 0.15
"""Cube center height that counts as a successful lift [m]."""

CUBE_SPAWN_POS = (0.0, 0.0, CUBE_REST_Z)
"""Nominal cube spawn [m]: tabletop center; randomized across the whole surface."""

HOLD_TARGET_OFFSET = (0.20, 0.0, 0.20)
"""Objective point in the ROBOT ROOT frame [m]: 20 cm ahead of and above the pelvis.

Wherever on the table the cube spawns — and wherever the (jittered) robot
stands — it must be brought to this same spot in front of the G1's chest,
just over the table edge, slightly above the ready-pose palms.
"""

G1_TOE_REACH = 0.142
"""Forward extent of the G1 toe tips from the pelvis origin [m].

Measured from the foot-mesh world bounding box in the standing pose
(scratchpad ``ready_pose_and_toes.py``, 2026-07-07).
"""

TOE_STANDOFF = 0.02
"""Gap between the toe tips and the table-edge plane [m].

Keeps the feet visibly clear of the table footprint; alignment tolerance in
the real world is about this size anyway.
"""

ROBOT_STAND_POS = (0.0, -(LAB_TABLE_WIDTH / 2 + G1_TOE_REACH + TOE_STANDOFF), 0.75)
"""Fixed-base pelvis position [m]: standing at the long table edge.

The y places the TOE TIPS a :data:`TOE_STANDOFF` gap short of the vertical
plane dropped from the table's near edge. Real-world alignment: the front
faces of the table legs are flush with that plane, so lay a straight bar on
the floor against both near legs and bring the toes up to it (the bar's
thickness conveniently provides the standoff), then remove the bar.
"""

READY_ARM_JOINT_POS = {"left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12}
"""Ready-pose tweak [rad]: a subtle shoulder splay over the asset's zero arms.

The G1 zero pose already has a built-in ~90 deg elbow bend that puts the
palms in front of the chest ~0.11 m above the tabletop — a natural
manipulation-ready stance; only the slight arm splay is added.
"""

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
            # DexCube is natively 6 cm; scale it to the Rubik's-cube size this task wants
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
    # raw actions are clipped: unbounded Gaussian exploration against the stiff
    # (3 kN·m/rad) arm PD can otherwise explode the fixed-step solver into NaNs
    upper_body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UPPER_BODY_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-3.0, 3.0)},
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
        cube_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        # SceneEntityCfgs must be passed via params: managers only resolve
        # body/joint names to indices for cfgs found in params, never for a
        # term function's default arguments
        palm_to_cube = ObsTerm(
            func=mdp.palms_to_object_vector,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # write the robot joint state back to defaults on every reset; without
    # this the articulation keeps (Newton) or drifts from (PhysX) its previous
    # joint state and only the PD targets pull it toward the squat
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # placement jitter: the real robot is never positioned exactly — vary the
    # stance around mid-table/up-close so the policy tolerates misalignment
    # (works for the fixed base: the world joint re-anchors on root writes)
    jitter_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.03, 0.02), "yaw": (-0.08, 0.08)},
            "velocity_range": {},
        },
    )
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            # the whole tabletop (with a margin so the cube never spawns
            # hanging over an edge); spawns beyond the fixed-base reach are
            # deliberate — the policy should handle any cube position
            "pose_range": {"x": (-0.87, 0.87), "y": (-0.27, 0.27)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Staged shaping: reach -> close fingers -> lift -> carry to the hold target.

    SceneEntityCfgs are passed via ``params`` deliberately: managers only
    resolve body/joint names for cfgs found there, never for a term
    function's default arguments.
    """

    reaching = RewTerm(
        func=mdp.palms_to_cube_distance_reward,
        params={
            "std": 0.30,
            "robot_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
        },
        weight=1.0,
    )
    reaching_fine = RewTerm(
        func=mdp.palms_to_cube_distance_reward,
        params={
            "std": 0.05,
            "robot_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
        },
        weight=1.0,
    )
    grasp_fingers = RewTerm(
        func=mdp.fingers_closed_near_cube,
        params={
            "distance_threshold": 0.08,
            "palms_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
            "left_fingers_cfg": SceneEntityCfg("robot", joint_names=["left_hand_index_.*", "left_hand_middle_.*"]),
            "right_fingers_cfg": SceneEntityCfg("robot", joint_names=["right_hand_index_.*", "right_hand_middle_.*"]),
        },
        weight=1.0,
    )
    lift_progress = RewTerm(
        func=mdp.object_lift_progress,
        params={"rest_height": CUBE_REST_Z, "target_height": LIFT_SUCCESS_HEIGHT},
        weight=4.0,
    )
    # the objective: carry the cube to the hold point in front of the chest,
    # wherever on the table it spawned and wherever the robot stands
    goal_tracking = RewTerm(
        func=mdp.object_to_target_distance_reward,
        params={"std": 0.25, "target_offset": HOLD_TARGET_OFFSET},
        weight=6.0,
    )
    held_at_target = RewTerm(
        func=mdp.object_near_target,
        params={"threshold": 0.10, "target_offset": HOLD_TARGET_OFFSET},
        weight=10.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # knocking the cube off the table must not be free: without this the
    # policy settles on flicking the cube around (74% drop terminations)
    dropped_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "cube_dropped"}, weight=-50.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # cube knocked off the table: well below the tabletop it is lost
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": LAB_TABLE_HEIGHT - 0.09, "asset_cfg": SceneEntityCfg("cube")},
    )
    # rare solver divergence guard: blowups either pass through finite huge
    # velocities on the way to NaN or jump straight to non-finite — both must
    # reset the env instead of poisoning the whole rollout. Fingers are
    # excluded from the velocity check: their tiny links spike legitimately on
    # contact, and they never diverge without the arm chain going with them.
    robot_exploded = DoneTerm(
        func=mdp.robot_or_object_state_invalid,
        params={
            "max_velocity": 200.0,
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
            ),
        },
    )


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
        # 4 substeps (1.25 ms): PPO exploration slams the stiff arms into the
        # ke=1e6 contacts; the stack task's 2 substeps NaN out under that
        num_substeps=4,
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
        # long enough to reach a far cube and carry it back to the hold point
        self.episode_length_s = 10.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        # fixed base: the G1 stands at the table, no locomotion in this task
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # stand at the long edge (-Y), toes on the table-edge plane (see
        # ROBOT_STAND_POS for the real-world alignment convention); the asset
        # default rot already faces the table (+Y, yaw +90deg)
        self.scene.robot.init_state.pos = ROBOT_STAND_POS
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **READY_ARM_JOINT_POS,
        }
        # pin the leg pose: the asset's explicit DC-motor leg actuators leave
        # unactuated legs sagging under gravity, so hold them with stiff
        # implicit PD instead (legs are static scenery with the root fixed)
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
        # bound the arm/hand efforts to hardware-plausible values: the asset's
        # 300 N*m everywhere lets random exploration targets fight the joint
        # limits with unbounded energy and NaN the solver (fingers blow up in
        # a single step); bounded torque makes arbitrary policy actions safe
        self.scene.robot.actuators["arms"].effort_limit = None
        self.scene.robot.actuators["arms"].effort_limit_sim = 60.0
        self.scene.robot.actuators["arms"].damping = 30.0
        self.scene.robot.actuators["hands"].effort_limit = None
        self.scene.robot.actuators["hands"].effort_limit_sim = 5.0
