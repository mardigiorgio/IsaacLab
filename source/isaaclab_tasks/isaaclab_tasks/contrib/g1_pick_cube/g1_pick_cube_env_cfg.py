# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 pick-cube env at the real lab table.

The G1 crouches at the table in a fixed-base squat: a CEM reachability probe
(2026-07-07, see the project ledger) showed that at the original standing
height (pelvis 0.75 m) no valid joint configuration brings a palm closer than
0.26 m to the cube — the 0.289 m tabletop is simply out of reach. With the
squat base pose and the cube spawned near the robot-side table edge, the palm
reaches a pre-grasp pose 4.6 cm above the cube center.

Rewards are staged: coarse + fine palm reaching, finger closing gated to
palm-near-cube, dense lift progress, and a hold bonus at the success height.
There is deliberately no success termination — terminating on lift while
paying dense height rewards teaches hovering just below the threshold;
holding the cube up pays instead.
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

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT, lab_table_cfgs
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

CUBE_REST_Z = LAB_TABLE_HEIGHT + 0.02
"""DexCube resting center height on the tabletop [m].

The DexCube asset is natively 6 cm; it is scaled down to 4 cm on spawn
(see :attr:`G1PickCubeSceneCfg.cube`), so half its 4 cm edge (0.02 m) is
added to the tabletop height.
"""

LIFT_SUCCESS_HEIGHT = LAB_TABLE_HEIGHT + 0.15
"""Cube center height that counts as a successful lift [m]."""

SQUAT_BASE_POS = (0.0, -0.60, 0.55)
"""Fixed-base pelvis position [m]: crouched at the long table edge.

Standing height (0.75 m) is kinematically unable to reach the tabletop
(see module docstring); 0.55 m emulates a squatting G1.
"""

SQUAT_LEG_JOINT_POS = {".*_hip_pitch_joint": -0.75, ".*_knee_joint": 1.5, ".*_ankle_pitch_joint": -0.7}
"""Leg joint defaults [rad] posing the legs as a squat.

Keeps the knees clear of the table edge and the feet just above the ground at
:data:`SQUAT_BASE_POS`; the legs carry no load with the root fixed. The env
holds this pose with stiff implicit actuators (the asset's explicit DC-motor
leg actuators would leave unactuated legs dangling under gravity).
"""

CUBE_SPAWN_POS = (0.0, -0.16, CUBE_REST_Z)
"""Nominal cube spawn [m]: near the robot-side table edge, within palm reach."""

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
            # DexCube is natively 6 cm; scale it down to the 4 cm cube this task wants.
            scale=(2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
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
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            # kept tight around CUBE_SPAWN_POS: the squatting G1's reachable
            # patch on the tabletop is small (see module docstring)
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.04, 0.04)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Staged pick-up shaping: reach -> close fingers -> lift -> hold.

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
        weight=8.0,
    )
    lifted_hold = RewTerm(func=mdp.cube_lifted, params={"minimum_height": LIFT_SUCCESS_HEIGHT}, weight=10.0)
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
        self.episode_length_s = 8.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        # fixed base: the G1 crouches at the table, no locomotion in this task
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # squat at the long edge (-Y); the asset default rot already faces the
        # table (+Y, yaw +90deg). Standing cannot reach the tabletop (see
        # module docstring), so the base is lowered and the legs posed bent.
        self.scene.robot.init_state.pos = SQUAT_BASE_POS
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **SQUAT_LEG_JOINT_POS,
        }
        # hold the squat: the asset's explicit DC-motor leg actuators leave
        # unactuated legs dangling under gravity, so pin the pose with stiff
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
