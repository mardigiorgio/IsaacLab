# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VR teleoperation of the G1 spatula pick-up scene.

Same scene as :mod:`.g1_spatula_lift_env_cfg` — fixed-base G1 at the 83 cm lab
table with the LBM spatula — but driven by a human through OpenXR instead of a
policy. The point is an existence proof: if a person can pick the spatula up in
THIS simulation, the physics support the grasp and the remaining problem is
learning. If they cannot, no reward shaping will help.

Everything teleop-related is inherited from
:class:`~isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg`
— the Pink IK upper-body action, the two ``TriHandMotionControllerRetargeter``
pipelines that map controller triggers onto the 7 TriHand joints per hand, and
the :class:`IsaacTeleopCfg` server. Only the scene and the physics preset are
replaced here.

Quest 2/3 connects through the CloudXR JS (auto-webrtc) profile; see
``isaaclab_teleop.isaac_teleop_cfg.CLOUDXR_JS_ENV``.
"""

from isaaclab_teleop import XrCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mdp import reset_scene_to_default as mdp_reset_scene_to_default
from isaaclab.envs.mdp import time_out as mdp_time_out
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)

from isaaclab_assets.props.lab_table import lab_table_cfgs
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from . import mdp
from .g1_spatula_lift_env_cfg import (
    DEFAULT_ARM_JOINT_POS,
    ROBOT_STAND_POS,
    SPATULA_SPAWN_POS,
    SPATULA_SPAWN_QUAT,
    SPATULA_USD_PATH,
    PhysicsCfg,
)
from .teleop_retargeters import build_spatula_teleop_pipeline

_TABLE = lab_table_cfgs("{ENV_REGEX_NS}/LabTable")

HAND_STIFFNESS = 60.0
"""Finger PD stiffness [N·m/rad] for teleop. Softer than the RL task's 100.0, but
NOT the 20.0 first tried: the pregrasp commands 1.28 rad of index flexion, and at
K=20 that demands 20 x 1.28 = 25.6 N.m against a 5 N.m effort limit, so the digit
stalls far short of the pose and reads as "not actuating". 60 x the residual error
stays inside the limit once the finger is near its target. Lower it if the hand
feels stiff against the spatula; raise it if a digit visibly fails to reach."""

HAND_DAMPING = 4.0
"""Finger PD damping [N·m·s/rad], scaled with :data:`HAND_STIFFNESS` to keep the
joint near critical damping rather than ringing on table contact."""

XR_ANCHOR_POS = (0.0, ROBOT_STAND_POS[1], 0.0)
"""OpenXR world anchor [m]: the floor directly under the robot's fixed pelvis,
so the operator standing at their play-space origin is co-located with the G1
rather than offset across the room. This is the FIRST thing to retune in
headset if the hands do not line up with the table."""


##
# Scene definition
##


@configclass
class G1SpatulaTeleopSceneCfg(InteractiveSceneCfg):
    """The spatula scene, named for the teleop env's observation/termination terms.

    The spatula entity is deliberately named ``object``: the inherited
    observations and terminations resolve ``SceneEntityCfg("object")``, so
    reusing that name keeps the whole parent pipeline working untouched.
    """

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

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Spatula",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=SPATULA_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=0.5,
            ),
            # identical contact regime to the RL task, so a successful human
            # pick here is evidence ABOUT the RL task and not about a
            # softer/stiffer variant of it
            compliant_contact_stiffness=111000.0,
            compliant_contact_damping=667.0,
            physics_material_prim_path=["collisions_blade/mesh", "collisions_handle/mesh"],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SPATULA_SPAWN_POS, rot=SPATULA_SPAWN_QUAT),
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class TeleopTerminationsCfg:
    """Minimal terminations: a human demo must not be cut short.

    The parent's pick-place ``success`` term and its 0.5 m drop check are
    dropped — the first is scored against a different task, and an operator
    fumbling the spatula should get to try again rather than have the episode
    reset out from under them. Only the (long) clock ends an episode.
    """

    time_out = DoneTerm(func=mdp_time_out, time_out=True)


@configclass
class TeleopEventCfg:
    """Reset events.

    ``hold_inert_joints`` is the load-bearing one. In this fork a joint with no
    action term keeps PD target 0.0 — it drives to the ZERO pose, NOT to
    ``init_state.joint_pos``. With the IK restricted to the right arm, that
    left the left arm standing straight out in front of the robot and every
    left-hand finger splayed, because zero is "straight" for both. Writing the
    defaults once per reset pins them where they belong: left arm hanging at
    the side (elbow 1.45), left hand relaxed, waist and legs static.
    """

    reset_all = EventTerm(func=mdp_reset_scene_to_default, mode="reset")

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


##
# Environment configuration
##


@configclass
class G1SpatulaTeleopEnvCfg(FixedBaseUpperBodyIKG1EnvCfg):
    """Human-in-the-loop version of the spatula pick-up task."""

    scene: G1SpatulaTeleopSceneCfg = G1SpatulaTeleopSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    terminations: TeleopTerminationsCfg = TeleopTerminationsCfg()
    events: TeleopEventCfg = TeleopEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # inherit the Pink IK action, the TriHand retargeters and the
        # IsaacTeleop server wiring
        super().__post_init__()

        # 5 minutes: long enough to fumble, re-approach and retry without a reset
        self.episode_length_s = 300.0

        # Physics IDENTICAL to the RL task (240 Hz, decimation 4 -> 60 Hz
        # control), not the parent's 200 Hz. The whole value of this env is that
        # a human pick here is evidence about THAT task; a different contact
        # timestep would break that argument. The docs' advice to drop sim dt to
        # the headset rate is declined for the same reason — a 6 mm-thick handle
        # raked off a table is exactly where a coarse contact step lies.
        # Smoothness is bought with render_interval instead: 6 x (1/240) = 40 Hz
        # render, comfortably feeding a Quest 2 at its native 72 Hz.
        self.sim.dt = 1 / 240
        self.decimation = 4
        self.sim.render_interval = 6

        # the spatula's thin raw collision meshes need the Newton preset the RL
        # task was tuned on. NOTE: unlike the training entry point,
        # ``teleop_se3_agent.py`` parses with ``parse_known_args`` and has no
        # ``presets=`` hook, so a CLI override is impossible — the preset has to
        # be chosen here or :class:`PresetCfg` silently falls back to ``default``
        # (PhysX) and the human demo would be collected under contact physics
        # the RL task never sees, invalidating the comparison.
        _physics = PhysicsCfg()
        _physics.default = _physics.newton_mjwarp
        self.sim.physics = _physics

        # Pink IK's gravity-compensation feedforward raises NotImplementedError
        # on Newton (no gravity-compensation primitive upstream). Same fix as
        # the G1 pick-place teleop task: worth ~3 mrad at these PD gains.
        self.actions.upper_body_ik.enable_gravity_compensation = False

        # stand the robot at the lab table exactly as the RL task does, so a
        # human pick here transfers as evidence about that task's geometry
        self.scene.robot.init_state.pos = ROBOT_STAND_POS
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            **DEFAULT_ARM_JOINT_POS,
        }

        # legs/waist are static scenery on a fixed base — the asset's DC-motor
        # gains sag without this (same fix as the pick-place teleop task)
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
        # SOFT finger PD (operator request). The RL task runs 100/10, which is
        # firm enough to shove the spatula rather than conform to it; under a
        # human driving preset poses a compliant hand lets the digits settle
        # around the handle instead of batting it. Raise these together if the
        # fingers sag under gravity and cannot press into the tabletop — that
        # is the failure mode 8/1.5 (the asset stock) exhibits.
        self.scene.robot.actuators["hands"].effort_limit = None
        self.scene.robot.actuators["hands"].effort_limit_sim = 5.0
        self.scene.robot.actuators["hands"].stiffness = HAND_STIFFNESS
        self.scene.robot.actuators["hands"].damping = HAND_DAMPING

        # put the operator's play-space origin under the robot, not under the
        # packing table the parent env anchors to
        self.xr = XrCfg(anchor_pos=XR_ANCHOR_POS, anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop.xr_cfg = self.xr

        # RIGHT ARM ONLY. The inherited G1_UPPER_BODY_IK_ACTION_CFG lists
        # ".*_shoulder_*", ".*_elbow_joint", ".*_wrist_*" and "waist_.*_joint",
        # so the IK solver was free to move the LEFT arm and the WAIST as well.
        # Restricting the solver's joint set leaves them out of the IK entirely;
        # the action tensor keeps its 28D layout (that is set by the eef targets
        # and hand joints, not by this list), so the pipeline is unaffected.
        self.actions.upper_body_ik.pink_controlled_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
            "right_wrist_yaw_joint",
        ]

        # right hand on PRESET POSES rather than the stock analytic mapping:
        #   trigger        -> the authored pregrasp claw straddle
        #   trigger+grip   -> claw raked shut
        # see teleop_retargeters.py to retune CLOSED_POSE from in-headset.
        self.isaac_teleop.pipeline_builder = lambda: build_spatula_teleop_pipeline()[0]
