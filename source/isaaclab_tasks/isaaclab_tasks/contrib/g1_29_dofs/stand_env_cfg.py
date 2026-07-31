# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legs-only standing balance task for the 29-DoF G1.

The policy actuates the 12 leg joints and holds the robot upright while the
waist/arm joints are driven by :class:`mdp_dr.UpperBodyMotionCommand`,
emulating an independent upper-body controller (pick-and-place etc.). No
control input: balance is the whole task. Full v2 DR set plus per-hand payload
randomization. Recovery steps are allowed but penalized.

Policy observation contract (76): biased gyro (3), biased projected gravity
(3), leg qpos - default (12), leg qvel (12), upper-body qpos - default (17),
upper-body commanded targets - default (17), previous action (12). Teacher
adds privileged base_lin_vel (3).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import mdp_dr
from .flat_env_cfg import PhysicsCfg
from .flat_env_cfg_dr import G1_29_DOFs_DREventsCfg
from .g1_29_dof_cfg import G1_29_DOF_CFG
from .rough_env_cfg import G1_29_DOFs_TerminationsCfg

_LEG_JOINTS = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
_UPPER_JOINTS = ["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]


##
# Observations
##


@configclass
class StandTeacherPolicyCfg(ObsGroup):
    """Privileged (teacher) observations — includes base linear velocity."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    upper_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UPPER_JOINTS)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    upper_joint_target = ObsTerm(func=mdp_dr.upper_body_command)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StandStudentPolicyCfg(ObsGroup):
    """Real-sensor-only (student) observations."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    upper_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UPPER_JOINTS)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    upper_joint_target = ObsTerm(func=mdp_dr.upper_body_command)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StandObservationsCfg:
    policy: StandTeacherPolicyCfg = StandTeacherPolicyCfg()


@configclass
class StandStudentObservationsCfg:
    policy: StandStudentPolicyCfg = StandStudentPolicyCfg()


@configclass
class StandTeacherStudentObservationsCfg:
    policy: StandStudentPolicyCfg = StandStudentPolicyCfg()
    teacher: StandTeacherPolicyCfg = StandTeacherPolicyCfg()


##
# Rewards / events / commands
##


@configclass
class StandRewards:
    """Stillness is the task; everything else shapes how it stands."""

    root_stillness = RewTerm(func=mdp_dr.root_stillness_exp, weight=2.0, params={"std": 0.25})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # height BAND, not target: squatting for balance is free; only collapsing
    # below (or hopping above) the band is penalized
    base_height_band = RewTerm(
        func=mdp_dr.base_height_band, weight=-20.0, params={"min_height": 0.55, "max_height": 0.78}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # dense feet-planted objective: drift from the episode anchor costs reward
    # every step (the termination only catches outright steps)
    feet_hold = RewTerm(func=mdp_dr.feet_displacement_penalty, weight=-10.0)
    # each airborne foot costs reward
    feet_contact_break = RewTerm(
        func=mdp_dr.feet_contact_break,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    # feet-forward geometry only: sagittal joints (hip pitch, knee, ankle
    # pitch) stay penalty-free so the squat dimension is unconstrained
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    ankle_action_rate_l2 = RewTerm(func=mdp_dr.ankle_action_rate_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class StandTerminationsCfg(G1_29_DOFs_TerminationsCfg):
    """Fall detector plus the no-step rule."""

    # an unambiguous step (>15 cm from anchor) ends the episode; the desk
    # contract's "feet planted" is enforced by the dense drift penalty, whose
    # optimum is zero displacement — the loose termination just closes the
    # stepping exit without being an unlearnable cliff
    feet_displacement = DoneTerm(func=mdp_dr.feet_displacement, params={"threshold": 0.15})


@configclass
class StandEventsCfg(G1_29_DOFs_DREventsCfg):
    """Full DR event set plus payload emulation and the hand-target hold."""

    # 0-1.5 kg added per wrist each episode (carried-object emulation; the
    # class-based mass event randomizes from stored defaults, no accumulation)
    payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
            "mass_distribution_params": (0.0, 1.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # hand joints are neither actioned nor driven — hold their defaults
    hold_hands = EventTerm(
        func=mdp_dr.hold_joints_at_default,
        mode="reset",
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=[".*_hand_.*"])},
    )
    # manipulation-reaction disturbances: bounded FORCES the posture can resist
    # (velocity kicks teleport momentum and force stepping — see post_init)
    torso_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "force_range": (-35.0, 35.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    wrist_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wrist_yaw_link"]),
            "force_range": (-15.0, 15.0),
            "torque_range": (-1.0, 1.0),
        },
    )
    # anchor AFTER the state-reset events (field order = execution order): the
    # feet_displacement termination measures against this
    anchor_feet = EventTerm(func=mdp_dr.anchor_feet, mode="reset")


@configclass
class StandCommandsCfg:
    # realistic manipulation speeds (deploy stacks rate-limit the arms; training
    # against full-limit 3 rad/s flailing over-hardens the task)
    upper_body = mdp_dr.UpperBodyMotionCommandCfg(speed_range=(0.3, 1.5))


##
# Environment configs
##


@configclass
class G1_29_DOFs_StandEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    observations: StandObservationsCfg = StandObservationsCfg()
    rewards: StandRewards = StandRewards()
    events: StandEventsCfg = StandEventsCfg()
    commands: StandCommandsCfg = StandCommandsCfg()
    terminations: StandTerminationsCfg = StandTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene: flat plane, no height scan
        self.scene.robot = G1_29_DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # actions: legs only, with actuation latency + first-order lag
        self.actions.joint_pos = mdp_dr.DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=_LEG_JOINTS,
            scale=0.5,
            use_default_offset=True,
            delay_prob=0.5,
            lag_beta_range=(0.5, 1.0),
        )

        # terminations: any non-foot ground contact is a fall
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
            ".*_knee_link",
            ".*_wrist_yaw_link",
            ".*_hand_palm_link",
        ]

        # dynamics DR (same levers as the walking DR task)
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (0.8, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        # reset kicks capped at what feet-planted balance can physically absorb
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.1, 0.1),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.15, 0.15),
        }
        # NO velocity-kick pushes: with feet planted, the capturable kick is
        # ~0.3 m/s — walking-grade shoves force stepping/stance-widening, which
        # the desk contract forbids. Disturbances are the upper-body motion,
        # payloads, and the bounded force events instead.
        self.events.push_robot = None

        # per-episode IMU bias enters through the policy-visible observations
        self.observations.policy.projected_gravity.func = mdp_dr.projected_gravity_biased
        self.observations.policy.base_ang_vel.func = mdp_dr.base_ang_vel_biased


@configclass
class G1_29_DOFs_StandTeacherStudentEnvCfg(G1_29_DOFs_StandEnvCfg):
    observations: StandTeacherStudentObservationsCfg = StandTeacherStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        # reduce the teacher observation noise during distillation
        self.observations.teacher.base_lin_vel.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.teacher.base_ang_vel.noise = Unoise(n_min=-0.002, n_max=0.002)
        self.observations.teacher.projected_gravity.noise = Unoise(n_min=-0.0005, n_max=0.0005)
        self.observations.teacher.joint_pos.noise = Unoise(n_min=-0.0001, n_max=0.0001)
        self.observations.teacher.joint_vel.noise = Unoise(n_min=-0.0001, n_max=0.0001)
        self.observations.teacher.upper_joint_pos.noise = Unoise(n_min=-0.0001, n_max=0.0001)


@configclass
class G1_29_DOFs_StandStudentEnvCfg(G1_29_DOFs_StandEnvCfg):
    observations: StandStudentObservationsCfg = StandStudentObservationsCfg()
