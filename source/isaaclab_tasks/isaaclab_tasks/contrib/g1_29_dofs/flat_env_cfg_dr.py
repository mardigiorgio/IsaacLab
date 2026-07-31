# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Total-domain-randomization variants of the 29-DoF G1 velocity task.

The DR levers are applied identically to the teacher, distillation, and
student-finetune stages: the student must train in the same randomized
dynamics.

r3 CONTRACT CHANGE: shoulder-pitch + elbow joined ``observed_joint_names``
(see :mod:`.rough_env_cfg`), so the policy now has 19 actions and the student
observation is 66-dim (gyro 3 / gravity 3 / cmd 3 / qpos 19 / qvel 19 /
prev-action 19; teacher adds base_lin_vel -> 69). The deploy stack needs a
fresh joint-order dump and updated obs/action constants before running an r3
checkpoint. Scaling (target = default + 0.5 * action) is unchanged.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.config.spot.mdp.rewards import foot_clearance_reward
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg

from . import mdp_dr
from .flat_env_cfg import (
    G1_29_DOFs_FlatEnvCfg,
    G1_29_DOFs_FlatStudentEnvCfg,
    G1_29_DOFs_FlatTeacherStudentEnvCfg,
)
from .rough_env_cfg import G1_29_DOFs_Rewards

# every actuated joint group except the ankles (which get their own, wider DR)
_NON_ANKLE_JOINTS = [
    "waist_.*_joint",
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hand_.*",
]
_ANKLE_JOINTS = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]


@configclass
class G1_29_DOFs_DREventsCfg(EventsCfg):
    """Base events plus the levers that don't exist as base fields."""

    # ±15% PD-gain scaling on non-ankle actuators. Startup-only:
    # implicit-actuator gain writes go through the CPU.
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_NON_ANKLE_JOINTS),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # ±50% on the ankles: the real ankle is a parallel mechanism the serial sim
    # model can't capture — the policy must not assume precise ankle authority.
    ankle_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_JOINTS),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ankle_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_JOINTS),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    ankle_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_JOINTS),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # per-episode constant IMU attitude bias (±2.5 deg) + gyro bias (±0.05 rad/s)
    imu_bias = EventTerm(
        func=mdp_dr.resample_imu_bias,
        mode="reset",
        params={"max_tilt_deg": 2.5, "max_gyro_bias": 0.05},
    )


@configclass
class G1_29_DOFs_DRRewards(G1_29_DOFs_Rewards):
    """Base rewards plus an ankle-weighted action-rate penalty (the ankles carry
    the action chatter; combined with the global action_rate_l2 the effective
    ankle rate weight is ~3x the other joints), the r3 stillness objective
    (settled double support at zero command), and a swing-apex clearance reward
    for the carpet-catch class of trips."""

    ankle_action_rate_l2 = RewTerm(func=mdp_dr.ankle_action_rate_l2, weight=-0.01)
    stand_still_feet_off_ground = RewTerm(
        func=mdp_dr.stand_still_feet_off_ground,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    stand_still_joint_vel = RewTerm(
        func=mdp_dr.stand_still_joint_vel,
        weight=-0.01,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # feet_air_time rewards swing DURATION, not height, and the effort penalties
    # favor ground-skimming swings — this term makes the swing apex track a real
    # clearance. target_height is the ankle_roll frame (~3.5 cm above the sole),
    # so 0.12 m ~= 8-9 cm sole clearance. Swing-gated by foot xy-speed.
    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.12,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )


def _apply_total_dr(cfg) -> None:
    """Re-enable and widen every randomization lever on a v1-task config.

    Must run after the rough cfg's post_init (which disables mass/COM/push),
    so plain field assignments here win.
    """
    # ground friction: widen from the fixed 0.8/0.6 base values
    cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
    # torso mass ±25% (payload / battery / model error)
    cfg.events.add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # center-of-mass randomization on the torso
    cfg.events.base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    # per-episode constant wrench on the torso (unmodeled cabling/backpack bias)
    cfg.events.base_external_force_torque.params["force_range"] = (-10.0, 10.0)
    cfg.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
    # reset-state diversity
    cfg.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }
    # interval shoves
    cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-0.5, 0.5)}},
    )
    # actuation latency: 0-1 control steps of per-episode delay + first-order lag
    cfg.actions.joint_pos = mdp_dr.DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(cfg.observed_joint_names),
        scale=0.5,
        use_default_offset=True,
        delay_prob=0.5,
        lag_beta_range=(0.5, 1.0),
    )
    # per-episode IMU bias enters through the student-visible observations;
    # the distillation teacher group stays clean.
    cfg.observations.policy.projected_gravity.func = mdp_dr.projected_gravity_biased
    cfg.observations.policy.base_ang_vel.func = mdp_dr.base_ang_vel_biased
    # command realism: faster resampling, joystick-style ramps, oversampled
    # decelerate-through-zero, and a velocity range that puts the deployed
    # envelope (|vx| <= 1.0 at the joystick) deep in the interior.
    old = cfg.commands.base_velocity
    cfg.commands.base_velocity = mdp_dr.SlewedVelocityCommandCfg(
        asset_name=old.asset_name,
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=old.rel_standing_envs,
        rel_heading_envs=old.rel_heading_envs,
        heading_command=old.heading_command,
        heading_control_stiffness=old.heading_control_stiffness,
        debug_vis=old.debug_vis,
        ranges=type(old.ranges)(
            # past the deployed envelope: wobble rose toward the old range edge,
            # so the 0.6 driving cap must sit deep in the interior
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=old.ranges.lin_vel_y,
            ang_vel_z=old.ranges.ang_vel_z,
            heading=old.ranges.heading,
        ),
        ramp_fraction=0.5,
        accel=1.5,
        backward_winddown_accel=0.25,
        stop_from_backward_prob=0.3,
    )


@configclass
class G1_29_DOFs_FlatEnvCfg_DR(G1_29_DOFs_FlatEnvCfg):
    events: G1_29_DOFs_DREventsCfg = G1_29_DOFs_DREventsCfg()
    rewards: G1_29_DOFs_DRRewards = G1_29_DOFs_DRRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_total_dr(self)


@configclass
class G1_29_DOFs_FlatTeacherStudentEnvCfg_DR(G1_29_DOFs_FlatTeacherStudentEnvCfg):
    events: G1_29_DOFs_DREventsCfg = G1_29_DOFs_DREventsCfg()
    rewards: G1_29_DOFs_DRRewards = G1_29_DOFs_DRRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_total_dr(self)


@configclass
class G1_29_DOFs_FlatStudentEnvCfg_DR(G1_29_DOFs_FlatStudentEnvCfg):
    events: G1_29_DOFs_DREventsCfg = G1_29_DOFs_DREventsCfg()
    rewards: G1_29_DOFs_DRRewards = G1_29_DOFs_DRRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_total_dr(self)
