# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real variant of the Trossen mug lift: the domain-randomized TEACHER
environment for the teacher->student distillation pipeline.

The randomization set mirrors the in-house-validated G1 walking recipe
(``core/velocity/velocity_env_cfg.py``), adapted to manipulation scale:
multiplicative log-uniform mass, per-shape friction bucketing, small COM
shifts, and interval nudges standing in for real-world disturbances. The
teacher keeps CLEAN privileged observations (exact object state) — dynamics
are randomized, sensors are not; heavy sensor corruption belongs to the
camera STUDENT, which is distilled from this teacher with rsl_rl's
distillation runner.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp
from .trossen_mug_lift_env_cfg import TrossenMugLiftEnvCfg


def _apply_teacher_dr(cfg) -> None:
    """Attach the shared dynamics-randomization set to a task cfg's events."""
    # Friction: the mug's authored ceramic mu is 0.2; randomize around it.
    # Pads and table keep their authored values through the robot's own cfg.
    cfg.events.mug_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.1, 0.4),
            "dynamic_friction_range": (0.1, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Mass: multiplicative +-30% log-uniform (G1 pattern; scale-invariant).
    cfg.events.mug_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (1 / 1.3, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # COM: a few millimeters — liquid in a real mug, asymmetric handles. ICF
    # reads body_com live from the model, unlike the MJWarp path where the G1
    # recipe disables this term.
    cfg.events.mug_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.01, 0.01)},
        },
    )
    # Interval nudge on the mug: bumps, tablecloth drag, imperfect resets.
    cfg.events.mug_nudge = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        },
    )
    # Finger-pad friction: the mug-side material is randomized above; the
    # pad side of the grip pair varies with rubber wear and dust in reality.
    # Nominal 1.0 is the vendor model's authored coefficient.
    cfg.events.pad_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="follower_left_gripper_.*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # PD gains: nominals are the vendor's own authored kp/kv (assets.py
    # matches trossen_arm_mujoco: 200/10, 100/5, 50/5 arm; 1000/50 gripper);
    # the scale bracket covers driver-mode and firmware variation.
    cfg.events.arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )
    # Joint transmission: vendor authors armature 0.032/0.0018 and
    # frictionloss 0.1; wear and temperature move both.
    cfg.events.joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
            "friction_distribution_params": (0.5, 2.0),
            "armature_distribution_params": (0.8, 1.25),
            "operation": "scale",
        },
    )


@configclass
class TrossenMugLiftTeacherEnvCfg(TrossenMugLiftEnvCfg):
    """Lift teacher: lift rewards + dynamics randomization, clean obs.

    Object placement randomization (operator spec): SUBTLE position deviation
    from the tape-measure center — the hardware protocol places the mug by
    hand, not by jig — and FULL 360-degree yaw, so the handle may face
    anywhere. Heavy SENSOR noise deliberately does NOT live here: the teacher
    trains on clean privileged state; corruption belongs to the camera
    student distilled from it."""

    def __post_init__(self):
        super().__post_init__()
        _apply_teacher_dr(self)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "yaw": (-3.14159, 3.14159),
        }
        # The banked pre-grasp FOLLOWS the randomized placement (one
        # Jacobian step, probe-verified sub-mm over this range), and bank
        # envs re-sample yaw into the handle-safe arc so the teleported
        # straddle can never intersect the handle.
        from .trossen_mug_lift_env_cfg import BANK_POSE_XY_JACOBIAN, _SPAWN_X, _SPAWN_Y  # noqa: PLC0415

        bank = self.events.reset_arm_grasp_bank
        bank.params["track_object_xy"] = BANK_POSE_XY_JACOBIAN
        bank.params["nominal_object_pos"] = (_SPAWN_X, _SPAWN_Y)
        bank.params["safe_yaw_range"] = (1.047, 5.236)


@configclass
class TrossenMugLiftDistillEnvCfg(TrossenMugLiftTeacherEnvCfg):
    """Distillation stage: the teacher's randomized world, TWO observation
    groups — ``policy`` (clean privileged state, what the trained teacher
    reads) and ``student`` (the same signals under HEAVY sensor noise, what
    the real rig's proprioception looks like). The G1-family pattern: the
    student is state-based; cameras are a later stage if hardware demands
    them."""

    def __post_init__(self):
        super().__post_init__()

        policy = self.observations.policy

        @configclass
        class StudentObsCfg(ObsGroup):
            joint_pos = ObsTerm(func=policy.joint_pos.func, noise=Unoise(n_min=-0.05, n_max=0.05))
            joint_vel = ObsTerm(func=policy.joint_vel.func, noise=Unoise(n_min=-2.0, n_max=2.0))
            contact = ObsTerm(
                func=policy.contact.func,
                params=dict(policy.contact.params),
                clip=(-20.0, 20.0),
                noise=Unoise(n_min=-2.0, n_max=2.0),
            )
            object_position = ObsTerm(
                func=policy.object_position.func, noise=Unoise(n_min=-0.02, n_max=0.02)
            )
            object_orientation = ObsTerm(
                func=policy.object_orientation.func,
                params=dict(policy.object_orientation.params),
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
            target_object_position = ObsTerm(
                func=policy.target_object_position.func, params=dict(policy.target_object_position.params)
            )
            actions = ObsTerm(func=policy.actions.func, clip=(-6.0, 6.0))

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        self.observations.student = StudentObsCfg()
        # the privileged group stays clean for the teacher side of distillation
        self.observations.policy.enable_corruption = False
