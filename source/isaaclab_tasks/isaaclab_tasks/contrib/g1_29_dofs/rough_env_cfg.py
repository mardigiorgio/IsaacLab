# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain base config for the 29-DoF G1 velocity task.

The teacher/student observation groups are defined here so the package stays
self-contained. Deviation from the original task: the base env's
``physics_material`` event (fixed 0.8/0.6 friction on robot bodies) is kept.
"""

import torch
import warp as wp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, TerminationsCfg

from .g1_29_dof_cfg import G1_29_DOF_CFG


@configclass
class G1_29_DOFs_Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )


@configclass
class TeacherPolicyCfg(ObsGroup):
    """Privileged (teacher) observations — includes base linear velocity, which
    real robot sensors cannot measure directly."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StudentPolicyCfg(ObsGroup):
    """Real-sensor-only (student) observations: IMU angular velocity and
    projected gravity, joint encoders, and the last action."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class G1_29_DOFs_ObservationsCfg:
    """Teacher-only observations (RL training of the teacher policy)."""

    policy: TeacherPolicyCfg = TeacherPolicyCfg()


@configclass
class StudentObservationsCfg:
    """Student-only observations (RL fine-tuning of the distilled student)."""

    policy: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class TeacherStudentObservationsCfg:
    """Observation groups for teacher-student distillation."""

    # the policy is the student policy
    policy: StudentPolicyCfg = StudentPolicyCfg()
    # the teacher gets the privileged observations
    teacher: TeacherPolicyCfg = TeacherPolicyCfg()


def solver_diverged(env) -> torch.Tensor:
    """Terminate envs whose solver world latched a divergence flag this step.

    Consumes (reads and clears) the adaptive solver's per-world ``diverged``
    latch so the episode ends and the world is re-seeded by the normal reset
    path. Returns all-False when the solver exposes no such latch (fixed-step
    solvers).
    """
    from isaaclab_newton.physics import NewtonManager

    solver = getattr(NewtonManager, "_solver", None)
    mask = getattr(solver, "diverged", None) if solver is not None else None
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    out = wp.to_torch(mask).clone().to(dtype=torch.bool)
    mask.fill_(False)
    return out


@configclass
class G1_29_DOFs_TerminationsCfg(TerminationsCfg):
    """Deviation from the stock task, which terminates only on torso-ground
    contact: the full 29-DoF g1.usd also has pelvis, knee, wrist, and hand
    colliders, so a fallen robot can prop itself on those without ever
    triggering a torso-only rule. Any non-foot ground contact is treated as a
    fall via the extended body list on ``base_contact`` (set in the env
    post_init), plus an orientation catch for tipped poses with no collider
    touching yet."""

    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # Adaptive-solver divergence latch -> episode end + world re-seed (all-False
    # under fixed-step solvers). See :func:`solver_diverged`.
    solver_diverged = DoneTerm(func=solver_diverged)


@configclass
class G1_29_DOFs_RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1_29_DOFs_Rewards = G1_29_DOFs_Rewards()
    observations: G1_29_DOFs_ObservationsCfg = G1_29_DOFs_ObservationsCfg()
    terminations: G1_29_DOFs_TerminationsCfg = G1_29_DOFs_TerminationsCfg()
    # shoulder-pitch + elbow are in the action space (19 actions) so the policy
    # owns arm counter-swing -- the natural damper for roll wobble at speed and
    # for open-loop yaw wander.
    observed_joint_names: list[str] = [
        "waist.*",
        ".*_hip.*",
        ".*_knee.*",
        ".*_ankle.*",
        ".*_shoulder_pitch_joint",
        ".*_elbow_joint",
    ]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_29_DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # Larger standing-env fraction than the base default so the
        # zero-command (hold still at 0 m/s) case is trained. Raised from 0.15:
        # r2 still marched in place at zero command (freeze-hold 0/5).
        self.commands.base_velocity.rel_standing_envs = 0.3
        # Half the envs get directly-sampled yaw-rate commands instead of
        # heading-derived ones: covers turn-in-place (zero linear + sustained
        # yaw never co-occur under pure heading control).
        self.commands.base_velocity.rel_heading_envs = 0.5

        # terminations: every collider except the feet is an illegal ground
        # contact
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
            ".*_knee_link",
            ".*_wrist_yaw_link",
            ".*_hand_palm_link",
        ]
        self.actions.joint_pos.joint_names = self.observed_joint_names
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )


@configclass
class G1_29_DOFs_RoughEnvCfg_PLAY(G1_29_DOFs_RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
