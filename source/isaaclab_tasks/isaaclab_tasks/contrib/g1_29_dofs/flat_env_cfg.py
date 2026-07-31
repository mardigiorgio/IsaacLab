# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain 29-DoF G1 velocity task (``Isaac-Velocity-Flat-G1-v1``).

Physics-configuration notes:

- ``implicitfast`` is the only implicit integrator the current backend exposes.
- Contacts come from Newton's CollisionPipeline (``use_mujoco_contacts=False``)
  with the foot colliders kept as raw collision meshes.
- The height scanner is dropped (the policy never observes it).
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import G1_29_DOFs_RoughEnvCfg, StudentObservationsCfg, TeacherStudentObservationsCfg

# Near-flat terrain with 0-2 cm uniform-noise roughness: the carpet/doorsill
# class of surface the deployed flat policy catches its swing feet on. The
# policy stays terrain-blind (no height scan); the roughness only forces real
# swing clearance instead of ground-skimming.
CARPET_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.02), noise_step=0.005, border_width=0.25
        ),
    },
)


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Newton CollisionPipeline contacts (not MuJoCo's internal convex-hull
            # pipeline); the adaptive solver consumes the same injected contacts.
            njmax=400,
            nconmax=200,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            ls_iterations=10,
            ls_parallel=True,
            use_mujoco_contacts=False,
        ),
        # rigid_contact_max must be >= nconmax * nworld; the auto-estimator can
        # size below that, which breaks CUDA graph capture.
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=2_000_000, max_triangle_pairs=2_500_000),
        num_substeps=1,
        # Stiff foot-ground contact instead of the more compliant default.
        default_shape_cfg=NewtonShapeCfg(ke=1.0e6, kd=2000.0),
        # Selective fidelity: the foot colliders keep their RAW collision meshes;
        # every other collider hulls as usual.
        simplify_meshes=True,
        simplify_meshes_exclude=[".*ankle_roll_link.*"],
        debug_mode=False,
    )


@configclass
class SoftPhysicsCfg(PresetCfg):
    """Adaptive-solver physics: identical to :class:`PhysicsCfg` except stock
    default contact stiffness. Required for the adaptive solver — the stiff
    (ke=1e6) foot contact is stable under the fixed solver but not under the
    adaptive one."""

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=400,
            nconmax=200,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            ls_iterations=10,
            ls_parallel=True,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=2_000_000, max_triangle_pairs=2_500_000),
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(),
        simplify_meshes=True,
        simplify_meshes_exclude=[".*ankle_roll_link.*"],
        debug_mode=False,
    )


@configclass
class G1_29_DOFs_FlatEnvCfg(G1_29_DOFs_RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # near-flat terrain: 0-2 cm carpet-grade roughness instead of a perfect
        # plane (swing feet must actually clear the ground)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = CARPET_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        # no height scan (the policy never observes it)
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1_29_DOFs_FlatEnvCfg_PLAY(G1_29_DOFs_FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1_29_DOFs_FlatTeacherStudentEnvCfg(G1_29_DOFs_FlatEnvCfg):
    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.teacher.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.teacher.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        # reduce the teacher observation noise during distillation
        self.observations.teacher.base_lin_vel.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.teacher.base_ang_vel.noise = Unoise(n_min=-0.002, n_max=0.002)
        self.observations.teacher.projected_gravity.noise = Unoise(n_min=-0.0005, n_max=0.0005)
        self.observations.teacher.joint_pos.noise = Unoise(n_min=-0.0001, n_max=0.0001)
        self.observations.teacher.joint_vel.noise = Unoise(n_min=-0.0001, n_max=0.0001)


@configclass
class G1_29_DOFs_FlatStudentEnvCfg(G1_29_DOFs_FlatEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )


##
# Soft-contact variants for the adaptive solver (see SoftPhysicsCfg docstring).
##


@configclass
class G1_29_DOFs_FlatEnvCfg_Soft(G1_29_DOFs_FlatEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=SoftPhysicsCfg())


@configclass
class G1_29_DOFs_FlatTeacherStudentEnvCfg_Soft(G1_29_DOFs_FlatTeacherStudentEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=SoftPhysicsCfg())


@configclass
class G1_29_DOFs_FlatStudentEnvCfg_Soft(G1_29_DOFs_FlatStudentEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=SoftPhysicsCfg())
