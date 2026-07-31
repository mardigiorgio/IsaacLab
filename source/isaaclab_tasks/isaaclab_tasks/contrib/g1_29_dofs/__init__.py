# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""29-DoF G1 velocity task (``Isaac-Velocity-Flat-G1-v1``).

Includes the teacher-student distillation pipeline: train teacher ->
distill student -> fine-tune student.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-G1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

###############################
# Teacher-Student Distillation #
###############################
gym.register(
    id="Velocity-G1-Distillation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatTeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_VelocityDistillationRunnerCfg",
    },
)

########################
# Student-Finetune #
########################
gym.register(
    id="Velocity-G1-Student-Finetune-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatStudentPPORunnerCfg",
    },
)

####################################################
# Soft-contact variants (adaptive solver; see cfg) #
####################################################
gym.register(
    id="Isaac-Velocity-Flat-G1-Soft-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatEnvCfg_Soft",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-Distillation-Soft-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatTeacherStudentEnvCfg_Soft",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_VelocityDistillationRunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-Student-Finetune-Soft-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29_DOFs_FlatStudentEnvCfg_Soft",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatStudentPPORunnerCfg",
    },
)

#############################################################
# Legs-only standing balance (external upper-body policies) #
#############################################################
gym.register(
    id="Isaac-Stand-G1-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:G1_29_DOFs_StandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_StandPPORunnerCfg",
    },
)

gym.register(
    id="Stand-G1-Distillation-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:G1_29_DOFs_StandTeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_StandDistillationRunnerCfg",
    },
)

gym.register(
    id="Stand-G1-Student-Finetune-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:G1_29_DOFs_StandStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_StandStudentPPORunnerCfg",
    },
)

##########################################
# Total-domain-randomization variants #
##########################################
gym.register(
    id="Isaac-Velocity-Flat-G1-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_dr:G1_29_DOFs_FlatEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatPPORunnerCfg_DR",
    },
)

gym.register(
    id="Velocity-G1-Distillation-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_dr:G1_29_DOFs_FlatTeacherStudentEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_VelocityDistillationRunnerCfg",
    },
)

gym.register(
    id="Velocity-G1-Student-Finetune-DR-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_dr:G1_29_DOFs_FlatStudentEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29_DOFs_FlatStudentPPORunnerCfg",
    },
)
