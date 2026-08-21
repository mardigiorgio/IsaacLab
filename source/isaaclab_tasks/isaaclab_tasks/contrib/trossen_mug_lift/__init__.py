# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug-lift (handle grasp, classic Franka-cube-lift shaping)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Lift-Mug-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_lift_env_cfg:TrossenMugLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Mug-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_lift_env_cfg:TrossenMugLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Mug-Trossen-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugLiftTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Mug-Trossen-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugLiftDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugLiftDistillationRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Mug-Trossen-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugLiftDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugLiftFinetunePPORunnerCfg",
    },
    disable_env_checker=True,
)
