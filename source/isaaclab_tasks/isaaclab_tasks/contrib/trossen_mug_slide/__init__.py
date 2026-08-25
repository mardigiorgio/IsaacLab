# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI mug SLIDE (push A to B without tipping)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_slide_env_cfg:TrossenMugSlideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_slide_env_cfg:TrossenMugSlideEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_slide_moving_cfg:TrossenMugSlideMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_slide_moving_cfg:TrossenMugSlideMovingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugSlideTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlidePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugSlideDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlideDistillationRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Slide-Mug-Trossen-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_sim2real_cfg:TrossenMugSlideDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugSlideFinetunePPORunnerCfg",
    },
    disable_env_checker=True,
)
