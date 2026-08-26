# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen mug-on-mug-tree task (TRI hang_mugs_on_mug_holder, single arm)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-MugHang-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_tree_env_cfg:TrossenMugHangEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugHangPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-MugHang-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_tree_env_cfg:TrossenMugHangEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugHangPPORunnerCfg",
    },
    disable_env_checker=True,
)
