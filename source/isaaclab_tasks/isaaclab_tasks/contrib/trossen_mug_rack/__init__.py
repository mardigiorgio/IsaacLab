# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen mug-into-dishrack task (TRI MugInDishRack, inverted rest on the wire lattice)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-MugRackPlace-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_rack_env_cfg:TrossenMugRackPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugRackPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-MugRackPlace-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_mug_rack_env_cfg:TrossenMugRackPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenMugRackPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
