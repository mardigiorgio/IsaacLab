# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen plate-and-dishrack tasks (thin-shell stiff contact)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-PlatePick-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_plate_rack_env_cfg:TrossenPlatePickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenPlatePickPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-PlatePick-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_plate_rack_env_cfg:TrossenPlatePickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenPlatePickPPORunnerCfg",
    },
    disable_env_checker=True,
)
