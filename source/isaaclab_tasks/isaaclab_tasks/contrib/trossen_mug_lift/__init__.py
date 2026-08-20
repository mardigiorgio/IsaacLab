# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI spatula-lift (blade grasp, classic Franka-cube-lift shaping)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Lift-Spatula-Trossen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_spatula_lift_env_cfg:TrossenSpatulaLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenSpatulaLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Spatula-Trossen-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trossen_spatula_lift_env_cfg:TrossenSpatulaLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrossenSpatulaLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
