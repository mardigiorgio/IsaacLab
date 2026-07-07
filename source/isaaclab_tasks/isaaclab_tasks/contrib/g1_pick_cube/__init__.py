# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 pick-cube task at the real lab table (skeleton env; rewards are user-owned)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Pick-Cube-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_pick_cube_env_cfg:G1PickCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PickCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
