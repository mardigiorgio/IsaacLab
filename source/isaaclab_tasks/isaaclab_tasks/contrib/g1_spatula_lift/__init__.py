# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 spatula-lift-by-the-handle task at the real lab table (LBM Thimma spatula)."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Lift-Spatula-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_spatula_lift_env_cfg:G1SpatulaLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1SpatulaLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# NOTE: the id deliberately avoids the substring "Lift". scripts/environments/
# teleoperation/teleop_se3_agent.py branches on `if "Lift" in args_cli.task` and
# then dereferences `env_cfg.commands.object_pose`, which only exists on the old
# Isaac-Lift-Cube-Franka style envs. This env has `commands = None` (inherited
# from the G1 IK teleop base), so matching that heuristic crashes at startup.
gym.register(
    id="IsaacContrib-Spatula-Pickup-G1-Teleop-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:G1SpatulaTeleopEnvCfg",
    },
    disable_env_checker=True,
)
