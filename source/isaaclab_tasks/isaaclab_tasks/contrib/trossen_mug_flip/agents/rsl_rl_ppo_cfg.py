# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the mug flip: the mug lift's runner with its own
experiment tree and a wider exploration band."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


@configclass
class TrossenMugFlipPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "mug_flip"
    experiment_name = "trossen_mug_flip"

