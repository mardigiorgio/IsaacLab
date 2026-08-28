# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the mug-rack place: the hang's (gamma 0.99, obs norm, std floor 0.05, no entropy bonus)."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_tree.agents.rsl_rl_ppo_cfg import TrossenMugHangPPORunnerCfg


@configclass
class TrossenMugRackPlacePPORunnerCfg(TrossenMugHangPPORunnerCfg):
    run_name = "mug_rack_place"
