# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the plate pick: the mug lift's, under the shared
experiment directory (one checkpoint tree for the whole rig family)."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


@configclass
class TrossenPlatePickPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "plate_pick"
