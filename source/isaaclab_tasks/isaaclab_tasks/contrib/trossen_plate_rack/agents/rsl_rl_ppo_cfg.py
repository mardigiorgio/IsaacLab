# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the plate pick: the mug lift's, under the shared
experiment directory (one checkpoint tree for the whole rig family)."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


@configclass
class TrossenPlatePickPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "plate_pick"

    def __post_init__(self):
        super().__post_init__()
        # The inherited init_std 0.5 / cap 1.5 is the mug pinch's jitter
        # guard; the plate's pre-grasp needs a ~90 deg wrist reorientation,
        # which that std cannot reach. Start hot and let PPO taper it; the
        # floor stays for log-prob safety, the cap high enough to not bind.
        self.actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.5, std_type="log", std_range=(0.05, 3.0)
        )
