# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the mug flip: the mug lift's runner with its own
experiment tree and a wider exploration band."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


@configclass
class TrossenMugFlipPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "mug_flip"
    experiment_name = "trossen_mug_flip"

    def __post_init__(self):
        super().__post_init__()
        # Between the lift's pinch guard (0.5/1.5 — a handle pinch must
        # survive sampled jitter) and the plate's hot search (1.5/3.0): the
        # flip needs the wrist reorientation DISCOVERED, then a pinch HELD.
        self.actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0, std_type="log", std_range=(0.05, 2.5)
        )
