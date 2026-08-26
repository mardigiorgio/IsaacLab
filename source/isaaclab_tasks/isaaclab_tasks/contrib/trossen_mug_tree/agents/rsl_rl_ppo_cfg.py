# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the mug hang: the mug lift's, under the shared experiment directory."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


@configclass
class TrossenMugHangPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "mug_hang"

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Longer credit horizon for the place->release->retreat chain (the
        # completion sits ~100 steps past the grasp), and normalized obs for
        # both networks: the reward refactor changes return scales anyway.
        self.algorithm.gamma = 0.99
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
