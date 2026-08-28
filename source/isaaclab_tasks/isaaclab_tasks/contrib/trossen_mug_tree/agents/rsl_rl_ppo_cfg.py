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
        # Std floor 0.05 (2026-08-28; the 0.3 floor was a discovery crutch
        # from the pre-ratchet economy and became a PERMANENT jitter source:
        # a converged policy still sampled +/-0.3 rad on the wrist every step
        # and never looked settled). Hot start (init 1.5) explores; once the
        # hang is learned sigma is free to collapse to 0.05.
        self.actor.distribution_cfg.std_range = (0.05, 3.0)
        # entropy_coef 0.006 -> 0.0 (2026-08-28): with init_std 1.5 the bonus held
        # sigma at ~1.5 for 1000 iterations (entropy rising 13.8 -> 14.0) while
        # the MEAN policy placed the mug 60-86% of the time and the sampled
        # rollouts ~0.6%. The std floor (0.3) is the exploration guarantee; the
        # bonus only fought the anneal the pinch needs.
        self.algorithm.entropy_coef = 0.0
