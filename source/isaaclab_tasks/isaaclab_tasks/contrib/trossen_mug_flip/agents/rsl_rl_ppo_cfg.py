# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the mug flip: the mug lift's runner with its own
experiment tree and a wider exploration band."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg


from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class TrossenMugFlipPPORunnerCfg(TrossenMugLiftPPORunnerCfg):
    run_name = "mug_flip"
    experiment_name = "trossen_mug_flip"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.gamma = 0.99  # the staged economy's discount (hang_fsm_core.GAMMA)
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        # Between the lift's pinch guard and the plate's hot search: the
        # flip needs the wrist reorientation DISCOVERED, then a pinch HELD.
        self.actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
            # init_std 1.0 -> 0.1 (2026-08-28, measured): the handle pinch is a
            # fingertip-edge hold with ~6 mm of margin (bar 8 mm off the wall,
            # fingers 36 mm blocks) and the vendor gripper gain gives ~6 N at the
            # bar. A held bank start survives 30 steps of exploration 65-77% at
            # sigma 0.05, 14-17% at 0.15, ~0% at 0.3 -- at ANY arm scale, gripper
            # scale, gripper gain (10k/50k) or EMA smoothing tested. Wrist joints
            # (scale 1.5) still explore +-0.15 rad at 0.1.
            init_std=0.1, std_type="log", std_range=(0.05, 2.5)
        )

