# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class G1SpatulaLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the G1 spatula-lift-by-the-handle task."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "g1_spatula_lift"
    logger = "wandb"
    wandb_project = "g1-spatula-lift"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # init 0.4 inside a HARD (0.15, 0.5) range. The FLOOR fixes the collapse
        # measured in run 2026-08-01_18-03-39, where per-dim stds fell to ~0.075
        # the moment reach converged and finger exploration stopped. The CAP is
        # the new half: with entropy_coef 0 removed as a counterweight, std
        # ballooned to 0.98 and the resulting flail drove the arm into the blade
        # (94% blade_contact terminations). rsl_rl 5.4.1 clamps std_param to this
        # range on every update, so both bounds are enforced by the library.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4, std_range=(0.15, 0.5)),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.0: std-ballooning reproduced at both 0.005 and 0.002. The std_range
        # floor now owns exploration, so the entropy bonus is redundant pressure
        # in the one direction that broke the last run
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        # the adaptive KL controller moves from here; 1e-3 was too hot a start
        # for a reward that now pays continuously from step 1
        learning_rate=1.0e-4,
        schedule="adaptive",
        # ~50-step effective horizon at 60 Hz control: the carry-point income is
        # dense and immediate, so a 100-step horizon just adds variance
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
