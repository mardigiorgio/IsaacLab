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
    max_iterations = 5000
    save_interval = 100
    experiment_name = "g1_spatula_lift"
    logger = "wandb"
    wandb_project = "g1-spatula-lift"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # 0.5: full-std relative-action exploration on 15 joints flails hard
        # enough to tip the free-standing base and knock the spatula away
        # before any task signal is collected
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # 0.99 like every proven lift here: with continuous hold income and
        # no terminal bonus, dense flow needs no long credit horizon
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
