# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner for the Trossen mug slide, pinned verbatim to the slidev1 recipe.

The trained K-ladder (slidev1 K1/K2/K3 and the slide teacher) ran exactly
this configuration; every future twin — including the adaptive-solver arm of
the comparison — must match it. The lift task tunes its own exploration
separately in its own package.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TrossenMugSlidePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # NaN observations are the object of study here, not a bug to abort on:
    # a fixed step that cannot resolve this contact drives joint state
    # non-finite, and the run has to survive that to record what it looks like.
    check_for_nan = False
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    # Historical continuity: every slidev1 checkpoint lives under this
    # experiment directory; renaming would orphan the trained ladder.
    experiment_name = "trossen_mug_lift"

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        # std floor keeps the effectively-binary gripper dim from collapsing
        # sigma to zero (log-prob blowup); the cap bounds exploration drift,
        # which is otherwise unbounded and runs away.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log", std_range=(0.05, 3.0)),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
