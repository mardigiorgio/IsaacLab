# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner for the Trossen mug lift.

The Stationary AI cube task's rig-validated recipe (which itself mirrors the reference
Franka lift PPO), with the reference single-observation-group layout: no privileged /
teacher split -- the policy sees proprioception + object position directly, exactly as
the Franka cube lift does.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TrossenMugLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # NaN observations are the object of study here, not a bug to abort on:
    # a fixed step that cannot resolve this contact drives joint state
    # non-finite, and the run has to survive that to record what it looks like.
    check_for_nan = False
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "trossen_mug_lift"

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        # std floor keeps the gripper dim from collapsing sigma to zero
        # (log-prob blowup); the cap bounds exploration drift, which is
        # otherwise unbounded and runs away.
        # init_std 0.5 (was 1.0): a held pinch survives only while sampled
        # jitter stays inside the clamp's seat; at std 1 the seat breaks in
        # 1-2 steps and the grasp income stream cannot exist to be learned.
        # The slide keeps its trained recipe in its own package.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="log", std_range=(0.05, 1.5)),
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

