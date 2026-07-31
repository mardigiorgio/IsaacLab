# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner configs for the 29-DoF G1 velocity task.

Uses the rsl-rl >= 4.0 model configs (actor/critic and student/teacher
RslRl*ModelCfg).
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class G1_29_DOFs_RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_29_dofs_rough"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1_29_DOFs_FlatPPORunnerCfg(G1_29_DOFs_RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_29_dofs_flat"
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]


@configclass
class G1_29_DOFs_FlatPPORunnerCfg_DR(G1_29_DOFs_FlatPPORunnerCfg):
    """Teacher runner for the total-DR variant: randomization slows convergence."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000


@configclass
class G1_29_DOFs_VelocityDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "g1_29_dofs_flat"
    run_name = "distillation"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
    )
    # matches the teacher policy network trained by G1_29_DOFs_FlatPPORunnerCfg
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-3,
        loss_type="mse",
    )


###########################
# Standing balance task ###
###########################


@configclass
class G1_29_DOFs_StandPPORunnerCfg(G1_29_DOFs_FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "g1_29_dofs_stand"


@configclass
class G1_29_DOFs_StandDistillationRunnerCfg(G1_29_DOFs_VelocityDistillationRunnerCfg):
    experiment_name = "g1_29_dofs_stand"


#########################
# Student Fine Tuning ###
#########################


@configclass
class G1_29_DOFs_FlatStudentPPORunnerCfg(G1_29_DOFs_FlatPPORunnerCfg):
    # The actor MUST mirror the distillation student architecture exactly so the
    # distilled weights transfer. The transfer itself needs a checkpoint bridge:
    # rsl-rl's PPO.load expects actor_state_dict, while DistillationRunner saves
    # student_state_dict.
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.run_name = "student_finetune"
        # re-assert after the parent post_init, which rewrites hidden_dims to the
        # teacher MLP sizes ([256, 128, 128])
        self.actor.hidden_dims = [256, 256, 128]
        self.critic.hidden_dims = [256, 256, 128]


@configclass
class G1_29_DOFs_StandStudentPPORunnerCfg(G1_29_DOFs_FlatStudentPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.experiment_name = "g1_29_dofs_stand"
