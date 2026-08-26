# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Post-hoc success-rate evaluation for a trained checkpoint.

One evaluator, every arm of every pair: loads a checkpoint, rolls the
deterministic policy for full episodes, and scores an episode a SUCCESS when
the object sits within a positional tolerance of its commanded goal,
upright, and stays there over the final hold window -- delivery, not a
drive-by. The criterion is task-agnostic across the Trossen scenes because
all of them command an object pose and all define upright the same way.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_eval_success.py --task IsaacContrib-Slide-Mug-Trossen-v0 --checkpoint <model.pt> --episodes 512
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--solver", type=str, required=True, help="icf or icf-adaptive; match the training run.")
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--pos_tol", type=float, default=0.05, help="Planar distance to the commanded goal [m].")
parser.add_argument("--min_up_cos", type=float, default=0.87, help="Upright gate, cos of tilt angle.")
parser.add_argument("--hold_steps", type=int, default=30, help="Final window the success state must hold.")
parser.add_argument("--dt_override", type=float, default=None, help="env.sim.dt for K1/K2 arms.")
parser.add_argument("--decimation_override", type=int, default=None)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> int:
    import gymnasium as gym
    import torch

    from isaaclab.utils.math import combine_frame_transforms
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        apply_solver_choice(env_cfg, args_cli.solver)
        if args_cli.dt_override is not None:
            env_cfg.sim.dt = args_cli.dt_override
        if args_cli.decimation_override is not None:
            env_cfg.decimation = args_cli.decimation_override
        env_cfg.observations.policy.enable_corruption = False
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env_w = RslRlVecEnvWrapper(env)

        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env_w, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=u.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=u.device)

        robot = u.scene["robot"]
        obj = u.scene["object"]
        horizon = int(u.max_episode_length)

        def success_now() -> torch.Tensor:
            command = u.command_manager.get_command("object_pose")
            des_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
            )
            dist = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
            quat = obj.data.root_quat_w.torch
            up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
            return (dist < args_cli.pos_tol) & (up_z > args_cli.min_up_cos)

        succ_total = 0
        div_total = 0
        epi_total = 0
        obs, _ = env_w.reset()
        with torch.inference_mode():
            while epi_total < args_cli.episodes:
                # Roll one full synchronized episode; the hold window is the
                # last hold_steps of the horizon.
                held = torch.ones(args_cli.num_envs, dtype=torch.bool, device=u.device)
                for k in range(horizon - 1):
                    obs, _, _, _ = env_w.step(policy(obs))
                    if k >= horizon - 1 - args_cli.hold_steps:
                        held &= success_now()
                    div_total += int(u.termination_manager.get_term("physics_diverged").sum().item())
                succ_total += int(held.sum())
                epi_total += args_cli.num_envs
                obs, _, _, _ = env_w.step(policy(obs))
        print(
            f"[success] {succ_total}/{epi_total} = {succ_total / max(epi_total, 1):.3f}"
            f"  (pos_tol {args_cli.pos_tol} m, upright cos {args_cli.min_up_cos},"
            f" hold {args_cli.hold_steps} steps; diverged episodes {div_total})"
        )
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
