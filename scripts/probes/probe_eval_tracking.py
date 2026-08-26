# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-tracking evaluation for the moving-goal slide.

Rolls the deterministic policy on the PLAY variant (canonical pinned
trajectory unless --randomized) and reports the tracking statistics the
online logger cannot: RMSE, 95th-percentile error, band fraction, lag
(the time shift that best aligns object to goal), and overshoot (the
worst along-track lead past the goal). Endpoint arrival alone is NOT
evidence the task is solved; these are the task's numbers.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_eval_tracking.py --checkpoint <model.pt> --episodes 50
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--solver", type=str, default="icf")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--randomized", action="store_true", help="Evaluate on randomized programs instead of the pinned canonical one.")
parser.add_argument("--band", type=float, default=0.025)
parser.add_argument("--max_lag_steps", type=int, default=30)

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

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, agent_cfg = resolve_task_config("IsaacContrib-Slide-Mug-Trossen-Play-v0", "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        apply_solver_choice(env_cfg, args_cli.solver)
        if args_cli.randomized:
            base_cfg, _ = resolve_task_config("IsaacContrib-Slide-Mug-Trossen-v0", "rsl_rl_cfg_entry_point")
            env_cfg.commands = base_cfg.commands
        env = gym.make("IsaacContrib-Slide-Mug-Trossen-Play-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env)

        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=wrapped.device)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=wrapped.device)

        cmd = env.unwrapped.command_manager.get_term("object_pose")
        obj = env.unwrapped.scene["object"]
        origins = env.unwrapped.scene.env_origins

        obs, _ = wrapped.reset()
        errs, obj_tr, goal_tr = [], [], []
        episodes_done = 0
        with torch.inference_mode():
            while episodes_done < args_cli.episodes:
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
                goal_xy = cmd.pose_command_w[:, :2] - origins[:, :2]
                obj_xy = obj.data.root_pos_w.torch[:, :2] - origins[:, :2]
                errs.append(torch.linalg.vector_norm(goal_xy - obj_xy, dim=1).cpu())
                obj_tr.append(obj_xy.cpu())
                goal_tr.append(goal_xy.cpu())
                episodes_done += int(dones.sum().item())

        E = torch.stack(errs)  # [T, N]
        O = torch.stack(obj_tr)  # [T, N, 2]
        G = torch.stack(goal_tr)
        rmse = torch.sqrt((E**2).mean()).item()
        p95 = torch.quantile(E.flatten(), 0.95).item()
        band_frac = (E < args_cli.band).float().mean().item()

        # Lag: the shift k >= 0 minimizing mean |obj(t) - goal(t-k)| — how far
        # behind the program the mug runs. Overshoot: the worst along-track
        # lead of the object past the CURRENT goal.
        best_k, best_v = 0, float("inf")
        for k in range(0, args_cli.max_lag_steps + 1):
            v = torch.linalg.vector_norm(O[k:] - G[: len(G) - k if k else len(G)], dim=-1).mean().item()
            if v < best_v:
                best_k, best_v = k, v
        step_dt = env.unwrapped.step_dt
        dirs = G[1:] - G[:-1]
        dn = torch.linalg.vector_norm(dirs, dim=-1, keepdim=True).clamp(min=1e-9)
        along = ((O[1:] - G[1:]) * (dirs / dn)).sum(dim=-1)
        moving = (dn[..., 0] > 1e-6)
        overshoot = along[moving].max().item() if moving.any() else 0.0

        print("\nTRACKING EVAL")
        print(f"  episodes:   {episodes_done}  ({'randomized' if args_cli.randomized else 'canonical'} program)")
        print(f"  RMSE:       {rmse * 100:.2f} cm")
        print(f"  p95 error:  {p95 * 100:.2f} cm")
        print(f"  band frac:  {band_frac:.3f}  (< {args_cli.band * 100:.1f} cm)")
        print(f"  lag:        {best_k * step_dt * 1000:.0f} ms ({best_k} steps)")
        print(f"  overshoot:  {max(overshoot, 0.0) * 100:.2f} cm")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
