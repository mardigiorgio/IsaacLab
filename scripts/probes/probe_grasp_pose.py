# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract the grasp-straddle arm pose for the mug-lift reset bank.

Rolls a frozen policy that reliably grasps the mug and records the robot's
joint configuration at every control step where BOTH finger pads press the
mug while it still rests undisturbed at its spawn — the straddle moment the
reset bank should reproduce. Reports the per-joint median over all samples
and the single sample closest to it (a really-visited configuration, unlike
the median vector itself).

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_grasp_pose.py --checkpoint <model.pt> --num_envs 128 --steps 120 --out /tmp/grasp_pose.json
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=120, help="Control steps to roll.")
parser.add_argument("--pad_force_min", type=float, default=1.0, help="Per-pad contact force [N] for a straddle sample.")
parser.add_argument("--rest_z_max", type=float, default=0.05, help="Mug root z [m] below which it counts as at rest.")
parser.add_argument("--xy_drift_max", type=float, default=0.015, help="Mug drift from spawn [m] beyond which the pose is a shove, not a straddle.")
parser.add_argument("--hover_lead", type=int, default=0, help="Sample the pose this many control steps BEFORE first pad contact (0 = at contact).")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

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

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        apply_solver_choice(env_cfg, "icf")
        control_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        env_cfg.episode_length_s = (args_cli.steps + 10) / control_hz

        env = gym.make(args_cli.task, cfg=env_cfg)
        env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env_w, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=str(env.unwrapped.device)
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(env.unwrapped.device))

        scene = env.unwrapped.scene
        robot = scene["robot"]
        obj = scene["object"]
        joint_names = list(robot.joint_names)
        spawn_xy = None
        samples: list[torch.Tensor] = []
        fallback: list[torch.Tensor] = []
        stat_both = stat_rest = stat_undist = 0
        max_pad = 0.0
        lead = max(args_cli.hover_lead, 0)
        history: list[torch.Tensor] = []
        already = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)

        obs, _ = env_w.reset()
        with torch.inference_mode():
            for step in range(args_cli.steps):
                # snapshot BEFORE stepping so history[-(lead+1)] is the pose lead steps back
                history.append(robot.data.joint_pos.torch.clone())
                if len(history) > lead + 1:
                    history.pop(0)
                obs, _, _, _ = env_w.step(policy(obs))
                forces = scene.sensors["pad_object_contact"].data.force_matrix_w
                if forces is None:
                    continue
                per_pad = torch.linalg.vector_norm(forces.torch.sum(dim=2), dim=-1).nan_to_num(0.0)
                straddle = (per_pad > args_cli.pad_force_min).all(dim=1)
                obj_pos = obj.data.root_pos_w.torch - scene.env_origins
                if spawn_xy is None:
                    spawn_xy = obj_pos[:, :2].clone()
                at_rest = obj_pos[:, 2] < args_cli.rest_z_max
                undisturbed = torch.linalg.vector_norm(obj_pos[:, :2] - spawn_xy, dim=1) < args_cli.xy_drift_max
                stat_both += int(straddle.sum())
                stat_rest += int((straddle & at_rest).sum())
                stat_undist += int((straddle & at_rest & undisturbed).sum())
                max_pad = max(max_pad, float(per_pad.max()))
                mask = straddle & at_rest & undisturbed & ~already
                if mask.any() and len(history) > lead:
                    samples.append(history[-(lead + 1)][mask].clone())
                    already |= mask
                # fallback tier: opposed contact on a not-yet-lifted mug, drift ignored
                fb = straddle & (obj_pos[:, 2] < 0.15) & ~already
                if fb.any() and len(history) > lead:
                    fallback.append(history[-(lead + 1)][fb].clone())
        env.close()

    print(
        f"[probe] step-env counts: both-pads>{args_cli.pad_force_min}N: {stat_both}"
        f"  +at-rest: {stat_rest}  +undisturbed: {stat_undist}  max per-pad force {max_pad:.1f} N"
    )
    if not samples and fallback:
        print("[probe] strict tier empty — using fallback tier (opposed contact, mug below 0.15 m, drift ignored).")
        samples = fallback
    if not samples:
        print("[probe] NO straddle samples found — policy never straddled an undisturbed mug.")
        return 1

    import numpy as np

    stack = torch.cat(samples, dim=0).cpu().numpy()
    median = np.median(stack, axis=0)
    best_idx = int(np.argmin(np.linalg.norm(stack - median, axis=1)))
    best = stack[best_idx]
    spread = np.std(stack, axis=0)

    print("\n================ GRASP STRADDLE POSE ================")
    print(f"  samples: {stack.shape[0]} over {args_cli.num_envs} envs x {args_cli.steps} steps")
    for i, name in enumerate(joint_names):
        print(f"  {name:<40} median {median[i]:+8.4f}  best {best[i]:+8.4f}  std {spread[i]:.4f}")
    print("=====================================================\n")

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(
                {
                    "checkpoint": args_cli.checkpoint,
                    "num_samples": int(stack.shape[0]),
                    "joint_names": joint_names,
                    "median": {n: float(v) for n, v in zip(joint_names, median)},
                    "best_sample": {n: float(v) for n, v in zip(joint_names, best)},
                    "std": {n: float(v) for n, v in zip(joint_names, spread)},
                },
                f,
                indent=2,
            )
        print(f"[probe] wrote {args_cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
