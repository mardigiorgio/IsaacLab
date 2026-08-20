# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the rim-reach term's geometry without recomputing it.

Recovers the underlying TCP-to-rim distance from the shaped reward by kernel
inversion at two different stds (both must recover the same distance, pinning
the kernel wiring), then checks it against invariants derived from positions
read independently of the term:

  * rim distance < TCP-to-root distance for the home-pose TCP (the rim is
    nearer than the buried bottom-center),
  * their gap is at most |root -> any rim point| = sqrt(H^2 + R^2), the
    triangle-inequality bound,
  * the shaped value lies in (0, 1) and is finite.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_rim_reach.py --num_envs 16
"""

from __future__ import annotations

import argparse
import math
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--num_envs", type=int, default=16)
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

    from isaaclab_tasks.contrib.trossen_mug_lift import mdp
    from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import MUG_RIM_HEIGHT, MUG_RIM_RADIUS
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        apply_solver_choice(env_cfg, "icf")
        env_cfg.observations.policy.enable_corruption = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        obj = scene["object"]
        env.reset()
        zero = torch.zeros(args_cli.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(2):
                env.step(zero)
            u = env.unwrapped
            s1 = mdp.mug_rim_ee_distance(u, std=0.1, rim_height=MUG_RIM_HEIGHT, rim_radius=MUG_RIM_RADIUS)
            s2 = mdp.mug_rim_ee_distance(u, std=0.2, rim_height=MUG_RIM_HEIGHT, rim_radius=MUG_RIM_RADIUS)
            d1 = 0.1 * torch.atanh((1.0 - s1).clamp(max=1 - 1e-7))
            d2 = 0.2 * torch.atanh((1.0 - s2).clamp(max=1 - 1e-7))
            tcp = scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
            droot = torch.linalg.vector_norm(tcp - obj.data.root_pos_w.torch, dim=-1)
        env.close()

    bound = math.sqrt(MUG_RIM_HEIGHT**2 + MUG_RIM_RADIUS**2) + 1e-3
    gap = droot - d1
    ok_kernel = bool(torch.allclose(d1, d2, atol=1e-4))
    ok_range = bool(((s1 > 0) & (s1 < 1) & torch.isfinite(s1)).all())
    ok_nearer = bool((gap > 0).all())
    ok_bound = bool((gap <= bound).all())
    print(f"[probe] rim distance (via kernel inversion): mean {d1.mean() * 1000:6.1f} mm")
    print(f"[probe] root distance:                       mean {droot.mean() * 1000:6.1f} mm")
    print(f"[probe] gap root-rim: min {gap.min() * 1000:6.1f} mm  max {gap.max() * 1000:6.1f} mm  bound {bound * 1000:6.1f} mm")
    print(f"[probe] kernel-inversion agreement: {ok_kernel}   value in (0,1) finite: {ok_range}")
    print(f"[probe] rim nearer than root: {ok_nearer}   gap within triangle bound: {ok_bound}")
    if not (ok_kernel and ok_range and ok_nearer and ok_bound):
        print("[probe] FAIL")
        return 1
    print("[probe] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
