# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# A/B the adaptive solver's quantile boundary stop on a REAL IsaacLab task.
#
# Every earlier measurement of ``landed_fraction`` came from standalone Newton scenes, not
# from a task with an actual env, actions, resets and observation pipeline. This runs the
# same task twice -- landed_fraction=1.0 (march until every world lands) versus the
# configured value -- and reports what changed.
#
# Reported per arm:
#   ms per env.step, adaptive iterations per boundary, per-world attempts, straggler waste
#   (loop length / per-world attempts), and the fraction of world-boundaries force-completed.
#
# The correctness gate is that NO world is left short of its boundary: forced completion
# must land every world at the exact right simulation time. A violation is a bug.
#
# Usage (from IsaacLab root):
#   uv run python scripts/benchmarks/bench_landed_fraction.py \
#       --task Isaac-Reorient-Cube-Allegro-Direct --num_envs 1024 --steps 100

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Quantile-stop A/B on a real IsaacLab task.")
parser.add_argument("--task", type=str, default="Isaac-Reorient-Cube-Allegro-Direct")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=24)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fraction", type=float, default=0.95)
parser.add_argument(
    "--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "landed_fraction.jsonl")
)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def newton_preset(task: str):
    """The task's newton_mjwarp preset (resolve_task_config collapses presets to default)."""
    if "Reorient-Cube-Allegro" in task:
        from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import PhysicsCfg

        return PhysicsCfg().newton_mjwarp
    if "Lift-KukaAllegro" in task:
        from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import KukaAllegroPhysicsCfg

        return KukaAllegroPhysicsCfg().newton_mjwarp
    from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: PLC0415

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    preset = getattr(getattr(cfg, "sim", None), "physics", None)
    if preset is None:
        raise ValueError(f"no newton_mjwarp preset wired for task {task!r}")
    return preset


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager

        results = {}
        for frac in (1.0, args_cli.fraction):
            env_cfg.scene.num_envs = args_cli.num_envs
            env_cfg.seed = args_cli.seed
            if args_cli.device is not None:
                env_cfg.sim.device = args_cli.device
            env_cfg.sim.physics = newton_preset(args_cli.task)
            env_cfg.sim.physics.solver_cfg.adaptive = True
            env_cfg.sim.physics.solver_cfg.adaptive_dt_histogram = True
            env_cfg.sim.physics.solver_cfg.adaptive_landed_fraction = frac

            env = gym.make(args_cli.task, cfg=env_cfg)
            solver = NewtonManager._solver
            assert type(solver).__name__ == "SolverMuJoCoAdaptive", type(solver).__name__
            # The whole point of the run: prove the cfg value actually reached the solver.
            got = solver._quantile_stop.landed_fraction
            assert abs(got - frac) < 1e-9, f"cfg landed_fraction={frac} did NOT reach the solver (got {got})"

            env.reset()
            device = env.unwrapped.device
            shape = env.action_space.shape

            def run(n):
                with torch.inference_mode():
                    for _ in range(n):
                        env.step(2 * torch.rand(shape, device=device) - 1)

            run(args_cli.warmup)
            wp.synchronize_device(env_cfg.sim.device)
            solver.reset_dt_histogram()
            solver.reset_compute_counter()

            t0 = time.perf_counter()
            run(args_cli.steps)
            wp.synchronize_device(env_cfg.sim.device)
            wall = time.perf_counter() - t0

            st = solver.dt_histogram_stats()
            iters = int(solver.cumulative_iterations.numpy()[0])
            boundaries = max(st["boundaries"], 1)
            wb = boundaries * args_cli.num_envs
            behind = float((solver.sim_time.numpy() - solver._next_time.numpy()).min())
            # A forced step is unchecked, so it can in principle blow a world up. Landing
            # on time is not enough -- the state has to still be finite.
            import numpy as np  # noqa: PLC0415

            obs = env.unwrapped._get_observations() if hasattr(env.unwrapped, "_get_observations") else None
            finite_state = (
                bool(np.all(np.isfinite(solver.state_0.joint_q.numpy()))) if hasattr(solver, "state_0") else True
            )
            nonfinite_obs = 0
            if isinstance(obs, dict):
                for v in obs.values():
                    if hasattr(v, "isfinite"):
                        nonfinite_obs += int((~v.isfinite()).sum().item())
            loop = iters / boundaries
            per_world = st["total_samples"] / max(wb, 1)
            results[frac] = {
                "landed_fraction": frac,
                "ms_per_env_step": 1e3 * wall / args_cli.steps,
                "iters_per_boundary": loop,
                "per_world_attempts": per_world,
                "straggler_waste": loop / max(per_world, 1e-9),
                "forced_pct": 100.0 * st["unfinished_worlds"] / max(wb, 1),
                "floor_pct": 100.0 * st["floor_fraction"],
                "worst_world_behind_s": behind,
                "state_finite": finite_state,
                "nonfinite_obs": nonfinite_obs,
            }
            env.close()
            del env
            wp.synchronize_device(env_cfg.sim.device)

        base, tuned = results[1.0], results[args_cli.fraction]
        rec = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "speedup": base["ms_per_env_step"] / max(tuned["ms_per_env_step"], 1e-9),
            "arms": {str(k): v for k, v in results.items()},
        }

        print()
        hdr = f"{'landed_fraction':>16}{'ms/env.step':>13}{'it/bnd':>9}{'per-world':>11}{'waste':>8}{'forced%':>9}{'behind[s]':>12}"
        print(f"{args_cli.task}  ({args_cli.num_envs} envs, {args_cli.steps} steps)")
        print(hdr)
        print("-" * len(hdr))
        for frac in (1.0, args_cli.fraction):
            r = results[frac]
            print(
                f"{frac:>16.2f}{r['ms_per_env_step']:>13.2f}{r['iters_per_boundary']:>9.2f}"
                f"{r['per_world_attempts']:>11.2f}{r['straggler_waste']:>8.2f}"
                f"{r['forced_pct']:>8.2f}%{r['worst_world_behind_s']:>12.2e}"
            )
        print(f"\n  speedup: {rec['speedup']:.2f}x")
        bad = [f for f, r in results.items() if r["worst_world_behind_s"] < -1e-6]
        nf = [f for f, r in results.items() if not r["state_finite"] or r["nonfinite_obs"]]
        if bad:
            print(f"  CORRECTNESS FAILURE: world left short of its boundary at {bad}")
        if nf:
            print(f"  CORRECTNESS FAILURE: non-finite state/observations at {nf}")
        if not bad and not nf:
            print("  correctness: every world landed, state and observations finite")

        with open(args_cli.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
