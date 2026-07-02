# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Standalone timing harness for the adaptive MuJoCo-Warp solver on the Allegro
# in-hand reorient task. Deliberately does NOT use newton-adaptive's scripts.bench.
#
# Measures, over a fixed number of env.step() calls with fixed-seed random actions:
#   * wall time (device-synchronized)
#   * adaptive boundary-loop iterations (solver.cumulative_iterations delta)
#   * derived: ms per env.step, ms per adaptive iteration, iterations per boundary
#
# Usage (from IsaacLab root; see PROTOCOL_5090_adaptive_solver.md for the full run matrix):
#   NEWTON_ADAPTIVE_LOG_EVERY=0 ./isaaclab.sh -p scripts/benchmarks/bench_adaptive_allegro.py \
#       --headless --num_envs 8192 --warmup 64 --steps 256 --label baseline
# Results append as JSON lines to bench_results.jsonl next to this script (untracked).

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

parser = argparse.ArgumentParser(description="Adaptive-solver benchmark (standalone).")
parser.add_argument("--task", type=str, default="Isaac-Reorient-Cube-Allegro-Direct")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--warmup", type=int, default=64)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--label", type=str, default="run")
parser.add_argument(
    "--no_manager_graph",
    action="store_true",
    default=False,
    help="Disable manager-level CUDA graph capture (isolates solver-internal capture tiers).",
)
parser.add_argument(
    "--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.jsonl")
)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = 42
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        # Select the Newton MJWarp backend and force the adaptive (step-doubling) solver.
        # resolve_task_config already collapsed the PresetCfg to its default (PhysX),
        # so grab the task's newton_mjwarp preset from the cfg module directly.
        from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import PhysicsCfg

        env_cfg.sim.physics = PhysicsCfg().newton_mjwarp
        env_cfg.sim.physics.solver_cfg.adaptive = True
        if args_cli.no_manager_graph:
            env_cfg.sim.physics.use_cuda_graph = False

        env = gym.make(args_cli.task, cfg=env_cfg)

        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver = NewtonManager._solver
        solver_name = type(solver).__name__
        print(f"[bench] solver = {solver_name}", flush=True)
        assert solver_name == "SolverMuJoCoAdaptive", f"expected adaptive solver, got {solver_name}"

        env.reset()
        device = env.unwrapped.device
        action_shape = env.action_space.shape

        def run(n_steps: int) -> None:
            with torch.inference_mode():
                for _ in range(n_steps):
                    actions = 2 * torch.rand(action_shape, device=device) - 1
                    env.step(actions)

        # Warmup: JIT compile, CUDA-graph capture, adaptive dt settling.
        run(args_cli.warmup)
        wp.synchronize_device(env_cfg.sim.device)

        iters_before = int(solver.cumulative_iterations.numpy()[0])
        t0 = time.perf_counter()
        run(args_cli.steps)
        wp.synchronize_device(env_cfg.sim.device)
        t1 = time.perf_counter()
        iters_after = int(solver.cumulative_iterations.numpy()[0])

        wall = t1 - t0
        iters = iters_after - iters_before
        n = args_cli.steps
        decimation = int(env_cfg.decimation)
        num_substeps = int(env_cfg.sim.physics.num_substeps)
        boundaries = n * decimation * num_substeps
        status = solver.get_status_summary()

        # MuJoCo constraint-solver convergence stats (sizes the iterations/ls_iterations budget).
        solver_niter = {}
        try:
            niter = solver.mjw_data.solver_niter.numpy()
            solver_niter = {
                "min": int(niter.min()),
                "mean": round(float(niter.mean()), 2),
                "max": int(niter.max()),
            }
        except Exception:
            pass

        result = {
            "label": args_cli.label,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps": n,
            "warmup": args_cli.warmup,
            "decimation": decimation,
            "num_substeps": num_substeps,
            "wall_time_s": round(wall, 4),
            "env_steps_per_s": round(n / wall, 2),
            "adaptive_iterations": iters,
            "mujoco_evals": iters * 3,
            "ms_per_env_step": round(1e3 * wall / n, 3),
            "ms_per_iteration": round(1e3 * wall / max(iters, 1), 4),
            "iters_per_boundary": round(iters / boundaries, 3),
            "status_summary": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in status.items()},
            "solver_niter": solver_niter,
        }
        print("[bench] " + json.dumps(result), flush=True)
        with open(args_cli.out, "a") as f:
            f.write(json.dumps(result) + "\n")

        env.close()


if __name__ == "__main__":
    main()
