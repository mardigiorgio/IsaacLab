# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Measures how often the adaptive MuJoCo-Warp solver runs at its dt_min floor over a
# smoke-test run, for the Allegro reorient and Kuka-Allegro lift tasks.
#
# Reports, over a fixed number of env.step() calls with fixed-seed random actions:
#   * the full log-spaced distribution of the inner timestep, bin 0 being exact-floor hits
#   * saturation depth: the smallest dt the controller ASKED for while pinned at the floor
#   * capped_boundaries: boundaries that consumed the entire max_substeps budget. This
#     includes the case where every world landed exactly on its final permitted iteration
#     with nothing short -- a capped boundary does NOT by itself mean truncation occurred.
#   * unfinished_worlds: worlds that actually ended a boundary short of their target time
#     (sim_time < next_time). This is the true truncation signal.
#
# The truncation counters matter as much as the floor percentage: with dt_outer = 1/120 s,
# dt_min = 1e-6 and max_substeps = 256, crossing a boundary at the floor would need ~8333
# substeps, so the max_substeps budget caps out long before the floor itself binds -- watch
# unfinished_worlds, not just capped_boundaries, to know whether that cap actually truncated
# a world short of its target time.
#
# Usage (from IsaacLab root):
#   uv run python scripts/benchmarks/dt_floor_occupancy.py \
#       --headless --task Isaac-Lift-KukaAllegro --num_envs 1024 --steps 256
# Results append as JSON lines to dt_floor_results.jsonl next to this script (untracked).

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

parser = argparse.ArgumentParser(description="Adaptive-solver dt floor-occupancy probe.")
parser.add_argument("--task", type=str, default="Isaac-Reorient-Cube-Allegro-Direct")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=64)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--label", type=str, default="run")
parser.add_argument(
    "--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dt_floor_results.jsonl")
)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _newton_preset(task: str):
    """Return the task's newton_mjwarp physics preset.

    ``resolve_task_config`` collapses the ``PresetCfg`` to its default (PhysX for both
    of these tasks), so the preset is fetched from the cfg module directly.

    Args:
        task: The task id, e.g. ``Isaac-Reorient-Cube-Allegro-Direct`` or ``Isaac-Lift-KukaAllegro``.
    """
    if "Reorient-Cube-Allegro" in task:
        from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import PhysicsCfg

        return PhysicsCfg().newton_mjwarp
    if "Lift-KukaAllegro" in task:
        from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import KukaAllegroPhysicsCfg

        return KukaAllegroPhysicsCfg().newton_mjwarp
    raise ValueError(f"no newton_mjwarp preset wired for task {task!r}")


def _format_report(rec: dict) -> str:
    """Render one result record as a human-readable report block.

    Args:
        rec: The result record produced in :func:`main`, matching one line of the output
            ``.jsonl`` file.
    """
    lines = [
        f"{rec['task']}   {rec['num_envs']} envs x {rec['steps']} steps   "
        f"dt_min={rec['dt_min']:.0e}  dt_init={rec['dt_init']:.0e}  dt_outer={rec['dt_outer']:.3e}"
    ]
    total = max(rec["total_samples"], 1)
    edges = rec["bin_edges"]
    counts = rec["bin_counts"]
    lines.append(f"  FLOOR (dt == {rec['dt_min']:.0e}){counts[0]:>14,}   {100.0 * counts[0] / total:5.1f}%")
    for i in range(1, len(counts) - 1):
        if counts[i] == 0:
            continue
        lines.append(f"  {edges[i - 1]:.2e} .. {edges[i]:.2e}{counts[i]:>12,}   {100.0 * counts[i] / total:5.1f}%")
    if counts[-1]:
        lines.append(f"  >= {edges[-1]:.2e}{counts[-1]:>20,}   {100.0 * counts[-1] / total:5.1f}%")
    sat = rec["saturation_depth"]
    if sat > 0.0:
        lines.append(f"  saturation depth: min(ideal_dt) = {sat:.2e}  ({rec['dt_min'] / sat:,.0f}x below floor)")
    else:
        lines.append("  saturation depth: floor never reached")
    lines.append(f"  iters/boundary: {rec['iters_per_boundary']:.2f} of max_substeps={rec['max_substeps']}")
    # capped_boundaries hit the max_substeps budget but may still have finished exactly on
    # time; unfinished_worlds is the count that actually landed short of next_time.
    lines.append(
        f"  capped boundaries: {rec['capped_boundaries']:,} / {rec['boundaries']:,}"
        f"      unfinished worlds: {rec['unfinished_worlds']:,}"
    )
    return "\n".join(lines)


def main():
    torch.manual_seed(args_cli.seed)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.sim.physics = _newton_preset(args_cli.task)
        env_cfg.sim.physics.solver_cfg.adaptive = True
        env_cfg.sim.physics.solver_cfg.adaptive_dt_histogram = True

        env = gym.make(args_cli.task, cfg=env_cfg)

        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver = NewtonManager._solver
        solver_name = type(solver).__name__
        assert solver_name == "SolverMuJoCoAdaptive", f"expected adaptive solver, got {solver_name}"
        assert solver.dt_histogram is not None, "dt histogram did not reach the solver"

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

        solver.reset_dt_histogram()
        solver.reset_compute_counter()

        t0 = time.perf_counter()
        run(args_cli.steps)
        wp.synchronize_device(env_cfg.sim.device)
        wall = time.perf_counter() - t0

        stats = solver.dt_histogram_stats()
        counts = solver.dt_histogram.numpy()
        edges = solver.dt_histogram_edges
        iters = int(solver.cumulative_iterations.numpy()[0])
        num_substeps = int(env_cfg.sim.physics.num_substeps)

        rec = {
            "label": args_cli.label,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "warmup": args_cli.warmup,
            "seed": args_cli.seed,
            "decimation": int(env_cfg.decimation),
            "num_substeps": num_substeps,
            # == _solver_dt * _num_substeps, the boundary period NewtonMJWarpManager passes to
            # the adaptive solver's step() (see mjwarp_manager._run_solver_substeps).
            "dt_outer": float(env_cfg.sim.dt),
            "dt_min": float(env_cfg.sim.physics.solver_cfg.adaptive_dt_min),
            "dt_init": float(env_cfg.sim.physics.solver_cfg.adaptive_dt_init),
            "max_substeps": int(env_cfg.sim.physics.solver_cfg.adaptive_max_substeps),
            "bin_edges": [float(x) for x in edges],
            "bin_counts": [int(x) for x in counts],
            "total_samples": int(stats["total_samples"]),
            "floor_samples": int(stats["floor_samples"]),
            "floor_pct": round(100.0 * stats["floor_fraction"], 4),
            "saturation_depth": float(stats["saturation_depth"]),
            "boundaries": int(stats["boundaries"]),
            "capped_boundaries": int(stats["capped_boundaries"]),
            "unfinished_worlds": int(stats["unfinished_worlds"]),
            "adaptive_iterations": iters,
            "iters_per_boundary": round(iters / max(stats["boundaries"], 1), 3),
            "wall_time_s": round(wall, 4),
        }

        print(_format_report(rec), flush=True)
        with open(args_cli.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

        env.close()


if __name__ == "__main__":
    main()
