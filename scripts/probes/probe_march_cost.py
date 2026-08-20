# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-kernel GPU-time decomposition of the adaptive march (§4.2 cost split).

Rolls a frozen policy to a grasp-heavy state, then records CUDA kernel
activity (Warp's timing capture) over a few control steps and aggregates
elapsed GPU time into the buckets the change decisions need:

  factorize    blocked Cholesky — the share Hessian reuse (CENIC §VI-C) can cut
  linesearch   the exact-root tile kernel
  assembly     gradient / patch-quantity / dof-constraint evaluation
  freemotion   CRB, Jacobians, dynamics matrix, tau, momentum residual
  patches      contact grouping / scatter / finalize
  collide      Newton narrow/broad phase (the re-query cost)
  other        everything else (integrate, commits, controller, copies)

Kernel-time shares are valid with graphs off (launch overhead is host-side
and excluded from CUDA activity timing); absolute wall time is NOT reported —
measure that separately under capture.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_march_cost.py --checkpoint <model.pt> --num_envs 1024 --warm_steps 35 --timed_steps 3 --viz none --out /tmp/march_cost.json
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warm_steps", type=int, default=35, help="Control steps to reach the grasp-heavy regime.")
parser.add_argument("--timed_steps", type=int, default=3, help="Control steps recorded with kernel timing.")
parser.add_argument("--top", type=int, default=25, help="Top-N kernels listed individually.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import warp as wp  # noqa: E402

BUCKETS = (
    ("factorize", ("cholesky", "factor")),
    ("linesearch", ("ls_solve", "linesearch", "ls_dof", "ls_pair")),
    ("assembly", ("assemble_gradient", "eval_patch_quantities", "build_dof_constraints", "reduce_patch", "newton_step", "update_active", "scale_and_tolerance")),
    ("freemotion", ("free_motion", "rigid_tau", "dynamics_matrix", "jacobian", "momentum", "rigid_id", "spatial_velocities", "mass")),
    ("patches", ("patch", "scatter_pairs", "contact_material", "finalize")),
    ("collide", ("broad", "narrow", "bvh", "sap_", "collide", "contact_kernel", "triangle", "mesh", "sdf", "aabb", "midphase")),
    ("integrate", ("integrate", "commit", "adapt_dt", "stage_dt", "error_norm", "seed_half", "march", "open_frame", "fill", "copy", "zero", "reset")),
)


def bucket_of(name: str) -> str:
    low = name.lower()
    for bucket, keys in BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return "other"


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
        apply_solver_choice(env_cfg, "icf-adaptive")
        env_cfg.sim.physics.use_cuda_graph = False  # kernel timing needs live launches
        control_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        env_cfg.episode_length_s = (args_cli.warm_steps + args_cli.timed_steps + 10) / control_hz
        ccfg = getattr(env_cfg.sim.physics, "collision_cfg", None)
        if ccfg is not None:
            scale = max(args_cli.num_envs / 8192.0, 1.0 / 512.0)
            ccfg.rigid_contact_max = max(65536, int(ccfg.rigid_contact_max * scale))
            ccfg.max_triangle_pairs = max(1_000_000, int(ccfg.max_triangle_pairs * scale))

        env = gym.make(args_cli.task, cfg=env_cfg)
        env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env_w, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=str(env.unwrapped.device)
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(env.unwrapped.device))

        obs, _ = env_w.reset()
        with torch.inference_mode():
            for _ in range(args_cli.warm_steps):
                obs, _, _, _ = env_w.step(policy(obs))

            wp.synchronize()
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
            for _ in range(args_cli.timed_steps):
                obs, _, _, _ = env_w.step(policy(obs))
            results = wp.timing_end()

        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver = NewtonManager._solver
        summary = solver.status_summary() if hasattr(solver, "status_summary") else {}
        env.close()

    per_kernel: dict[str, float] = {}
    for r in results:
        per_kernel[r.name] = per_kernel.get(r.name, 0.0) + float(r.elapsed)
    total = sum(per_kernel.values())
    per_bucket: dict[str, float] = {}
    for name, ms in per_kernel.items():
        b = bucket_of(name)
        per_bucket[b] = per_bucket.get(b, 0.0) + ms

    print("\n================ MARCH COST SPLIT (GPU kernel time) ================")
    print(f"  envs {args_cli.num_envs}  warm {args_cli.warm_steps}  timed {args_cli.timed_steps} control steps")
    print(f"  total kernel time: {total:.1f} ms over {len(per_kernel)} distinct kernels")
    for b, ms in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<11} {ms:10.1f} ms   {ms / total * 100:5.1f}%")
    print("  top kernels:")
    for name, ms in sorted(per_kernel.items(), key=lambda kv: -kv[1])[: args_cli.top]:
        print(f"    {ms:9.1f} ms  {ms / total * 100:5.1f}%  [{bucket_of(name):<10}] {name[:90]}")
    print("=====================================================================\n")

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(
                {
                    "num_envs": args_cli.num_envs,
                    "warm_steps": args_cli.warm_steps,
                    "timed_steps": args_cli.timed_steps,
                    "total_ms": total,
                    "buckets_ms": per_bucket,
                    "kernels_ms": dict(sorted(per_kernel.items(), key=lambda kv: -kv[1])),
                    "adaptive": summary,
                },
                f,
                indent=2,
            )
        print(f"[probe] wrote {args_cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
