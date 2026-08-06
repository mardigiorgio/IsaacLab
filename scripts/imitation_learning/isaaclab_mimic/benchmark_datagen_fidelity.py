# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fidelity/throughput benchmark for mimic dataset generation across solver modes.

For each requested solver mode: run generate_dataset.py headless, time it,
parse the success counter, then post-process the generated HDF5 for contact
fidelity (cube-table penetration). Emits one markdown table — the
fidelity-vs-throughput tradeoff the adaptive solvers exist to win.

The full fidelity table requires source demos that succeed under the target backend.
With the stack tasks' stabilized Newton contact preset and the boundingCube collider fix,
cross-backend (PhysX-annotated) sources accept demos at measured rates of ~24% on
--solver mujoco and ~19% on mujoco-adaptive (PhysX ~50%); the SAP modes measured 0/25
(SolverSAP applies its own contact compliance and ignores the preset's ke/kd -> solref
stiffness). Newton-native recorded demos (see scripts/tools/record_demos.py
--record_subtask_signals) remain the path to parity.

Run (from the repo root; needs a GPU and an annotated source dataset):
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/benchmark_datagen_fidelity.py \\
        --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \\
        --input_file datasets/annotated_dataset.hdf5 \\
        --num_trials 100 --num_envs 8 \\
        --solvers mujoco mujoco-adaptive sap sap-adaptive
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np

# Block half-height of the stack task's DexCube assets [m]: init z = 0.0203
# with the cube resting on the table plane (z=0).
CUBE_HALF_HEIGHT = 0.0203
CUBE_NAMES = ("cube_1", "cube_2", "cube_3")


def run_generation(solver: str, out_file: str, args) -> tuple[float, int, int]:
    """Run generate_dataset.py for one solver mode. Returns (wall_s, successes, attempts)."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_dataset.py"),
        "--task",
        args.task,
        "--input_file",
        args.input_file,
        "--output_file",
        out_file,
        "--generation_num_trials",
        str(args.num_trials),
        "--num_envs",
        str(args.num_envs),
        "--physics_preset",
        args.physics_preset,
        "--solver",
        solver,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    # last "N/M (x%) successful demos generated" line from env_loop
    matches = re.findall(r"(\d+)/(\d+) \([\d.]+%\) successful demos generated", proc.stdout)
    if not matches:
        raise RuntimeError(
            f"[{solver}] no success counter in generate_dataset output (rc={proc.returncode}).\n"
            f"stdout tail: {proc.stdout[-3000:]}\nstderr tail: {proc.stderr[-3000:]}"
        )
    successes, attempts = (int(x) for x in matches[-1])
    return wall, successes, attempts


def penetration_stats(dataset_path: str) -> dict[str, float]:
    """Cube-table penetration [m] over every recorded state of every episode."""
    pens = []
    with h5py.File(dataset_path, "r") as f:
        for demo in f["data"].values():
            # per-step scene states recorded by the ActionState recorder
            for cube in CUBE_NAMES:
                key = f"states/rigid_object/{cube}/root_pose"
                if key not in demo:
                    continue
                z = np.asarray(demo[key])[:, 2]
                pen = np.clip(CUBE_HALF_HEIGHT - z, 0.0, None)
                pens.append(pen)
    if not pens:
        return {"p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    pens = np.concatenate(pens)
    return {
        "p50": float(np.percentile(pens, 50)),
        "p95": float(np.percentile(pens, 95)),
        "max": float(pens.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True, help="Annotated source dataset.")
    parser.add_argument("--num_trials", type=int, default=100, help="Accepted demos per solver mode.")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--solvers", nargs="+", default=["mujoco", "mujoco-adaptive", "sap", "sap-adaptive"])
    parser.add_argument("--physics_preset", type=str, default="newton_mjwarp")
    parser.add_argument("--output", type=str, default="datagen_fidelity_results.md")
    parser.add_argument("--workdir", type=str, default=None, help="Where generated datasets land (default: tmpdir).")
    args = parser.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(suffix="_datagen_fidelity")
    os.makedirs(workdir, exist_ok=True)
    rows = []
    for solver in args.solvers:
        out_file = os.path.join(workdir, f"generated_{solver}.hdf5")
        print(f"=== {solver}: generating {args.num_trials} demos ===", flush=True)
        wall, successes, attempts = run_generation(solver, out_file, args)
        stats = penetration_stats(out_file)
        rows.append(
            {
                "solver": solver,
                "wall_s_per_demo": wall / max(successes, 1),
                "success_rate": successes / max(attempts, 1),
                **stats,
            }
        )
        print(json.dumps(rows[-1], indent=2), flush=True)

    header = "| solver | s/accepted demo | success rate | pen p50 [m] | pen p95 [m] | pen max [m] |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep] + [
        f"| {r['solver']} | {r['wall_s_per_demo']:.1f} | {r['success_rate']:.2%}"
        f" | {r['p50']:.2e} | {r['p95']:.2e} | {r['max']:.2e} |"
        for r in rows
    ]
    table = "\n".join(lines)
    print("\n" + table)
    with open(args.output, "w") as f:
        f.write(f"# Mimic datagen fidelity/throughput ({args.task})\n\n{table}\n")
    print(f"\nWrote {args.output}; datasets in {workdir}")


if __name__ == "__main__":
    main()
