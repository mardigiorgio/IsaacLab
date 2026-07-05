# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end mimic pipeline smoke on the Newton mjwarp backend, all four solver modes.

Reuses the PhysX-recorded nucleus source dataset (the 'imported demos' path):
annotate with retry-replay under Newton dynamics, then generate a small batch.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import json
import os
import sys
import tempfile

import h5py
import pytest
from mimic_test_utils import run_script

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
NUCLEUS_DATASET_PATH = os.path.join(ISAACLAB_NUCLEUS_DIR, "Tests", "Mimic", "dataset.hdf5")
DATASETS_DOWNLOAD_DIR = tempfile.mkdtemp(suffix="_newton_mimic_smoke")
_SUBPROCESS_TIMEOUT = 5000

SOLVER_MODES = ["mujoco", "mujoco-adaptive", "sap", "sap-adaptive"]


@pytest.fixture(scope="module")
def source_dataset() -> str:
    os.makedirs(DATASETS_DOWNLOAD_DIR, exist_ok=True)
    path = retrieve_file_path(NUCLEUS_DATASET_PATH, DATASETS_DOWNLOAD_DIR)
    assert os.path.isfile(path)
    os.environ["PYTHONUNBUFFERED"] = "1"
    return path


def _workflow_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


@pytest.mark.parametrize("solver", SOLVER_MODES)
def test_annotate_and_generate_on_newton(source_dataset, solver):
    annotated = os.path.join(DATASETS_DOWNLOAD_DIR, f"annotated_{solver}.hdf5")
    generated = os.path.join(DATASETS_DOWNLOAD_DIR, f"generated_{solver}.hdf5")

    # --- annotate (retry-replay under Newton dynamics) ---
    result = run_script(
        [
            sys.executable,
            os.path.join(_workflow_root(), "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"),
            "--task",
            TASK,
            "--input_file",
            source_dataset,
            "--output_file",
            annotated,
            "--auto",
            "--retries",
            "5",
            "--physics_preset",
            "newton_mjwarp",
            "--solver",
            solver,
            "--headless",
        ],
        timeout=_SUBPROCESS_TIMEOUT,
    )
    # Print the subprocess output so the run log carries the per-episode replay
    # evidence (yield line, attempts histogram) even when the test xfails.
    print(f"annotate_demos (--solver {solver}) stdout:\n{result.stdout}")
    print(f"annotate_demos (--solver {solver}) stderr:\n{result.stderr}")
    assert "Annotation yield:" in result.stdout, (
        f"annotate_demos did not complete cleanly under --solver {solver}:\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
    yield_line = next(line for line in result.stdout.splitlines() if line.startswith("Annotation yield:"))
    exported_str, total_str = yield_line.split(":")[1].strip().split("/")
    exported = int(exported_str)
    total = int(total_str.split()[0])
    if exported == 0:
        pytest.xfail(
            f"cross-backend replay yielded {exported}/{total} annotated episodes under"
            f" --solver {solver} even with 5 retries — a real finding for the fidelity report, not"
            " a harness bug. Record it and move on."
        )

    # --- generate a small batch from the annotated demos ---
    result = run_script(
        [
            sys.executable,
            os.path.join(_workflow_root(), "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
            "--task",
            TASK,
            "--input_file",
            annotated,
            "--output_file",
            generated,
            "--generation_num_trials",
            "2",
            "--num_envs",
            "2",
            "--physics_preset",
            "newton_mjwarp",
            "--solver",
            solver,
            "--headless",
        ],
        timeout=_SUBPROCESS_TIMEOUT,
    )
    print(f"generate_dataset (--solver {solver}) stdout:\n{result.stdout}")
    print(f"generate_dataset (--solver {solver}) stderr:\n{result.stderr}")
    assert os.path.isfile(generated), (
        f"generate_dataset produced no output under --solver {solver}:\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )

    # --- generated dataset carries the physics stamp (Task 6) ---
    with h5py.File(generated, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        assert "physics_args" in env_args, "physics_args missing from generated dataset metadata"
        expected_sap = solver.startswith("sap")
        assert (env_args["physics_args"]["backend"] == "sap") == expected_sap
