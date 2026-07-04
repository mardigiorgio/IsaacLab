# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test applying physics presets and solver choices to resolved env configs."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
from isaaclab_newton.physics import NewtonCfg

import isaaclab_tasks  # noqa: F401  # registers tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import (
    PHYSICS_SOLVER_CHOICES,
    apply_physics_preset,
    apply_solver_choice,
)

# The mimic-free stack task id lives in isaaclab_tasks.contrib and does not
# depend on isaaclab_mimic (verified via the registered task ids in
# source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/__init__.py).
TASK = "IsaacContrib-Stack-Cube-Franka-IK-Rel"


def test_apply_physics_preset_swaps_to_newton():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=3)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.scene.num_envs == 3, "num_envs clobbered by preset application"
    assert env_cfg.sim.device == "cuda:0", "device clobbered by preset application"


@pytest.mark.parametrize("solver", list(PHYSICS_SOLVER_CHOICES))
def test_apply_solver_choice_sets_latches(solver):
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    apply_solver_choice(env_cfg, solver)
    solver_cfg = env_cfg.sim.physics.solver_cfg
    for attr, value in PHYSICS_SOLVER_CHOICES[solver].items():
        assert getattr(solver_cfg, attr) == value, f"{attr} not latched for --solver {solver}"


def test_apply_solver_choice_requires_newton_physics():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)  # default = PhysX
    with pytest.raises(ValueError, match="solver_cfg"):
        apply_solver_choice(env_cfg, "mujoco-adaptive")


def test_apply_physics_preset_rejects_unknown_preset():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    with pytest.raises(ValueError, match="nonexistent_preset"):
        apply_physics_preset(env_cfg, TASK, "nonexistent_preset")
