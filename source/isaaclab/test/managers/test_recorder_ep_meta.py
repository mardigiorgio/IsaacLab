# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that episode metadata stamps the physics/solver configuration."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from types import SimpleNamespace

from isaaclab.managers.recorder_manager import RecorderManager


def _manager_with_cfg(physics) -> RecorderManager:
    """Build an uninitialized RecorderManager around a stub env (get_ep_meta only reads env.cfg)."""
    manager = RecorderManager.__new__(RecorderManager)
    manager._env = SimpleNamespace(
        cfg=SimpleNamespace(
            sim=SimpleNamespace(dt=0.01, physics=physics, render_interval=2),
            decimation=5,
            scene=SimpleNamespace(num_envs=4),
        )
    )
    return manager


def test_ep_meta_stamps_newton_solver_config():
    solver_cfg = SimpleNamespace(
        solver_type="mujoco_warp", backend="sap", adaptive=False, sap_adaptive=True, adaptive_tol=1e-3
    )
    physics = SimpleNamespace(solver_cfg=solver_cfg)
    ep_meta = _manager_with_cfg(physics).get_ep_meta()
    pa = ep_meta["physics_args"]
    assert pa["backend"] == "sap"
    assert pa["sap_adaptive"] is True
    assert pa["adaptive"] is False
    assert pa["solver_type"] == "mujoco_warp"
    assert pa["sim_dt"] == 0.01


def test_ep_meta_handles_physx_without_solver_cfg():
    physics = SimpleNamespace()  # PhysX-like: no solver_cfg attribute
    ep_meta = _manager_with_cfg(physics).get_ep_meta()
    pa = ep_meta["physics_args"]
    assert pa["physics_cfg"] == "SimpleNamespace"
    assert "backend" not in pa
