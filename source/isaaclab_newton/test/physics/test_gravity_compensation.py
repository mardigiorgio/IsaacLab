# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for per-body ``disable_gravity`` on the Newton backend.

``FRANKA_PANDA_HIGH_PD_CFG`` sets ``spawn.rigid_props.disable_gravity = True`` on every link and
relies on a gravity-free arm for its 400/80 PD tracking. Newton's USD importer only reads
``physxRigidBody:disableGravity`` at the physics-scene level (a scene-wide gravity toggle), so
per-body intent was previously dropped: holding the arm at its default pose with no further
actions let gravity sag it under every MuJoCo-Warp mode. See
``.superpowers/sdd/task-10b-report.md`` (Gap 1) and ``.superpowers/sdd/task-13-brief.md``.

The fix maps ``disable_gravity`` bodies onto MuJoCo-Warp's per-body ``gravcomp`` custom
attribute (``mjwarp_manager._apply_gravity_compensation``), which both the fixed-step and
adaptive MuJoCo solvers honor. The SAP backend has no per-body gravity mechanism (gravity is
applied per-world only), so it can only warn -- see ``MJWarpSolverCfg.backend``'s docstring.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import logging

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


SIM_DT = 1.0 / 120.0
NUM_ENVS = 2
N_STEPS = 100
DROOP_TOLERANCE_CM = 1.0


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """A lone gravity-compensated Franka: the minimal scene that reproduces Gap 1."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# mujoco fixed + mujoco adaptive per the brief. adaptive_dt_init sits below sim.dt so the
# controller has room to subdivide; adaptive_dt_min < adaptive_dt_init per the adaptive rule.
MUJOCO_MODES = [
    pytest.param(
        lambda: MJWarpSolverCfg(njmax=200, ls_iterations=20, cone="elliptic", integrator="implicitfast"),
        id="mujoco_fixed",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(
            njmax=200,
            ls_iterations=20,
            cone="elliptic",
            integrator="implicitfast",
            adaptive=True,
            adaptive_tol=1e-3,
            adaptive_dt_init=2e-3,
            adaptive_dt_min=1e-6,
        ),
        id="mujoco_adaptive",
    ),
]


def _make_sim_cfg(solver_cfg: MJWarpSolverCfg) -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=2, use_cuda_graph=False),
    )


def _hold_at_default_pose_and_measure_droop(sim, scene) -> torch.Tensor:
    """Teleport the robot to its default joint pose, hold that target with zero further action
    writes for ``N_STEPS``, and return the per-env EEF (``panda_hand``) world-z droop [cm]
    (positive = sagged downward).

    ``scene.reset()`` alone does not teleport joints to the configured default pose in a
    standalone (non-env) harness -- the manager-based env's reset path that normally does this
    is not invoked here (see ``_setup_franka_at_home_pose`` in
    ``isaaclab_newton/test/assets/test_articulation.py``) -- so this does it explicitly once,
    then holds the target constant ("zero actions" per the task brief).
    """
    robot = scene["robot"]
    default_pos = robot.data.default_joint_pos.torch.clone()
    default_vel = robot.data.default_joint_vel.torch.clone()
    robot.write_joint_state_to_sim_index(position=default_pos, velocity=default_vel)
    robot.set_joint_position_target_index(target=default_pos)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt=SIM_DT)

    ee_idx = robot.find_bodies("panda_hand")[0][0]
    z0 = robot.data.body_pos_w.torch[:, ee_idx, 2].clone()

    for _ in range(N_STEPS):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=SIM_DT)

    z1 = robot.data.body_pos_w.torch[:, ee_idx, 2].clone()
    return (z0 - z1) * 100.0


@pytest.mark.parametrize("solver_cfg_factory", MUJOCO_MODES)
def test_disable_gravity_honored_on_mujoco(solver_cfg_factory):
    """A gravity-free Franka held at its default pose must not sag on either MuJoCo-Warp mode."""
    with build_simulation_context(sim_cfg=_make_sim_cfg(solver_cfg_factory()), auto_add_lighting=False) as sim:
        scene = InteractiveScene(FrankaSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
        sim.reset()
        scene.reset()

        droop_cm = _hold_at_default_pose_and_measure_droop(sim, scene)

        assert torch.all(droop_cm.abs() < DROOP_TOLERANCE_CM), (
            f"EEF world-z droop {droop_cm.tolist()} cm exceeds the {DROOP_TOLERANCE_CM} cm "
            "tolerance -- disable_gravity is not being honored (gravcomp not applied)."
        )


def test_disable_gravity_warns_on_sap(caplog):
    """SAP has no per-body gravity-compensation mechanism; it must warn instead of silently
    sagging the arm (see the ``MJWarpSolverCfg.backend`` docstring for the documented gap)."""
    solver_cfg = MJWarpSolverCfg(backend="sap", sap_solver_iterations=30)
    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.mjwarp_manager"):
        with build_simulation_context(sim_cfg=_make_sim_cfg(solver_cfg), auto_add_lighting=False) as sim:
            scene = InteractiveScene(FrankaSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
            sim.reset()
            scene.reset()

    assert "disable_gravity" in caplog.text and "SAP" in caplog.text, (
        "expected an actionable disable_gravity/SAP warning to be logged when a scene with "
        f"disable_gravity bodies runs on the SAP backend; captured log:\n{caplog.text}"
    )
