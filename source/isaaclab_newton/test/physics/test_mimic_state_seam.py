# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the mimic state-restore seam on the Newton backend.

IsaacMimic drives physics exclusively through ``scene.get_state()`` /
``scene.reset_to()`` (mid-episode, per-env, staggered). These tests exercise
that exact pattern against all four mjwarp solver modes.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


# The four solver modes from the spec. Factories (not instances) so each test
# gets a fresh cfg. adaptive_dt_init sits below the solver dt so the
# controller has room to subdivide; dt_min < dt_init per the adaptive rule.
SOLVER_MODES = [
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
    pytest.param(
        lambda: MJWarpSolverCfg(backend="sap", sap_solver_iterations=30),
        id="sap_fixed",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(backend="sap", sap_adaptive=True, sap_solver_iterations=30),
        id="sap_adaptive",
    ),
]

NUM_ENVS = 4
SIM_DT = 1.0 / 120.0


@configclass
class SeamSceneCfg(InteractiveSceneCfg):
    """Franka + free cube per env: the minimal mimic-shaped scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )


def _make_sim_cfg(solver_cfg: MJWarpSolverCfg, device: str = "cuda:0") -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=2, use_cuda_graph=False),
    )


def _step(sim, scene, n: int = 1) -> None:
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=SIM_DT)


def _assert_state_allclose(expected: dict, actual: dict, atol: float = 1e-5) -> None:
    """Recursively compare two scene-state dicts (articulation/rigid_object trees)."""
    for group_name, group in expected.items():
        for entity_name, entity_state in group.items():
            for key, value in entity_state.items():
                other = actual[group_name][entity_name][key]
                torch.testing.assert_close(
                    torch.as_tensor(other),
                    torch.as_tensor(value),
                    atol=atol,
                    rtol=0.0,
                    msg=f"{group_name}/{entity_name}/{key} mismatch after reset_to",
                )


@pytest.mark.parametrize("solver_cfg_factory", SOLVER_MODES)
def test_state_round_trip_mid_episode(solver_cfg_factory):
    """get_state -> step N -> reset_to(state) restores the written state and the sim keeps stepping stably."""
    with build_simulation_context(sim_cfg=_make_sim_cfg(solver_cfg_factory()), auto_add_lighting=False) as sim:
        scene = InteractiveScene(SeamSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
        sim.reset()
        scene.reset()

        # settle, then capture a mid-episode state
        _step(sim, scene, 10)
        captured = scene.get_state(is_relative=True)

        # keep evolving well past the captured state
        _step(sim, scene, 20)

        # mimic's restore: mid-episode, all envs
        scene.reset_to(captured, env_ids=None, is_relative=True)
        restored = scene.get_state(is_relative=True)
        _assert_state_allclose(captured, restored)

        # the sim must continue stepping stably from the restored state
        _step(sim, scene, 10)
        final = scene.get_state(is_relative=True)
        for group in final.values():
            for entity_state in group.values():
                for key, value in entity_state.items():
                    assert torch.isfinite(torch.as_tensor(value)).all(), f"non-finite {key} after restore+step"
