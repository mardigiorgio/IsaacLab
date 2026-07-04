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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

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


def _select_env_state(state: dict, env_ids: torch.Tensor) -> dict:
    """Slice a nested ``scene.get_state()`` dict down to ``env_ids``.

    ``scene.reset_to(..., env_ids=...)`` forwards straight to the ``*_index`` asset
    write methods, which document that they "expect partial data" — i.e. tensors
    already sized ``(len(env_ids), ...)``, matching how recorded mimic episodes are
    stored per-env (see ``RecorderManager.add_to_episodes``). A full-batch state must
    be sliced to the target envs before a partial ``reset_to``.
    """
    return {
        group_name: {
            entity_name: {key: value[env_ids] for key, value in entity_state.items()}
            for entity_name, entity_state in group.items()
        }
        for group_name, group in state.items()
    }


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


ADAPTIVE_MODES = [SOLVER_MODES[1], SOLVER_MODES[3]]  # mujoco_adaptive, sap_adaptive


@pytest.mark.parametrize("solver_cfg_factory", SOLVER_MODES)
def test_staggered_single_env_reset_flags_world_mask(solver_cfg_factory):
    """reset_to on a subset of envs flags exactly those worlds in _world_reset_mask."""
    with build_simulation_context(sim_cfg=_make_sim_cfg(solver_cfg_factory()), auto_add_lighting=False) as sim:
        scene = InteractiveScene(SeamSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
        sim.reset()
        scene.reset()
        _step(sim, scene, 10)
        captured = scene.get_state(is_relative=True)
        _step(sim, scene, 10)

        # partial restore: env 1 only (mimic's staggered-datagen pattern)
        env_ids = torch.tensor([1], dtype=torch.int32, device=sim.device)
        scene.reset_to(_select_env_state(captured, env_ids), env_ids=env_ids, is_relative=True)

        # mask must be flagged for world 1 (and NOT for world 0) before the
        # next step consumes and zeros it
        mask = NewtonManager._world_reset_mask.numpy()
        assert bool(mask[1]), "world 1 not flagged after partial reset_to"
        assert not bool(mask[0]), "world 0 spuriously flagged by a single-env reset_to"

        # stepping consumes the mask and must stay stable
        _step(sim, scene, 5)
        assert not NewtonManager._world_reset_mask.numpy().any(), "mask not consumed by step"


@pytest.mark.parametrize("solver_cfg_factory", ADAPTIVE_MODES)
def test_partial_reset_restores_adaptive_dt_only_for_reset_world(solver_cfg_factory):
    """Adaptive controller dt re-initializes for the reset world and is untouched elsewhere."""
    cfg = solver_cfg_factory()
    dt_init = cfg.adaptive_dt_init
    with build_simulation_context(sim_cfg=_make_sim_cfg(cfg), auto_add_lighting=False) as sim:
        scene = InteractiveScene(SeamSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
        sim.reset()
        scene.reset()

        # let per-world dt evolve away from dt_init (contact-rich settling)
        _step(sim, scene, 30)
        captured = scene.get_state(is_relative=True)
        _step(sim, scene, 10)
        dt_before = NewtonManager._solver.dt.numpy().copy()

        env_ids = torch.tensor([1], dtype=torch.int32, device=sim.device)
        scene.reset_to(_select_env_state(captured, env_ids), env_ids=env_ids, is_relative=True)

        # NewtonManager.step() calls solver.reset(world_mask=...) exactly once, to
        # consume the world-reset mask, before the physics step itself runs. That same
        # step then immediately re-grows dt via step-doubling over the whole control
        # period (see mjwarp_manager._run_solver_substeps), so reading .dt only *after*
        # the step is confounded by that same-step regrowth (measured: dt can regrow
        # from dt_init back to its pre-reset ceiling within that single step, for both
        # solvers). Capture the solver's dt array at the exact moment reset() lands by
        # wrapping the solver instance's own .reset for the duration of this one step.
        solver = NewtonManager._solver
        original_reset = solver.reset
        dt_at_reset: list = []

        def _capturing_reset(state, world_mask=None, flags=None):
            original_reset(state, world_mask=world_mask, flags=flags)
            if not dt_at_reset:
                dt_at_reset.append(solver.dt.numpy().copy())

        solver.reset = _capturing_reset
        try:
            _step(sim, scene, 1)  # step consumes the mask -> solver.reset(world_mask) fires
        finally:
            solver.reset = original_reset

        assert dt_at_reset, "solver.reset() was not invoked while consuming the world-reset mask"
        dt_snapshot = dt_at_reset[0]
        # reset world snapped back to construction default at the moment of reset
        assert dt_snapshot[1] == pytest.approx(dt_init, rel=1e-5), (
            f"world 1 dt {dt_snapshot[1]:.3e} != dt_init {dt_init:.3e} at reset"
        )
        # the other worlds' controller state must be bit-for-bit untouched by env 1's reset
        for w in (0, 2, 3):
            assert dt_snapshot[w] == pytest.approx(dt_before[w], rel=1e-6), (
                f"world {w} dt was spuriously reset by env 1's restore"
            )

        # the controller must keep evolving stably afterward
        dt_after = NewtonManager._solver.dt.numpy()
        assert torch.isfinite(torch.as_tensor(dt_after)).all(), "non-finite dt after partial reset"


def test_rigid_object_only_write_flags_world_mask():
    """A write touching only a rigid object (no articulation) still flags the world mask."""
    with build_simulation_context(sim_cfg=_make_sim_cfg(SOLVER_MODES[1].values[0]()), auto_add_lighting=False) as sim:
        scene = InteractiveScene(SeamSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0))
        sim.reset()
        scene.reset()
        _step(sim, scene, 5)

        cube = scene["cube"]
        env_ids = torch.tensor([2], dtype=torch.int32, device=sim.device)
        pose = cube.data.root_link_pose_w.torch  # write current pose back — content is irrelevant, the flagging is
        cube.write_root_link_pose_to_sim_index(root_pose=pose[env_ids.cpu().numpy().tolist()], env_ids=env_ids)

        mask = NewtonManager._world_reset_mask.numpy()
        assert bool(mask[2]), "rigid-object-only write did not flag its world"
