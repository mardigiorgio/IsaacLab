# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

"""
Regression tests for NewtonCfg.enforce_velocity_limit (task 14).

SolverMuJoCo does not consume ``joint_velocity_limit`` (its docstring says so explicitly) and
the vendored SAP solver has no velocity-limit concept at all, so a stiff implicit-PD joint
commanded a large position step can move several times faster than its configured drive
velocity limit -- PhysX enforces this limit natively; Newton's solvers do not. These tests drive
a single-revolute-joint articulation (mirrors ``generate_articulation_cfg("single_joint_implicit",
...)`` in ``test_articulation.py``) directly (no env, no decimation) and verify the
:meth:`~isaaclab_newton.assets.Articulation.write_data_to_sim` target-rate clamp, run once per
physics tick, bounds the measured per-tick displacement to ``velocity_limit * dt``, and leaves
joints without a real (sub-sentinel) configured limit completely unaffected.
"""

DT = 1.0 / 120.0
"""Physics timestep [s]. This test calls write_data_to_sim + sim.step directly once per tick (no
env, decimation=1), matching the cadence :meth:`Articulation._clamp_joint_position_target_rate`
assumes: one commanded-target change per call, so ``dt`` here is exactly the interval the clamp
rate-limits against.
"""

STIFFNESS = 2000.0
DAMPING = 100.0
TOLERANCE = 1.05
"""Allowed overshoot above ``velocity_limit * dt`` per tick (5%), covering the stiff-PD's
transient approach to the ramping (clamped) target before it settles into steady-state tracking
one dt-sized step behind it."""


def _solver_cfg(backend: str) -> MJWarpSolverCfg:
    """Fixed-step MJWarpSolverCfg for the given inner backend ("mujoco" or "sap")."""
    if backend == "mujoco":
        return MJWarpSolverCfg(
            njmax=20,
            nconmax=20,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        )
    if backend == "sap":
        return MJWarpSolverCfg(backend="sap")
    raise ValueError(f"Unknown backend: {backend}")


def _fix_reversed_joints(stage) -> None:
    """Swap body0/body1 on ``revolute_articulation.usd``'s joint (known reversed authoring).

    Mirrors the identically-named helper in ``test_articulation.py``: this asset authors
    ``physics:body0``/``physics:body1`` the opposite way from what Newton expects (body0 should
    be the parent).
    """
    from pxr import UsdPhysics

    root_bodies: set[str] = set()
    joints_to_check = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        body0_targets = prim.GetRelationship("physics:body0").GetTargets()
        body1_targets = prim.GetRelationship("physics:body1").GetTargets()
        if body0_targets and not body1_targets:
            root_bodies.add(str(body0_targets[0]))
        elif body1_targets and not body0_targets:
            root_bodies.add(str(body1_targets[0]))
        elif body0_targets and body1_targets:
            joints_to_check.append(prim)
    if not root_bodies:
        return
    for prim in joints_to_check:
        body0_rel = prim.GetRelationship("physics:body0")
        body1_rel = prim.GetRelationship("physics:body1")
        body0_path = str(body0_rel.GetTargets()[0])
        body1_path = str(body1_rel.GetTargets()[0])
        body1_is_parent = body1_path in root_bodies or body0_path.startswith(body1_path + "/")
        body0_is_parent = body0_path in root_bodies or body1_path.startswith(body0_path + "/")
        if body0_is_parent or not body1_is_parent:
            continue
        body0_rel.SetTargets(body1_rel.GetTargets())
        body1_rel.SetTargets([body0_path])
        for attr_suffix in ("localPos", "localRot"):
            attr0 = prim.GetAttribute(f"physics:{attr_suffix}0")
            attr1 = prim.GetAttribute(f"physics:{attr_suffix}1")
            val0, val1 = attr0.Get(), attr1.Get()
            if val0 is not None and val1 is not None:
                attr0.Set(val1)
                attr1.Set(val0)


def _single_joint_cfg(velocity_limit_sim: float) -> ArticulationCfg:
    """A single-revolute-joint implicit-actuator articulation.

    Mirrors ``generate_articulation_cfg("single_joint_implicit", ...)`` in
    ``test_articulation.py``.
    """
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=5.0),
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit_sim=velocity_limit_sim,
                stiffness=STIFFNESS,
                damping=DAMPING,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"RevoluteJoint": 1.5708},
            rot=(0.7071081, 0, 0, 0.7071055),
        ),
    )


def _run(
    *,
    velocity_limit_sim: float,
    target_offset: float,
    backend: str,
    enforce_velocity_limit: bool,
    num_steps: int,
    num_articulations: int = 1,
) -> torch.Tensor:
    """Build a single-joint articulation, command a step target, and return the joint-position
    history.

    Steps the simulation directly (``write_data_to_sim`` + ``sim.step`` once per tick, no env, no
    decimation), matching the cadence :attr:`DT` documents.

    Returns:
        Joint position at t=0 (post-homing) and after every subsequent physics tick. Shape is
        (num_articulations, num_steps + 1).
    """
    newton_cfg = NewtonCfg(
        solver_cfg=_solver_cfg(backend),
        num_substeps=1,
        use_cuda_graph=False,
        enforce_velocity_limit=enforce_velocity_limit,
    )
    sim_cfg = SimulationCfg(dt=DT, physics=newton_cfg)
    art_cfg = _single_joint_cfg(velocity_limit_sim)

    with build_simulation_context(
        device="cuda:0", gravity_enabled=False, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim:
        sim._app_control_on_stop_handle = None
        for i in range(num_articulations):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 2.5, 0.0, 0.0))
        articulation = Articulation(art_cfg.replace(prim_path="/World/Env_.*/Robot"))

        import omni.usd  # noqa: PLC0415

        _fix_reversed_joints(omni.usd.get_context().get_stage())
        sim.reset()
        assert articulation.is_initialized

        # Standalone tests (no env) do not go through the reset path that teleports joints to
        # their default pose -- do it explicitly (mirrors _setup_franka_at_home_pose in
        # test_articulation.py).
        articulation.write_joint_position_to_sim_index(position=articulation.data.default_joint_pos.torch.clone())
        articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch.clone())

        init_pos = articulation.data.joint_pos.torch.clone()
        target = init_pos + target_offset
        articulation.set_joint_position_target_index(target=target)

        history = [init_pos.clone()]
        for _ in range(num_steps):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)
            history.append(articulation.data.joint_pos.torch.clone())
        return torch.cat(history, dim=1)


@pytest.mark.parametrize("backend", ["mujoco", "sap"])
def test_velocity_limit_clamps_target_rate(backend):
    """A configured velocity limit bounds the per-tick measured joint displacement.

    Commands a 2 rad step -- far beyond what a 0.5 rad/s limit could cover in a 1/120 s tick --
    and verifies every tick's measured displacement stays within 5% of ``velocity_limit * dt``,
    the PhysX-parity bound documented on :attr:`~isaaclab_newton.physics.NewtonCfg.
    enforce_velocity_limit`. Two articulation instances exercise the clamp's env axis.
    """
    velocity_limit = 0.5
    history = _run(
        velocity_limit_sim=velocity_limit,
        target_offset=2.0,
        backend=backend,
        enforce_velocity_limit=True,
        num_steps=40,
        num_articulations=2,
    )
    per_tick_displacement = (history[:, 1:] - history[:, :-1]).abs()
    max_allowed = velocity_limit * DT * TOLERANCE
    max_observed = per_tick_displacement.max().item()
    assert max_observed <= max_allowed, (
        f"measured joint displacement exceeded velocity_limit * dt * {TOLERANCE} "
        f"({max_allowed:.6f} rad/tick): max observed = {max_observed:.6f} rad/tick"
    )


@pytest.mark.parametrize("backend", ["mujoco", "sap"])
def test_unconfigured_velocity_limit_is_unaffected(backend):
    """A joint at the PhysX/Newton "no limit" sentinel (1e6) is completely unaffected.

    Compares the trajectory with :attr:`~isaaclab_newton.physics.NewtonCfg.
    enforce_velocity_limit` on (but no joint below the sentinel, so the clamp's whole-
    articulation early exit keeps it inactive) against the same run with the cfg gate off:
    both must produce identical joint trajectories.
    """
    kwargs = dict(velocity_limit_sim=1.0e6, target_offset=0.02, backend=backend, num_steps=15)
    with_gate_on = _run(enforce_velocity_limit=True, **kwargs)
    with_gate_off = _run(enforce_velocity_limit=False, **kwargs)
    torch.testing.assert_close(with_gate_on, with_gate_off, atol=1e-5, rtol=1e-5)
