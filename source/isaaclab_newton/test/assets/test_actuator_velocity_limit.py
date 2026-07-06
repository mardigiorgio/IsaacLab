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
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip

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


def _franka_mixed_limit_cfg(velocity_limit_sim: float) -> ArticulationCfg:
    """:data:`FRANKA_PANDA_CFG` with the arm joints rate-limited and the finger joints unlimited.

    Overrides FRANKA_PANDA_CFG's actuators (which give every joint -- arm and finger alike -- a
    finite ``velocity_limit_sim`` per the real asset defaults) with a 2-group split: the arm joints
    (including ``panda_joint1``, used below) get a finite, rate-limited ``velocity_limit_sim``; the
    finger joints (including ``panda_finger_joint1``) are explicitly pinned to the PhysX/Newton "no
    limit" sentinel (``1.0e6``). This gives a single multi-DOF articulation that exercises the
    clamp kernel's per-element branch (limited vs. pass-through) in one launch -- the task-14
    report flagged the lack of such coverage as a residual gap (the single-joint fixture used by
    the other tests in this file cannot exercise it in isolation).
    """
    return FRANKA_PANDA_CFG.replace(
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint.*"],
                velocity_limit_sim=velocity_limit_sim,
                stiffness=STIFFNESS,
                damping=DAMPING,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                velocity_limit_sim=1.0e6,
                stiffness=STIFFNESS,
                damping=DAMPING,
            ),
        },
    )


def _run(
    *,
    target_offset: float | torch.Tensor,
    backend: str,
    enforce_velocity_limit: bool,
    num_steps: int,
    velocity_limit_sim: float | None = None,
    art_cfg: ArticulationCfg | None = None,
    num_articulations: int = 1,
) -> tuple[torch.Tensor, list[str]]:
    """Build an articulation, command a step target, and return the joint-position history.

    Steps the simulation directly (``write_data_to_sim`` + ``sim.step`` once per tick, no env, no
    decimation), matching the cadence :attr:`DT` documents.

    Args:
        target_offset: Added to the post-homing joint position(s) to form the commanded target.
            Broadcast against shape (num_articulations, num_joints).
        backend: ``"mujoco"`` or ``"sap"``.
        enforce_velocity_limit: Forwarded to :attr:`~isaaclab_newton.physics.NewtonCfg.
            enforce_velocity_limit`.
        num_steps: Number of physics ticks to run after the target is commanded.
        velocity_limit_sim: Used to build the default single-joint fixture via
            :func:`_single_joint_cfg` when ``art_cfg`` is not given.
        art_cfg: Explicit articulation configuration (e.g. :func:`_franka_mixed_limit_cfg`). Overrides
            ``velocity_limit_sim`` when given.
        num_articulations: Number of articulation instances to spawn side by side.

    Returns:
        A tuple of:

        * Joint position at t=0 (post-homing) and after every subsequent physics tick. Shape is
          (num_articulations, num_joints, num_steps + 1).
        * The articulation's joint names, in the same order as the joint axis above.
    """
    newton_cfg = NewtonCfg(
        solver_cfg=_solver_cfg(backend),
        num_substeps=1,
        use_cuda_graph=False,
        enforce_velocity_limit=enforce_velocity_limit,
    )
    sim_cfg = SimulationCfg(dt=DT, physics=newton_cfg)
    if art_cfg is None:
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
        return torch.stack(history, dim=-1), articulation.joint_names


@pytest.mark.parametrize("backend", ["mujoco", "sap"])
def test_velocity_limit_clamps_target_rate(backend):
    """A configured velocity limit bounds the per-tick measured joint displacement.

    Commands a 2 rad step -- far beyond what a 0.5 rad/s limit could cover in a 1/120 s tick --
    and verifies every tick's measured displacement stays within 5% of ``velocity_limit * dt``,
    the PhysX-parity bound documented on :attr:`~isaaclab_newton.physics.NewtonCfg.
    enforce_velocity_limit`. Two articulation instances exercise the clamp's env axis.
    """
    velocity_limit = 0.5
    history, _ = _run(
        velocity_limit_sim=velocity_limit,
        target_offset=2.0,
        backend=backend,
        enforce_velocity_limit=True,
        num_steps=40,
        num_articulations=2,
    )
    per_tick_displacement = (history[:, :, 1:] - history[:, :, :-1]).abs()
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
    with_gate_on, _ = _run(enforce_velocity_limit=True, **kwargs)
    with_gate_off, _ = _run(enforce_velocity_limit=False, **kwargs)
    torch.testing.assert_close(with_gate_on, with_gate_off, atol=1e-5, rtol=1e-5)


_FRANKA_LIMITED_JOINT_NAME = "panda_joint1"
_FRANKA_LIMITED_JOINT_IDX = 0
_FRANKA_UNLIMITED_JOINT_NAME = "panda_finger_joint1"
_FRANKA_UNLIMITED_JOINT_IDX = 7
"""Joint name/index pairs assumed by :func:`test_mixed_limit_articulation_clamps_only_the_limited_joint`.

:data:`FRANKA_PANDA_CFG`'s joint order is ``[panda_joint1..7, panda_finger_joint1, panda_finger_joint2]``;
the test asserts this assumption explicitly before relying on the indices, so a future asset/ordering
change fails loudly instead of silently testing the wrong joints.
"""


@pytest.mark.parametrize("backend", ["mujoco", "sap"])
def test_mixed_limit_articulation_clamps_only_the_limited_joint(backend):
    """Within one multi-DOF articulation, only the joint with a finite configured velocity limit
    is rate-limited; a joint explicitly left at the sentinel (no configured limit) tracks its
    commanded target unrestricted by any clamp.

    Uses :func:`_franka_mixed_limit_cfg` (:data:`FRANKA_PANDA_CFG` with a 2-group actuator
    override) so the clamp kernel's per-element branch (limited vs. pass-through) is exercised
    within a single articulation launch -- the task-14 report flagged the lack of such coverage as
    a residual gap, since the single-joint fixture used by the other tests in this file cannot
    exercise it in isolation.

    The first check reads the clamp kernel's output (:attr:`~isaaclab_newton.assets.Articulation.
    _joint_pos_target_sim`) directly after a single :meth:`~isaaclab_newton.assets.Articulation.
    write_data_to_sim` call, before any physics step: this isolates the kernel's per-element branch
    selection from each backend's own transient physics response (which differs enough between
    mujoco and sap that thresholding a *measured* first-tick displacement is not reliable across
    both). The remaining checks then drive the sim forward and confirm the physically measured
    behavior over a full run is consistent with that.
    """
    velocity_limit = 0.5
    newton_cfg = NewtonCfg(
        solver_cfg=_solver_cfg(backend), num_substeps=1, use_cuda_graph=False, enforce_velocity_limit=True
    )
    sim_cfg = SimulationCfg(dt=DT, physics=newton_cfg)
    art_cfg = _franka_mixed_limit_cfg(velocity_limit)

    with build_simulation_context(
        device="cuda:0", gravity_enabled=False, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        articulation = Articulation(art_cfg.replace(prim_path="/World/Env_0/Robot"))
        sim.reset()
        assert articulation.is_initialized

        joint_names = articulation.joint_names
        assert joint_names[_FRANKA_LIMITED_JOINT_IDX] == _FRANKA_LIMITED_JOINT_NAME
        assert joint_names[_FRANKA_UNLIMITED_JOINT_IDX] == _FRANKA_UNLIMITED_JOINT_NAME

        articulation.write_joint_position_to_sim_index(position=articulation.data.default_joint_pos.torch.clone())
        articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch.clone())

        home = articulation.data.joint_pos.torch.clone()
        target = home.clone()
        target[:, _FRANKA_LIMITED_JOINT_IDX] += 2.0  # far beyond one tick's reach at the limit
        target[:, _FRANKA_UNLIMITED_JOINT_IDX] -= 0.04  # full finger stroke, position-limit-safe
        articulation.set_joint_position_target_index(target=target)

        articulation.write_data_to_sim()
        commanded = wp.to_torch(articulation._joint_pos_target_sim).clone()

        max_allowed = velocity_limit * DT * TOLERANCE
        max_delta = velocity_limit * DT
        expected_limited = home[:, _FRANKA_LIMITED_JOINT_IDX] + max_delta
        torch.testing.assert_close(commanded[:, _FRANKA_LIMITED_JOINT_IDX], expected_limited, atol=1e-5, rtol=0.0)
        # The unlimited joint's raw target passes through the kernel completely unmodified.
        torch.testing.assert_close(
            commanded[:, _FRANKA_UNLIMITED_JOINT_IDX], target[:, _FRANKA_UNLIMITED_JOINT_IDX], atol=1e-6, rtol=0.0
        )

        # Drive the sim forward and confirm the physically measured behavior matches: the limited
        # joint's per-tick displacement stays within its velocity_limit * dt budget, while the
        # unlimited joint actually reaches its commanded target within the run.
        history = [home.clone()]
        for _ in range(40):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)
            history.append(articulation.data.joint_pos.torch.clone())
        history = torch.stack(history, dim=-1)

        limited = history[:, _FRANKA_LIMITED_JOINT_IDX, :]
        limited_disp = (limited[:, 1:] - limited[:, :-1]).abs()
        max_observed_limited = limited_disp.max().item()
        assert max_observed_limited <= max_allowed, (
            f"the joint with a finite configured velocity limit ({_FRANKA_LIMITED_JOINT_NAME}) exceeded"
            f" velocity_limit * dt * {TOLERANCE} ({max_allowed:.6f}): max observed ="
            f" {max_observed_limited:.6f}"
        )

        final_unlimited = history[:, _FRANKA_UNLIMITED_JOINT_IDX, -1]
        target_unlimited = target[:, _FRANKA_UNLIMITED_JOINT_IDX]
        torch.testing.assert_close(final_unlimited, target_unlimited, atol=2e-3, rtol=0.0)


@pytest.mark.parametrize("backend", ["mujoco", "sap"])
def test_joint_position_write_reseeds_clamp_anchor(backend):
    """A mid-run joint-position write re-seeds the velocity-limit clamp's anchor to the written pose.

    :meth:`~isaaclab_newton.assets.Articulation.write_joint_position_to_sim_index` is exactly what
    ``InteractiveScene.reset_to`` (mimic's state injection) and reset-mode joint randomization
    (e.g. the stack task's ``randomize_joint_by_gaussian_offset``) call to teleport a joint to a
    new pose -- neither goes through a fresh :meth:`~isaaclab_newton.assets.Articulation.reset`
    call afterward with the new pose in hand. Without re-seeding the clamp's "previous commanded
    target" anchor at the write site, the anchor is left wherever the prior episode's
    :meth:`~isaaclab_newton.assets.Articulation.write_data_to_sim` last put it. This test
    reproduces that shape: it manufactures a stale anchor (a far-off target commanded and stepped
    once), teleports the joint to an unrelated new pose the way state injection does, then commands
    a target close to the new pose. Without the anchor being re-seeded at the write site, that
    small step is spuriously clamped against the stale, now-irrelevant anchor instead of the pose
    the joint was just teleported to -- this test fails without that fix (see the fix-pass section
    of the task-14 report for the captured failing-first output).
    """
    velocity_limit = 0.5
    newton_cfg = NewtonCfg(
        solver_cfg=_solver_cfg(backend), num_substeps=1, use_cuda_graph=False, enforce_velocity_limit=True
    )
    sim_cfg = SimulationCfg(dt=DT, physics=newton_cfg)
    art_cfg = _single_joint_cfg(velocity_limit)

    with build_simulation_context(
        device="cuda:0", gravity_enabled=False, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        articulation = Articulation(art_cfg.replace(prim_path="/World/Env_0/Robot"))

        import omni.usd  # noqa: PLC0415

        _fix_reversed_joints(omni.usd.get_context().get_stage())
        sim.reset()
        assert articulation.is_initialized

        articulation.write_joint_position_to_sim_index(position=articulation.data.default_joint_pos.torch.clone())
        articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch.clone())

        # Manufacture a stale anchor, as if a previous episode had commanded a far-away target and
        # advanced the clamp's memory by one tick's worth of ramp before the episode ended.
        home = articulation.data.joint_pos.torch.clone()
        far_target = home + 2.0
        articulation.set_joint_position_target_index(target=far_target)
        articulation.write_data_to_sim()
        sim.step()
        articulation.update(DT)

        # Simulate state injection (InteractiveScene.reset_to): teleport the joint to a new pose,
        # far from wherever the stale anchor above ended up, without going through Articulation.reset().
        new_pose = torch.full_like(home, 1.4)
        articulation.write_joint_position_to_sim_index(position=new_pose.clone())
        articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(new_pose))

        # Command a target close to the injected pose -- well within one tick's velocity_limit *
        # dt budget.
        small_delta = 0.5 * velocity_limit * DT
        near_target = new_pose + small_delta
        articulation.set_joint_position_target_index(target=near_target)
        articulation.write_data_to_sim()

        # Read the clamp kernel's output directly, before any physics step: at the stiff PD gains
        # used here, one physics tick's actual *position* response is too small (dominated by
        # inertia over a 1/120 s step) to reliably distinguish a correctly- from an incorrectly-
        # clamped *target* -- the commanded target itself is the precise, solver-independent
        # signal this fix changes.
        commanded = wp.to_torch(articulation._joint_pos_target_sim).clone()
        # Correctly re-seeded, the anchor sits at new_pose, so near_target passes through the clamp
        # completely unmodified. A stale anchor instead clamps the commanded target toward wherever
        # the stale anchor is (here, back up near the pre-injection ~1.575 rad), even though the
        # joint has already been teleported down to new_pose (1.4 rad).
        torch.testing.assert_close(commanded, near_target, atol=1e-6, rtol=0.0)

        # Sanity-check the sim still steps stably with the (correctly) clamped-through target.
        sim.step()
        articulation.update(DT)
        measured = articulation.data.joint_pos.torch.clone()
        torch.testing.assert_close(measured, near_target, atol=5e-3, rtol=0.0)
