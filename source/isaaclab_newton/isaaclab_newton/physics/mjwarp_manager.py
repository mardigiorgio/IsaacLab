# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import inspect
import logging
import os
import re

import numpy as np
from newton import Contacts, Model
from newton.solvers import SolverMuJoCo, SolverMuJoCoAdaptive

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"\d+")


def _prim_disable_gravity(stage, path: str) -> bool:
    """Read ``physxRigidBody:disableGravity`` off the USD prim at ``path``.

    Returns ``False`` for an empty path, a prim that no longer resolves, or an unauthored/invalid
    attribute -- mirrors the previous inline per-body walk exactly.
    """
    if not path:
        return False
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return False
    attr = prim.GetAttribute("physxRigidBody:disableGravity")
    return bool(attr.IsValid() and bool(attr.Get()))


def _template_world_body_partition(model: Model) -> tuple[int, int, int, int] | None:
    """Return ``(template_start, template_end, per_world_tail, world_count)`` for the contiguous
    block of world-0 body indices, or ``None`` if per-world bodies cannot be safely tiled from a
    single template and the caller must fall back to walking every body.

    :attr:`~newton.Model.body_world_start` always describes a clean, contiguous per-world
    partition of body indices (an optional prefix/suffix of "global", world-index ``-1`` bodies --
    e.g. a scene-wide ground plane -- around ``world_count`` per-world blocks); Newton's
    ``ModelBuilder.finalize`` validates this by construction (see ``_validate_world_ordering`` in
    ``newton/_src/sim/builder.py``), so IsaacLab does not need to re-check it.

    What Newton does *not* guarantee -- and what the template-and-broadcast fast path in
    :func:`_disable_gravity_body_mask` needs -- is that every per-world block is the *same size*
    and lays out its bodies *identically* (up to the per-env numeric index baked into the USD
    path, e.g. ``env_0`` vs ``env_1``). IsaacLab's env clones are structurally identical, but that
    is an IsaacLab-level convention, not a Newton invariant, so it is verified here: block sizes
    must match, and (cheaply, bounded by one env's body count rather than the total) world 0's and
    world 1's labels must match once per-env digits are normalized out.
    """
    starts_arr = getattr(model, "body_world_start", None)
    world_count = int(getattr(model, "world_count", 0) or 0)
    if starts_arr is None or world_count <= 0:
        return None
    starts = starts_arr.numpy()
    if len(starts) < world_count + 2:
        return None

    template_start, template_end = int(starts[0]), int(starts[1])
    per_world_tail = int(starts[world_count])  # end of the last per-world block (== starts[-2])

    block_sizes = starts[1 : world_count + 1] - starts[0:world_count]
    if not np.all(block_sizes == block_sizes[0]):
        return None

    if world_count > 1:
        next_start, next_end = int(starts[1]), int(starts[2])
        template_labels = [_DIGITS_RE.sub("#", label) for label in model.body_label[template_start:template_end]]
        next_labels = [_DIGITS_RE.sub("#", label) for label in model.body_label[next_start:next_end]]
        if template_labels != next_labels:
            return None

    return template_start, template_end, per_world_tail, world_count


def _disable_gravity_body_mask(model: Model) -> np.ndarray:
    """Return a per-body boolean mask, True where the body's USD prim (:attr:`Model.body_label`)
    has ``physxRigidBody:disableGravity`` authored ``True``.

    :class:`~isaaclab.sim.schemas.RigidBodyBaseCfg.disable_gravity` writes this PhysX-namespaced
    attribute directly onto each rigid-body prim (see :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties`);
    Newton's USD importer only reads it at the physics-scene level (scene-wide gravity toggle), so this
    reproduces the per-body intent by re-reading the same attribute from the prims the finalized model
    was built from. Bodies with no valid backing prim default to ``False``.

    Cost: re-reading a USD prim + attribute per body is expensive (a live stage traversal), and this
    runs on every full solver rebuild (every non-soft :meth:`NewtonManager.reset`, e.g. the mimic
    annotate replay loop). Walking every body across every one of ``num_envs`` clones would make this
    ``O(num_envs * bodies_per_env)``. IsaacLab's env clones are structurally identical and MJWarp's
    per-body model params are shared across worlds regardless (``separate_worlds=True`` consumes only
    world 0's template), so instead this reads world 0's prims once (see
    :func:`_template_world_body_partition`) and broadcasts the result to every world -- ``O(1)`` in
    ``num_envs``. "Global" bodies outside any per-env world (world index ``-1``, e.g. a shared ground
    plane) are not per-env clones and have no template to broadcast from, so they are always read
    directly; in practice there are few to none of them, so this stays cheap. If the per-world
    partition does not cleanly tile (block sizes differ, or world 0/1 bodies differ once per-env
    indices are normalized out), this falls back to the original full walk so behavior is unchanged
    for heterogeneous, non-cloned scenes.
    """
    mask = np.zeros(model.body_count, dtype=bool)
    if model.body_count == 0:
        return mask
    stage = get_current_stage()

    partition = _template_world_body_partition(model)
    if partition is None:
        for i, path in enumerate(model.body_label):
            mask[i] = _prim_disable_gravity(stage, path)
        return mask

    template_start, template_end, per_world_tail, world_count = partition

    # Global (non-cloned) bodies: no template to broadcast from, so read them directly. Bounded by
    # the (typically zero) number of such bodies, not by num_envs.
    for i in range(0, template_start):
        mask[i] = _prim_disable_gravity(stage, model.body_label[i])
    for i in range(per_world_tail, model.body_count):
        mask[i] = _prim_disable_gravity(stage, model.body_label[i])

    # Per-env (cloned) bodies: read world 0's prims once and broadcast to every world.
    template_mask = np.fromiter(
        (_prim_disable_gravity(stage, path) for path in model.body_label[template_start:template_end]),
        dtype=bool,
        count=template_end - template_start,
    )
    if template_mask.any():
        mask[template_start:per_world_tail] = np.tile(template_mask, world_count)
    return mask


def _apply_gravity_compensation(model: Model, mask: np.ndarray) -> None:
    """Set MuJoCo-Warp per-body ``gravcomp=1.0`` for bodies flagged by ``mask``.

    Must run on ``model`` before :class:`~newton.solvers.SolverMuJoCo` /
    :class:`~newton.solvers.SolverMuJoCoAdaptive` construction: the solver copies
    ``model.mujoco.gravcomp`` into its internal MJWarp model once, at construction time, and never
    re-reads the Newton :class:`~newton.Model` afterward (verified: patching ``gravcomp`` post-construction
    has no effect on the running solver).

    .. note::
        This unconditionally overwrites ``gravcomp`` to ``1.0`` for every masked body: a body with
        ``disable_gravity=True`` always ends up fully gravity-compensated, even if a user-authored
        ``mujoco:gravcomp`` custom value (any value other than ``1.0``) was set on that body. There is
        no way to author partial gravity compensation on a ``disable_gravity=True`` body today.
    """
    if not mask.any():
        return
    gravcomp = getattr(getattr(model, "mujoco", None), "gravcomp", None)
    if gravcomp is None:
        logger.warning(
            "NewtonMJWarpManager: %d body(ies) have disable_gravity=True, but this model has no "
            "'mujoco.gravcomp' custom attribute registered -- gravity compensation cannot be applied "
            "and these bodies will sag under gravity.",
            int(mask.sum()),
        )
        return
    gravcomp_np = gravcomp.numpy()
    # disable_gravity=True always wins: forces gravcomp=1.0, overwriting any user-authored value.
    gravcomp_np[mask] = 1.0
    gravcomp.assign(gravcomp_np)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    _adaptive: bool = False
    """Set by :meth:`_build_solver`: True when the active solver is the step-doubling adaptive
    solver (:class:`SolverMuJoCoAdaptive` or :class:`SolverSAPAdaptive`)."""

    _adaptive_frame: int = 0
    """Frame counter for throttled adaptive dt/substep telemetry."""

    _sap: bool = False
    """Set by :meth:`_build_solver`: True when the active backend is SAP
    (:class:`SolverSAP` fixed-step or :class:`SolverSAPAdaptive` step-doubling)."""

    _sap_model = None
    """The vendored ``SapModel`` wrapping the Newton model (fixed-step SAP path only)."""

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct the solver (:class:`SolverMuJoCo` or, when ``solver_cfg.adaptive``,
        :class:`SolverMuJoCoAdaptive`) and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so non-constructor metadata
        (``solver_type``, ``class_type``, the adaptive control fields) and the deprecated ``ls_parallel``
        field are not forwarded. Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_mujoco_contacts=False``.
        """
        # SAP backend selector (parallel branch off the new _sap latch). cfg.backend is the source of
        # truth; NEWTON_SOLVER / NEWTON_SAP=1 are shell-level overrides for quick toggling.
        backend = str(getattr(solver_cfg, "backend", "mujoco"))
        if os.environ.get("NEWTON_SOLVER"):
            backend = os.environ["NEWTON_SOLVER"]
        if os.environ.get("NEWTON_SAP") == "1":
            backend = "sap"

        # Bodies whose IsaacLab cfg set disable_gravity=True (e.g. FRANKA_PANDA_HIGH_PD_CFG's
        # gravity-free arm for its 400/80 PD tracking). MuJoCo-Warp honors this per body via
        # gravcomp (see _apply_gravity_compensation); SAP has no per-body mechanism (gravity is
        # applied per-world only, see sap_warp/sim/sap_runtime.py) so it can only warn.
        disable_gravity_mask = _disable_gravity_body_mask(model)

        if backend == "sap":
            if disable_gravity_mask.any():
                logger.warning(
                    "NewtonMJWarpManager: %d body(ies) in this scene have disable_gravity=True, "
                    "but the SAP backend applies gravity per-world only and has no per-body "
                    "gravity-compensation mechanism -- these bodies will sag under gravity on SAP. "
                    "Use MJWarpSolverCfg(backend='mujoco') (fixed or adaptive) instead if per-body "
                    "gravity compensation is required.",
                    int(disable_gravity_mask.sum()),
                )
            from newton.solvers import SolverSAP, SolverSAPAdaptive, sap_model_from_newton

            _env = os.environ.get
            sap_adaptive = bool(getattr(solver_cfg, "sap_adaptive", False)) or _env("NEWTON_SAP_ADAPTIVE") == "1"
            if sap_adaptive:
                # Error-controlled step-doubling SAP (even+global tiling). Owns its own pipeline,
                # so no manager-level collision pipeline; reuses the existing _adaptive step/reset/
                # no-graph wiring (host-synced boundary, like SolverMuJoCoAdaptive).
                NewtonManager._solver = SolverSAPAdaptive(
                    model,
                    tol=float(_env("NEWTON_ADAPTIVE_TOL", getattr(solver_cfg, "adaptive_tol", 1e-3))),
                    dt_inner_init=float(_env("NEWTON_ADAPTIVE_DT_INIT", getattr(solver_cfg, "adaptive_dt_init", 0.01))),
                    dt_inner_min=float(_env("NEWTON_ADAPTIVE_DT_MIN", getattr(solver_cfg, "adaptive_dt_min", 1e-6))),
                    max_substeps=int(
                        _env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))
                    ),
                    max_rigid_contact=int(solver_cfg.sap_max_rigid_contact),
                    max_iterations=int(solver_cfg.sap_solver_iterations),
                    contact_preset_variant=str(_env("NEWTON_SAP_PRESET", solver_cfg.sap_contact_preset)),
                    line_search_variant=str(_env("NEWTON_SAP_LINE_SEARCH", solver_cfg.sap_line_search)),
                    contact_tau_d=float(solver_cfg.sap_contact_tau_d),
                )
                cls._adaptive = True
                cls._adaptive_frame = 0
                NewtonManager._needs_collision_pipeline = False
                logger.info(
                    "NewtonMJWarpManager: SolverSAPAdaptive (SAP step-doubling; even+global; "
                    "solver-internal per-N CUDA-graph capture, set NEWTON_SAP_ADAPTIVE_GRAPH=0 to disable)"
                )
            else:
                # Fixed-step SAP: the RL-ideal (stable+consistent+fast). Newton's CollisionPipeline
                # feeds SapContacts each step (converted in _step_solver).
                sap_model = sap_model_from_newton(model)
                NewtonManager._solver = SolverSAP(
                    sap_model,
                    max_rigid_contact=int(solver_cfg.sap_max_rigid_contact),
                    max_iterations=int(solver_cfg.sap_solver_iterations),
                    contact_tau_d=float(solver_cfg.sap_contact_tau_d),
                    contact_preset_variant=str(solver_cfg.sap_contact_preset),
                    line_search_variant=str(solver_cfg.sap_line_search),
                )
                NewtonManager._sap_model = sap_model
                cls._adaptive = False
                NewtonManager._needs_collision_pipeline = True
                logger.info("NewtonMJWarpManager: SolverSAP (fixed-step convex contact; CUDA graph disabled)")
            cls._sap = True
            NewtonManager._use_single_state = True
            return
        cls._sap = False

        # Must run before SolverMuJoCo/SolverMuJoCoAdaptive construction below (see
        # _apply_gravity_compensation's docstring for why post-construction is too late).
        _apply_gravity_compensation(model, disable_gravity_mask)

        # Adaptive is opt-in via the cfg field; NEWTON_ADAPTIVE=1 is a shell-level override for quick toggling.
        adaptive = bool(getattr(solver_cfg, "adaptive", False)) or os.environ.get("NEWTON_ADAPTIVE") == "1"
        # GUI toggle: the Adaptive-timestepping checkbox sets this carb setting; reading it at solver-build
        # time means flipping it + Stop/Play (which rebuilds the solver) switches the integrator interactively.
        if not adaptive:
            try:
                import carb

                adaptive = bool(carb.settings.get_settings().get_as_bool("/isaaclab/newton/adaptive"))
            except Exception:
                pass
        ignored = {
            "class_type",
            "solver_type",
            "ls_parallel",
            "adaptive",
            "adaptive_tol",
            "adaptive_dt_mode",
            "adaptive_dt_init",
            "adaptive_dt_min",
            "adaptive_tiling",
            "adaptive_max_substeps",
        }
        if adaptive:
            # SolverMuJoCoAdaptive forces use_mujoco_contacts/use_mujoco_cpu/separate_worlds itself
            # (its own contact pipeline + step-doubling), so those must not be forwarded.
            forced = {"use_mujoco_contacts", "use_mujoco_cpu", "separate_worlds"}
            valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"} - ignored - forced
            kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
            # cfg fields are the source of truth; NEWTON_ADAPTIVE_* env vars override for quick tuning.
            _env = os.environ.get
            NewtonManager._solver = SolverMuJoCoAdaptive(
                model,
                tol=float(_env("NEWTON_ADAPTIVE_TOL", getattr(solver_cfg, "adaptive_tol", 1e-3))),
                dt_mode=str(_env("NEWTON_ADAPTIVE_DTMODE", getattr(solver_cfg, "adaptive_dt_mode", "per_world"))),
                dt_inner_init=float(_env("NEWTON_ADAPTIVE_DT_INIT", getattr(solver_cfg, "adaptive_dt_init", 0.01))),
                dt_inner_min=float(_env("NEWTON_ADAPTIVE_DT_MIN", getattr(solver_cfg, "adaptive_dt_min", 1e-6))),
                tiling=str(_env("NEWTON_ADAPTIVE_TILING", getattr(solver_cfg, "adaptive_tiling", "ragged"))),
                max_substeps=int(
                    _env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))
                ),
                **kwargs,
            )
            cls._adaptive = True
            cls._adaptive_frame = 0
            logger.info(
                "NewtonMJWarpManager: SolverMuJoCoAdaptive (adaptive step-doubling; solver-internal "
                "per-iteration CUDA-graph replay, set NEWTON_MJ_ADAPTIVE_GRAPH=0 to disable)"
            )
        else:
            valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"} - ignored
            kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
            NewtonManager._solver = SolverMuJoCo(model, **kwargs)
            cls._adaptive = False
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt) -> None:
        """Run one solver substep.

        Adaptive: drive :class:`SolverMuJoCoAdaptive` via ``step`` (owns its inner
        error-controlled dt loop + its own contacts, updates ``state_0`` in place).
        Otherwise the stock single ``solver.step`` (5-positional).
        """
        if getattr(cls, "_sap", False):
            if cls._adaptive:
                # SAP-adaptive owns its inner even+global loop + its own contacts; updates state_0.
                # step() is the boundary call (Newton signature: state_in, state_out, control,
                # contacts, dt); the whole step-doubling + N-substep sequence is ONE single-level
                # CUDA-graph capture owned INSIDE the solver, so the manager must NOT also wrap it.
                cls._solver.step(state_0, state_1, control, contacts, substep_dt)
                cls._solver.reset(state_0, world_mask=cls._solver.diverged, flags=0)
            else:
                from newton.solvers import (
                    sap_contacts_from_newton,
                    sap_control_from_newton,
                    sap_state_from_newton,
                )

                # Fixed-step SAP writes the stepped result IN PLACE into state_0 (the
                # state the env reads), mirroring the adaptive paths. _build_solver
                # forces _use_single_state=True, so the substep loop already passes
                # state_0 as both args; we pass state_0 explicitly as in AND out so the
                # result lands in state_0 regardless of the loop's buffering mode --
                # writing only into a distinct state_1 (single-state loop does no swap)
                # would leave the env reading a stale state_0 (silent no-op dynamics).
                # SAP copies the input velocity into its own boundary buffer and
                # integrates positions element-wise, so the in==out aliasing is safe.
                s0 = sap_state_from_newton(state_0)
                c = sap_control_from_newton(control)
                sc = sap_contacts_from_newton(contacts)
                cls._solver.step(s0, s0, c, sc, substep_dt)
            return
        if cls._adaptive:
            # MuJoCo-adaptive: step() is the boundary call (state_in, state_out, control,
            # contacts, dt); it owns its inner step-doubling loop + its own contacts.
            # No per-substep divergence reset: the solver's floor latch was removed
            # (`diverged` stays all-False), so that reset call was a dead kernel launch.
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        else:
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """MuJoCo-adaptive: march the WHOLE control period in one boundary call.

        The adaptive solver is itself the substepper (error-controlled inner dt), so
        driving it once per manager substep would only add forced boundary landings,
        duplicate control application, and duplicate Newton<->MuJoCo conversion + FK
        round-trips -- control is constant across the decimation tick (actuators run
        once per tick, before this call). SAP and fixed-step paths keep the stock
        per-substep loop.
        """
        if cls._adaptive and not cls._sap:
            cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt * cls._num_substeps)
            cls._state_0.clear_forces()
            return
        super()._run_solver_substeps(contacts)

    @classmethod
    def _reset_solver_state(cls, world_mask) -> None:
        """Adaptive: restore the step-doubling controller's persistent per-world
        state (dt / sim_time / next_time / latches) for worlds the env reset this
        step, so pre-reset controller state does not leak into the post-reset
        dynamics. flags=0 preserves the env's randomized post-reset joint state
        (only MuJoCo warm-start buffers + controller buffers are cleared)."""
        # Fixed-step SAP clears its contact-solve warm-start; SAP-adaptive falls through to the
        # _adaptive branch (its own .reset restores controller buffers + clears the SAP warm-start).
        if getattr(cls, "_sap", False) and not cls._adaptive and cls._solver is not None:
            cls._solver.reset_runtime_state()
            return
        if cls._adaptive and world_mask is not None and cls._solver is not None:
            cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        # MANAGER-level capture stays OFF for SAP (owns its capture internally). For the
        # MuJoCo adaptive solver it is opt-in via NEWTON_MJ_ADAPTIVE_CONDITIONAL=1: in
        # that mode the solver's data-dependent boundary loop records as a CUDA
        # conditional while-node (wp.capture_while), with mujoco_warp's per-step scratch
        # allocations hidden behind the MjwStepAllocCache shim (CUDA forbids allocation
        # nodes inside conditional bodies). By default the adaptive solver instead owns
        # its capture internally (one regular graph per iteration body, replayed with a
        # 4-byte boundary-flag poll), which the manager must not wrap.
        if cls._sap:
            return False
        if cls._adaptive:
            return os.environ.get("NEWTON_MJ_ADAPTIVE_CONDITIONAL", "0") == "1"
        return True

    @classmethod
    def _log_adaptive_telemetry(cls) -> None:
        """File-based dt/substep telemetry (Kit swallows stdout), throttled to every Nth frame.

        Writes per-world inner-dt spread + cumulative substeps so adaptivity is observable: ``spread > 0``
        (per-world mode) or a changing cumulative substep count means the controller is subdividing.
        """
        cls._adaptive_frame += 1
        every = int(os.environ.get("NEWTON_ADAPTIVE_LOG_EVERY", "30"))
        if every <= 0 or cls._adaptive_frame % every != 0:
            return
        try:
            dt = cls._solver.dt.numpy()
            subs = int(cls._solver.cumulative_substeps())
            path = os.environ.get("NEWTON_ADAPTIVE_LOG", "/tmp/newton_adaptive.log")
            with open(path, "a") as f:
                f.write(
                    f"frame={cls._adaptive_frame} inner_dt[min={dt.min():.3e} max={dt.max():.3e} "
                    f"spread={float(dt.max() - dt.min()):.3e}] cumulative_substeps={subs}\n"
                )
        except Exception as exc:  # telemetry must never break the sim
            logger.debug(f"adaptive telemetry skipped: {exc}")

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Adaptive dt/substep telemetry (to a file) + optional MuJoCo convergence logging."""
        if cls._adaptive:
            cls._log_adaptive_telemetry()
        cfg = PhysicsManager._cfg
        # MuJoCo convergence stats read mjw_data, which the SAP backends do not have.
        if cfg is not None and cfg.debug_mode and not cls._sap:  # type: ignore[union-attr]
            data = cls._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
