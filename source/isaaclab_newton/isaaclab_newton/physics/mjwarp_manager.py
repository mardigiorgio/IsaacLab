# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np
from newton import Contacts, Model
from newton.solvers import SolverMuJoCo, SolverMuJoCoAdaptive

from isaaclab.physics import PhysicsManager

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


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
        if backend == "sap":
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
                    max_substeps=int(_env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))),
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
            "class_type", "solver_type", "ls_parallel",
            "adaptive", "adaptive_tol", "adaptive_dt_mode", "adaptive_dt_init", "adaptive_dt_min",
            "adaptive_tiling", "adaptive_max_substeps",
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
                max_substeps=int(_env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))),
                **kwargs,
            )
            cls._adaptive = True
            cls._adaptive_frame = 0
            logger.info("NewtonMJWarpManager: SolverMuJoCoAdaptive (adaptive step-doubling; CUDA graph disabled)")
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

        Adaptive: drive :class:`SolverMuJoCoAdaptive` via ``step_dt`` (owns its inner
        error-controlled dt loop + its own contacts, updates ``state_0`` in place), then
        consume Fix A's divergence latch — a world that hit the dt_min floor non-finite
        held its last-good state; reset its controller buffers (flags=0 keeps its joint
        state) so the latched floor dt / diverged flag don't persist. Otherwise the stock
        single ``solver.step`` (5-positional).
        """
        if getattr(cls, "_sap", False):
            if cls._adaptive:
                # SAP-adaptive owns its inner even+global loop + its own contacts; updates state_0.
                cls._solver.step_dt(substep_dt, state_0, state_1, control)
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
            cls._solver.step_dt(substep_dt, state_0, state_1, control)
            cls._solver.reset(state_0, world_mask=cls._solver.diverged, flags=0)
        else:
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)

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
        # The adaptive solver's per-frame substep count is data-dependent (host-synced boundary check),
        # so a statically captured CUDA graph cannot represent it. Both SAP paths also host-sync
        # (fixed-step SAP reads solve_result.converged each step; SAP-adaptive reads the shared N),
        # so capture is disabled for any SAP backend too.
        return not (cls._adaptive or cls._sap)

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
