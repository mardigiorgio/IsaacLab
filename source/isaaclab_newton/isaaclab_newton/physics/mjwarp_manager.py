# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverMuJoCoAdaptive

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"\d+")


def _prim_disable_gravity(stage, path: str) -> bool:
    """Read ``physxRigidBody:disableGravity`` off the USD prim at ``path``.

    Returns ``False`` for an empty path, a prim that no longer resolves, or an unauthored/invalid
    attribute.
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

    :attr:`~newton.Model.body_world_start` describes the per-world partition of body indices,
    with an optional prefix/suffix of "global", world-index ``-1`` bodies (e.g. a scene-wide
    ground plane) around ``world_count`` per-world blocks.

    The template-and-broadcast fast path in :func:`_disable_gravity_body_mask` additionally
    requires every per-world block to be the *same size* and lay out its bodies *identically*
    (up to the per-env numeric index baked into the USD path, e.g. ``env_0`` vs ``env_1``).
    That is verified here: block sizes must match, and world 0's and world 1's labels must
    match once per-env digits are normalized out.
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
    attribute directly onto each rigid-body prim (see :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties`),
    so the per-body intent is recovered by re-reading the same attribute from the prims the
    finalized model was built from. Bodies with no valid backing prim default to ``False``.

    Re-reading a USD prim + attribute per body is expensive, so when the per-world partition
    cleanly tiles from a single template (see :func:`_template_world_body_partition`), world 0's
    prims are read once and the result is broadcast to every world. "Global" bodies outside any
    per-env world (world index ``-1``, e.g. a shared ground plane) have no template to broadcast
    from and are always read directly. If the partition does not cleanly tile, this falls back
    to walking every body.
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
    :class:`~newton.solvers.SolverMuJoCoAdaptive` construction: the solver reads
    ``model.mujoco.gravcomp`` at construction time only.

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

    Owns construction of :class:`SolverMuJoCo` (or, per :attr:`MJWarpSolverCfg.backend`
    / :attr:`MJWarpSolverCfg.adaptive`, its adaptive and SAP variants), contact-buffer
    allocation in both internal-MuJoCo and Newton-pipeline contact modes, and the debug
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
    def _resolve_solver_mode(cls, solver_cfg: MJWarpSolverCfg) -> tuple[str, bool]:
        """Resolve the active backend and adaptivity from the cfg and env overrides.

        The cfg is the source of truth; ``NEWTON_SOLVER`` / ``NEWTON_SAP=1`` override the
        backend and ``NEWTON_ADAPTIVE=1`` / ``NEWTON_SAP_ADAPTIVE=1`` (or the
        ``/isaaclab/newton/adaptive`` carb setting) override adaptivity, for shell-level
        toggling without touching task configs.

        Returns:
            Tuple ``(backend, adaptive)`` where ``backend`` is ``"mujoco"`` or ``"sap"``
            and ``adaptive`` selects the step-doubling variant of that backend.
        """
        # Backend selection: cfg.backend is the source of truth; NEWTON_SOLVER / NEWTON_SAP=1
        # are shell-level env overrides.
        backend = str(getattr(solver_cfg, "backend", "mujoco"))
        if os.environ.get("NEWTON_SOLVER"):
            backend = os.environ["NEWTON_SOLVER"]
        if os.environ.get("NEWTON_SAP") == "1":
            backend = "sap"

        if backend == "sap":
            adaptive = bool(getattr(solver_cfg, "sap_adaptive", False)) or os.environ.get("NEWTON_SAP_ADAPTIVE") == "1"
            return backend, adaptive

        # Adaptive is opt-in via the cfg field; NEWTON_ADAPTIVE=1 is a shell-level override for quick toggling.
        adaptive = bool(getattr(solver_cfg, "adaptive", False)) or os.environ.get("NEWTON_ADAPTIVE") == "1"
        # Escape hatch: anything that sets this carb setting flips the integrator at the next
        # solver build (Stop/Play rebuilds it). The auto-loading GUI checkbox that used to set
        # it was removed; the cfg field and NEWTON_ADAPTIVE=1 above are the supported paths.
        if not adaptive:
            try:
                import carb

                adaptive = bool(carb.settings.get_settings().get_as_bool("/isaaclab/newton/adaptive"))
            except Exception:
                pass
        return backend, adaptive

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverBase:
        """Construct the configured solver for ``model`` without changing manager state.

        Dispatches on :meth:`_resolve_solver_mode`:

        * ``("mujoco", False)`` — stock :class:`SolverMuJoCo`; ctor kwargs are
          signature-filtered from the cfg via
          :meth:`NewtonManager._filter_solver_kwargs` (dropping non-constructor
          metadata and the ignored deprecated ``ls_parallel`` field).
        * ``("mujoco", True)`` — :class:`SolverMuJoCoAdaptive` (error-controlled step
          doubling); the filtered MuJoCo kwargs are forwarded through its ``**kwargs``,
          minus ``use_mujoco_contacts`` / ``use_mujoco_cpu`` / ``separate_worlds`` which
          the solver forces itself (its own contact pipeline + step doubling).
        * ``("sap", False)`` — vendored fixed-step :class:`~newton.solvers.SolverSAP`
          built on a ``SapModel`` (fed each step from Newton's
          :class:`CollisionPipeline` contacts, see :meth:`_step_solver`).
        * ``("sap", True)`` — :class:`~newton.solvers.SolverSAPAdaptive` (step doubling
          over SAP; owns its own contact pipeline).

        Per-body ``disable_gravity`` prims are mapped onto MuJoCo per-body ``gravcomp=1.0``
        *before* construction on the MuJoCo backends (the solver reads ``gravcomp`` at
        construction time only); the SAP backends have no per-body mechanism and log an
        actionable warning instead. The SAP solvers duck-type the
        :class:`~newton.solvers.SolverBase` ``step``/``reset`` contract.
        """
        backend, adaptive = cls._resolve_solver_mode(solver_cfg)

        # Bodies whose IsaacLab cfg set disable_gravity=True. MuJoCo-Warp honors this per body
        # via gravcomp (see _apply_gravity_compensation); the SAP backend has no per-body
        # gravity-compensation mechanism, so it can only warn.
        disable_gravity_mask = _disable_gravity_body_mask(model)

        _env = os.environ.get
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

            if adaptive:
                # Error-controlled step-doubling SAP. Owns its own contact pipeline, so no
                # manager-level collision pipeline; reuses the _adaptive step/reset/no-graph
                # wiring (host-synced boundary, like SolverMuJoCoAdaptive).
                return SolverSAPAdaptive(
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
            # Fixed-step SAP: Newton's CollisionPipeline feeds SapContacts each step
            # (converted in _step_solver).
            sap_model = sap_model_from_newton(model)
            return SolverSAP(
                sap_model,
                max_rigid_contact=int(solver_cfg.sap_max_rigid_contact),
                max_iterations=int(solver_cfg.sap_solver_iterations),
                contact_tau_d=float(solver_cfg.sap_contact_tau_d),
                contact_preset_variant=str(solver_cfg.sap_contact_preset),
                line_search_variant=str(solver_cfg.sap_line_search),
            )

        # Must run before SolverMuJoCo/SolverMuJoCoAdaptive construction below (see
        # _apply_gravity_compensation's docstring for why post-construction is too late).
        _apply_gravity_compensation(model, disable_gravity_mask)

        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if not adaptive:
            return SolverMuJoCo(model, **kwargs)

        # SolverMuJoCoAdaptive forces use_mujoco_contacts/use_mujoco_cpu/separate_worlds itself
        # (its own contact pipeline + step-doubling), so those must not be forwarded.
        for forced in ("use_mujoco_contacts", "use_mujoco_cpu", "separate_worlds"):
            kwargs.pop(forced, None)
        # cfg fields are the source of truth; NEWTON_ADAPTIVE_* env vars override for quick tuning.
        solver = SolverMuJoCoAdaptive(
            model,
            # honor the cfg's contact source: with use_mujoco_contacts=False the
            # manager's Newton CollisionPipeline contacts are injected per boundary
            use_newton_contacts=not bool(getattr(solver_cfg, "use_mujoco_contacts", True)),
            tol=float(_env("NEWTON_ADAPTIVE_TOL", getattr(solver_cfg, "adaptive_tol", 1e-3))),
            dt_mode=str(_env("NEWTON_ADAPTIVE_DTMODE", getattr(solver_cfg, "adaptive_dt_mode", "per_world"))),
            dt_inner_init=float(_env("NEWTON_ADAPTIVE_DT_INIT", getattr(solver_cfg, "adaptive_dt_init", 0.01))),
            dt_inner_min=float(_env("NEWTON_ADAPTIVE_DT_MIN", getattr(solver_cfg, "adaptive_dt_min", 1e-6))),
            tiling=str(_env("NEWTON_ADAPTIVE_TILING", getattr(solver_cfg, "adaptive_tiling", "ragged"))),
            max_substeps=int(_env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))),
            dt_histogram=str(
                _env("NEWTON_ADAPTIVE_DT_HIST", "1" if getattr(solver_cfg, "adaptive_dt_histogram", False) else "0")
            )
            not in ("0", "", "false", "False"),
            **kwargs,
        )
        # NEWTON_ADAPTIVE_JOINT_SCALE=<s> down-weights hinge/slide qpos coords in the error
        # metric: sets S=s for every hinge/slide coord (free-joint coords keep S=1), i.e.
        # tol/s on joints only. Runs before the first step, so the in-place copy is
        # CUDA-graph-capture safe.
        js = os.environ.get("NEWTON_ADAPTIVE_JOINT_SCALE")
        if js:
            mjm = solver.mj_model
            scale = solver._state_scale.numpy()
            for j in range(mjm.njnt):
                if mjm.jnt_type[j] in (2, 3):  # slide, hinge
                    scale[:, mjm.jnt_qposadr[j]] = float(js)
            wp.copy(
                solver._state_scale,
                wp.array(scale, dtype=wp.float32, device=solver._state_scale.device),
            )
            logger.info(f"NewtonMJWarpManager: joint error-scale override S={js} on hinge/slide coords")
        return solver

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct the configured solver via :meth:`_create_solver` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``, the adaptive/SAP
        control fields) and the ignored deprecated ``ls_parallel`` field are not
        forwarded. Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True``
        only when the solver consumes Newton :class:`CollisionPipeline` contacts:
        ``use_mujoco_contacts=False`` on the MuJoCo backends, always for fixed-step
        SAP, and never for SAP-adaptive (which owns its own contact pipeline).
        """
        backend, adaptive = cls._resolve_solver_mode(solver_cfg)
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        cls._adaptive = adaptive
        cls._sap = backend == "sap"
        if cls._adaptive:
            cls._adaptive_frame = 0

        if cls._sap:
            if cls._adaptive:
                NewtonManager._sap_model = None
                NewtonManager._needs_collision_pipeline = False
                logger.info(
                    "NewtonMJWarpManager: SolverSAPAdaptive (SAP step-doubling; even+global; "
                    "solver-internal per-N CUDA-graph capture, set NEWTON_SAP_ADAPTIVE_GRAPH=0 to disable)"
                )
            else:
                # SolverSAP stores the SapModel built in _create_solver as its .model.
                NewtonManager._sap_model = getattr(NewtonManager._solver, "model", None)
                NewtonManager._needs_collision_pipeline = True
                logger.info("NewtonMJWarpManager: SolverSAP (fixed-step convex contact; CUDA graph disabled)")
            return

        if cls._adaptive:
            logger.info(
                "NewtonMJWarpManager: SolverMuJoCoAdaptive (adaptive step-doubling; solver-internal "
                "per-iteration CUDA-graph replay, set NEWTON_MJ_ADAPTIVE_GRAPH=0 to disable)"
            )
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
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one solver substep.

        Adaptive: drive :class:`SolverMuJoCoAdaptive` via ``step`` (owns its inner
        error-controlled dt loop + its own contacts, updates ``state_0`` in place).
        Otherwise the stock single ``solver.step`` (5-positional).
        """
        if cls._sap:
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
        # MuJoCo fixed and adaptive share the stock 5-positional call; for the adaptive
        # solver step() is the boundary call (it owns its inner step-doubling loop + its
        # own contacts).
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """MuJoCo-adaptive: march the whole control period in one boundary call.

        The adaptive solver is itself the substepper (error-controlled inner dt), and
        control is constant across the decimation tick (actuators run once per tick,
        before this call), so one boundary call per tick suffices. SAP and fixed-step
        paths keep the stock per-substep loop.
        """
        # NEWTON_ADAPTIVE_SINGLE_BOUNDARY=0 routes the adaptive solver through the
        # stock per-substep loop instead: shorter boundaries mean injected contacts
        # are re-detected num_substeps times per tick, bounding how long the march
        # integrates against a frozen contact set.
        if cls._adaptive and not cls._sap and os.environ.get("NEWTON_ADAPTIVE_SINGLE_BOUNDARY", "1") == "1":
            cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt * cls._num_substeps)
            cls._state_0.clear_forces()
            return
        super()._run_solver_substeps(contacts)

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
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Clear solver-internal state for flagged worlds.

        Specializes the base hook, whose :meth:`SolverBase.reset` call resolves here
        per backend:

        * **Fixed-step MuJoCo** — :meth:`SolverMuJoCo.reset` with ``flags=0`` zeroes
          only the solver-owned buffers persisting across steps (``qacc_warmstart``,
          ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``) for the flagged
          worlds, while the joint state IsaacLab authored during the env reset is
          left untouched.  Without this, a NaN produced in one solve persists
          across :meth:`isaaclab.envs.ManagerBasedEnv.reset` because the next
          solver substep warm-starts from the NaN — the world is then permanently
          dead.  See https://github.com/newton-physics/newton/issues/1266.
        * **Adaptive (MuJoCo or SAP step-doubling)** — the solver's ``reset``
          additionally restores the step-doubling controller's persistent per-world
          state (dt / sim_time / next_time / accepted+diverged latches) so pre-reset
          controller state does not leak into the post-reset dynamics; ``flags=0``
          again preserves the env's randomized post-reset joint state.
        * **Fixed-step SAP** — the vendored solver has no per-world reset; its
          contact-solve warm-start is cleared globally via ``reset_runtime_state()``,
          gated on at least one world actually being flagged.  With staggered
          per-env resets, untouched envs pay a small re-convergence cost, measured
          to be dynamically negligible (see ``test_mimic_state_seam.py``).

        With ``use_mujoco_cpu=True`` the solver owns a single global ``MjData``
        and its reset path is not mask-aware — it clears the buffers for every
        world.  Since this hook fires on every step/forward boundary (usually
        with an all-``False`` mask), the CPU path is gated on at least one
        world actually being flagged so warm-starting is not defeated on every
        step.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count + 1,)``.
                Entries before the last select local worlds; the final entry
                selects global entities in world -1. ``None`` is a no-op.
        """
        if world_mask is None or cls._solver is None:
            return
        # solver.reset takes the canonical (world_count + 1,) mask (trailing
        # global-entities slot); slice only for the host-side any() gates.
        local_mask = world_mask[: cls._model.world_count]
        if cls._sap and not cls._adaptive:
            if local_mask.numpy().any():
                cls._solver.reset_runtime_state()
            return
        if cls._adaptive:
            cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)
            return
        if cls._solver.use_mujoco_cpu and not local_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active solver configuration supports CUDA graph capture.

        MANAGER-level capture stays OFF for SAP (owns its capture internally). For the
        MuJoCo adaptive solver it is opt-in via ``NEWTON_MJ_ADAPTIVE_CONDITIONAL=1``: in
        that mode the solver's data-dependent boundary loop records as a CUDA
        conditional while-node (``wp.capture_while``), with mujoco_warp's per-step
        scratch allocations hidden behind the MjwStepAllocCache shim (CUDA forbids
        allocation nodes inside conditional bodies). By default the adaptive solver
        instead owns its capture internally (one regular graph per iteration body,
        replayed with a 4-byte boundary-flag poll), which the manager must not wrap.
        """
        if cls._sap:
            return False
        if cls._adaptive:
            return os.environ.get("NEWTON_MJ_ADAPTIVE_CONDITIONAL", "0") == "1"
        return True

    @classmethod
    def _log_adaptive_telemetry(cls) -> None:
        """File-based dt/substep telemetry, throttled to every Nth frame (``NEWTON_ADAPTIVE_LOG_EVERY``).

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
