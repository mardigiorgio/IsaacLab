# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import copy
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


_SAP_TRIANGLE_PAIRS_PER_WORLD = 65536
"""Per-world triangle-pair capacity the SAP arms size their collision pipeline to.

The pair cap is a GLOBAL pooled buffer whose overflow drops mesh contacts
silently, so it must exceed live per-world demand times the world count; it is
also carried by the global contact reducer, which allocates ~56 bytes per unit
on top of the narrow phase's 12, so an oversized cap costs gigabytes that scale
with nothing. This constant is the per-world budget the sizing rule below
multiplies by the world count; ``tools/probes`` measures live demand.

Demand must be measured under a TRAINED policy, not a scripted stream: a
learned policy holds the object against the gripper in every world at once and
generates several times the mesh proximity any scripted rig in this campaign
reached, and per-world demand does not fall with world count.

An upper bound also applies and is not a memory bound. The reducer's hashtable
is sized to the next power of two above a quarter of this capacity, and the
stock fill-ratio diagnostic compares ``hashtable_capacity * warn_percent``
against an int32, which wraps once the hashtable passes 2**24 entries and then
warns on every collide. That puts a hard ceiling of 2**26 pooled pairs on a
quiet log, which at 1024 worlds is exactly this constant.
"""


def _sap_triangle_pair_budget(authored: int, world_count: int) -> int:
    """Scene-sized triangle-pair capacity, derived from the authored cap.

    Takes the smaller of what the task authored and the per-world budget times
    the world count, with the historical 1M floor. A cap authored for a
    different world count (or sized blind against a clamp that no longer fires)
    otherwise allocates a fixed multi-gigabyte block that does not shrink with
    the scene and can exhaust the device at high world counts.
    """
    return max(1_000_000, min(int(authored), _SAP_TRIANGLE_PAIRS_PER_WORLD * max(int(world_count), 1)))


def _clamp_deterministic_triangle_pairs(pairs: int) -> int:
    """Clamp a triangle-pair capacity to the deterministic contact-id budget.

    Deterministic contact packing indexes every buffered candidate with
    ``CONTACT_ID_BITS`` bits and the reducer REJECTS a larger capacity outright,
    so a deterministic run must not be handed one.
    """
    from newton._src.geometry.contact_reduction_global import CONTACT_ID_BITS

    return min(int(pairs), 1 << int(CONTACT_ID_BITS))


@wp.kernel(enable_backward=False)
def _accumulate_diverged_pending(diverged: wp.array(dtype=wp.bool), pending: wp.array(dtype=wp.int32)):
    """OR the solver's per-world divergence latch into the persistent pending mask.

    The adaptive solvers clear (or same-step consume) their latch before the
    env layer runs its termination terms, so the latch must be captured into a
    mask that survives until a termination term reads it.
    """
    i = wp.tid()
    if diverged[i]:
        pending[i] = 1


@wp.kernel(enable_backward=False)
def _latch_sap_solve_failure(
    converged_env: wp.array(dtype=wp.int32),
    world_active: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
    pending: wp.array(dtype=wp.int32),
):
    """Fixed-step SAP convergence certificate: latch, isolate, report.

    ``contact_solve.converged_env`` is 0 exactly where the inner Newton loop
    left an env at its iteration cap without reaching the optimality (or cost)
    test -- the same per-env array the adaptive solver folds into its own
    solve-ok state. An env that was not participating this substep is
    pre-converged by the solve's own entry kernel, so the ``world_active`` guard
    only keeps a world that has already latched from re-reporting.

    A failing world is excised from every later substep (``world_active`` 0),
    which freezes its state, and its bit is raised in the pending mask the
    ``physics_diverged`` termination consumes, so the episode is ended rather
    than continued from a solve that never converged.
    """
    i = wp.tid()
    if world_active[i] != 0 and converged_env[i] == 0:
        failed[i] = 1
        pending[i] = 1
        world_active[i] = 0


@wp.kernel(enable_backward=False)
def _latch_icf_solve_failure(
    converged_env: wp.array(dtype=wp.int32),
    pending: wp.array(dtype=wp.int32),
):
    """Fixed-step ICF convergence certificate: report a solve that never converged.

    ``IcfContactSolve.converged_env`` is 0 exactly where the inner Newton loop
    left an env at its iteration cap without meeting a convergence criterion --
    the same array, with the same 1=converged convention, that the adaptive ICF
    arm folds into its solve-ok state and turns into a rejection. Latching it
    here raises the bit the ``physics_diverged`` termination consumes, so the
    two ICF arms end an episode on the same underlying event instead of only
    the adaptive one having a divergence pathway.

    Unlike the SAP counterpart this cannot also FREEZE the world: ``SolverICF.step``
    takes no participation mask, so there is nothing to switch the world out of.
    The bit is therefore a report only, and the episode ends at the next
    termination evaluation rather than at this substep.
    """
    i = wp.tid()
    if converged_env[i] == 0:
        pending[i] = 1


@wp.kernel(enable_backward=False)
def _release_sap_failed(
    world_mask: wp.array(dtype=wp.bool),
    failed: wp.array(dtype=wp.int32),
    world_active: wp.array(dtype=wp.int32),
):
    """Return reset worlds to the active set and drop their failure latch.

    The env has rebuilt the world's state, so the solve that failed is no longer
    the state being stepped and the world must participate again.
    """
    i = wp.tid()
    if world_mask[i]:
        failed[i] = 0
        world_active[i] = 1


@wp.kernel(enable_backward=False)
def _clear_diverged_pending(world_mask: wp.array(dtype=wp.bool), pending: wp.array(dtype=wp.int32)):
    """Clear the pending divergence mask for worlds being reset.

    A reset rebuilds the world's state, so the diverged episode the mask
    reported is over; leaving the bit set would re-terminate the fresh episode
    forever.
    """
    i = wp.tid()
    if world_mask[i]:
        pending[i] = 0


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
    _icf: bool = False
    """Set by :meth:`_build_solver`: True when the active backend is SAP
    (:class:`SolverSAP` fixed-step or :class:`SolverSAPAdaptive` step-doubling)."""

    _sap_model = None
    """The vendored ``SapModel`` wrapping the Newton model (fixed-step SAP path only)."""

    _sap_failed: wp.array | None = None
    """Fixed-step SAP only: per-world int32 latch, nonzero where the inner SAP solve
    failed to converge since the world's last env reset. Drives world isolation via
    :attr:`_sap_world_active`; cleared per reset world by :meth:`_reset_solver_internals`."""

    _sap_world_active: wp.array | None = None
    """Fixed-step SAP only: per-world int32 participation mask handed to
    ``SolverSAP.step``. Zero for latched worlds, which keeps their state and their
    contact-solve rows untouched for every later substep (the freeze)."""

    _sap_strict: bool = False
    """Fixed-step SAP only: ``NEWTON_SAP_CONTAINMENT=0`` selects strict converge-or-throw
    instead of containment, mirroring the adaptive solver's own gate. Strict mode reads the
    pending mask on the host every substep; containment (the default) never syncs."""

    _diverged_pending: wp.array | None = None
    """Adaptive solvers and the fixed-step SAP and ICF arms: per-world int32 mask, nonzero where the solver latched
    ``diverged`` in some boundary since the world's last env reset. The solver-side latch is
    transient (cleared at boundary open and, on the SAP-adaptive path, consumed same-step as
    a controller reset mask), so this mask is the signal that survives for the env layer's
    termination path (:meth:`get_diverged_env_mask`); :meth:`_reset_solver_internals` clears
    it per reset world."""

    @classmethod
    def _resolve_solver_mode(cls, solver_cfg: MJWarpSolverCfg) -> tuple[str, bool]:
        """Resolve the active backend and adaptivity from the cfg and env overrides.

        The cfg is the source of truth; ``NEWTON_SOLVER`` / ``NEWTON_SAP=1`` override the
        backend and ``NEWTON_ADAPTIVE=1`` / ``NEWTON_SAP_ADAPTIVE=1`` (or the
        ``/isaaclab/newton/adaptive`` carb setting) override adaptivity, for shell-level
        toggling without touching task configs.

        Returns:
            Tuple ``(backend, adaptive)`` where ``backend`` is ``"mujoco"``, ``"sap"`` or
            ``"icf"`` and ``adaptive`` selects the step-doubling variant of that backend.
            Each backend reads its own adaptivity latch (``adaptive`` for MuJoCo and ICF,
            ``sap_adaptive`` for SAP) because a single field would make ``--solver sap``
            and ``--solver icf-adaptive`` collide on one cfg attribute.
        """
        # Backend selection: cfg.backend is the source of truth; NEWTON_SOLVER / NEWTON_SAP=1
        # are shell-level env overrides.
        backend = str(getattr(solver_cfg, "backend", "mujoco"))
        if os.environ.get("NEWTON_SOLVER"):
            backend = os.environ["NEWTON_SOLVER"]
        if os.environ.get("NEWTON_SAP") == "1":
            backend = "sap"

        if backend == "icf":
            adaptive = bool(getattr(solver_cfg, "adaptive", False)) or os.environ.get("NEWTON_ICF_ADAPTIVE") == "1"
            return backend, adaptive

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
        * ``("icf", False)`` — ``icf_warp.SolverICF`` (fixed-step Irrotational Contact
          Fields); consumes the manager's :class:`CollisionPipeline` contacts.
        * ``("icf", True)`` — ``icf_warp.SolverICFAdaptive`` (per-world step doubling
          over ICF). It owns no contact pipeline either, so both ICF arms collide on
          the same cadence against the same contact set; the same ``IcfParams`` object
          is handed to both, so only the stepping scheme differs between them.

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
        if backend == "icf":
            # icf_warp has no installable package; add its checkout to sys.path
            # the same way the SAP backend does (override with ICF_WARP_PATH).
            import sys as _sys

            _icf_root = os.environ.get("ICF_WARP_PATH", "/home/mdigiorgio/Documents/code/icf_warp_adaptive")
            if _icf_root not in _sys.path:
                _sys.path.insert(0, _icf_root)
            from icf_warp import IcfParams, SolverICF

            # CONTACT LAW AND CONTACT CAPACITY ARE RESOLVED ONCE, ABOVE THE
            # FIXED/ADAPTIVE SPLIT, and the SAME IcfParams object reaches both
            # constructors. These two solvers are the two arms of a
            # fixed-vs-adaptive comparison; a knob read on one branch only would
            # let a shell variable change one arm's physics and not the other's,
            # and the comparison would no longer isolate timestepping.
            #
            # Contact compliance is GLOBAL in ICF (only friction is read
            # per-shape off the model), so both knobs are exposed here rather
            # than authored per-asset.
            _icf_kwargs = {}
            _k = _env("ICF_CONTACT_STIFFNESS")
            _d = _env("ICF_HC_DISSIPATION")
            if _k:
                _icf_kwargs["contact_stiffness"] = float(_k)
            if _d:
                _icf_kwargs["contact_hc_dissipation"] = float(_d)
            # Per-WORLD contact budget; contacts past it are dropped, which
            # silently changes the physics rather than failing. Buffers scale
            # as max_rigid_contact * num_envs, so it is a memory/fidelity
            # trade and must be sized from the scene's measured peak.
            _c = _env("ICF_MAX_RIGID_CONTACT")
            if _c:
                _icf_kwargs["max_rigid_contact"] = int(_c)
            _icf_params = IcfParams(**_icf_kwargs)
            if not adaptive:
                logger.info("NewtonMJWarpManager: SolverICF (fixed-step ICF convex contact) kwargs=%s", _icf_kwargs)
                return SolverICF(model, params=_icf_params)

            try:
                from icf_warp import IcfAdaptiveParams, SolverICFAdaptive
            except ImportError as exc:
                raise ImportError(
                    "--solver icf-adaptive needs IcfAdaptiveParams and SolverICFAdaptive from "
                    f"icf_warp (checkout {_icf_root!r}, override with ICF_WARP_PATH). The import "
                    f"failed with: {exc}. Use --solver icf for the fixed-step ICF arm."
                ) from exc

            # SEED = THE FIXED ARM'S STEP. The outer boundary handed to step()
            # is _solver_dt * _num_substeps, which is exactly the step the fixed
            # ICF arm takes, so seeding the controller with it makes the adaptive
            # arm start where the fixed arm sits and subdivide only where the
            # error controller demands it. solver_cfg.adaptive_dt_init is NOT
            # consulted: its default is a constant sized for a different
            # boundary, and a seed that does not equal this arm's own boundary
            # breaks that property silently.
            _icf_dt_boundary = float(cls._solver_dt) * int(cls._num_substeps)
            _icf_dt_init = float(_env("ICF_ADAPTIVE_DT_INIT", _icf_dt_boundary))
            _icf_dt_max_env = _env("ICF_ADAPTIVE_DT_MAX")
            _icf_adaptive_kwargs = {
                "mode": "adaptive",
                # tol and max_substeps are backend-agnostic controller knobs and
                # share the cfg fields the other adaptive arms read.
                "tol": float(_env("NEWTON_ADAPTIVE_TOL", getattr(solver_cfg, "adaptive_tol", 1e-3))),
                "dt_inner_init": _icf_dt_init,
                "max_substeps": int(
                    _env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))
                ),
            }
            # dt_inner_min, dt_inner_max and rtol fall through to
            # IcfAdaptiveParams' own defaults unless overridden here:
            # solver_cfg.adaptive_dt_min carries a floor chosen for the MuJoCo
            # controller, and there is no cfg field for the other two, so
            # reading them off the cfg would import a policy this controller
            # never validated.
            _icf_dt_min_env = _env("ICF_ADAPTIVE_DT_MIN")
            if _icf_dt_min_env:
                _icf_adaptive_kwargs["dt_inner_min"] = float(_icf_dt_min_env)
            if _icf_dt_max_env:
                _icf_adaptive_kwargs["dt_inner_max"] = float(_icf_dt_max_env)
            _icf_rtol_env = _env("ICF_ADAPTIVE_RTOL")
            if _icf_rtol_env:
                _icf_adaptive_kwargs["rtol"] = float(_icf_rtol_env)
            _icf_adaptive = IcfAdaptiveParams(**_icf_adaptive_kwargs)
            # IcfAdaptiveParams enforces dt_inner_min < dt_inner_init <=
            # dt_inner_max itself; re-state the boundary relation it cannot see,
            # because a seed above the boundary is clamped away every step and a
            # seed below it makes the adaptive arm start finer than the fixed one.
            if abs(_icf_adaptive.dt_inner_init - _icf_dt_boundary) > 1e-12:
                logger.warning(
                    "NewtonMJWarpManager: ICF adaptive seed dt_inner_init=%.6g s differs from the "
                    "outer boundary %.6g s (the fixed ICF arm's step). The two ICF arms no longer "
                    "start from the same step.",
                    _icf_adaptive.dt_inner_init,
                    _icf_dt_boundary,
                )
            logger.info(
                "NewtonMJWarpManager: SolverICFAdaptive (ICF step-doubling; per-world adaptive dt) "
                "params=%s adaptive=%s boundary_dt=%.6g",
                _icf_kwargs,
                _icf_adaptive,
                _icf_dt_boundary,
            )
            return SolverICFAdaptive(model, params=_icf_params, adaptive=_icf_adaptive)

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

            # Contact law and contact capacity are resolved ONCE, ABOVE the
            # fixed/adaptive split. These two solvers are the two arms of a
            # fixed-vs-adaptive comparison, so a knob read on one branch only
            # would let a shell variable change one arm's physics and not the
            # other's, and the comparison would no longer isolate timestepping.
            # Every value below must reach both constructors.
            _sap_preset = str(_env("NEWTON_SAP_PRESET", solver_cfg.sap_contact_preset))
            _sap_line_search = str(_env("NEWTON_SAP_LINE_SEARCH", solver_cfg.sap_line_search))
            _sap_solve_precision = str(
                _env("NEWTON_SAP_SOLVE_PRECISION", getattr(solver_cfg, "sap_solve_precision", "fp64"))
            )
            # SCENE-SIZED contact capacity, for both arms. The per-world contact
            # buffer drops contacts SILENTLY on overflow, and per-world demand is
            # a property of the scene, not of the integrator -- so it must be
            # derived from the task's authored global budget rather than left at
            # the small per-world default. An arm that sheds contacts under load
            # is no longer reporting anything about its timestepping.
            _sap_contact_per_world = int(solver_cfg.sap_max_rigid_contact)
            _sap_tri_pairs = 1_000_000
            _ccfg = getattr(cls, "_collision_cfg", None)
            if _ccfg is not None:
                _wc = max(int(getattr(model, "world_count", 1)), 1)
                # Clamped: global//world_count explodes at small world counts
                # while SAP's per-contact structures are far heavier than the
                # pipeline's rows.
                _sap_contact_per_world = max(
                    _sap_contact_per_world,
                    min(2048, int(_ccfg.rigid_contact_max) // _wc),
                )
                _sap_tri_pairs = _sap_triangle_pair_budget(int(_ccfg.max_triangle_pairs), _wc)

            if adaptive:
                # Error-controlled step-doubling SAP. Owns its own contact pipeline, so no
                # manager-level collision pipeline; reuses the _adaptive step/reset/no-graph
                # wiring (host-synced boundary, like SolverMuJoCoAdaptive).
                # Its internal pipeline carries the scene-sized caps resolved
                # above: both are global pooled buffers whose overflow drops mesh
                # contacts silently.
                return SolverSAPAdaptive(
                    model,
                    tol=float(_env("NEWTON_ADAPTIVE_TOL", getattr(solver_cfg, "adaptive_tol", 1e-3))),
                    dt_inner_init=float(_env("NEWTON_ADAPTIVE_DT_INIT", getattr(solver_cfg, "adaptive_dt_init", 0.01))),
                    dt_inner_min=float(_env("NEWTON_ADAPTIVE_DT_MIN", getattr(solver_cfg, "adaptive_dt_min", 1e-12))),
                    max_substeps=int(
                        _env("NEWTON_ADAPTIVE_MAX_SUBSTEPS", getattr(solver_cfg, "adaptive_max_substeps", 256))
                    ),
                    dt_histogram=str(
                        _env(
                            "NEWTON_ADAPTIVE_DT_HIST",
                            "1" if getattr(solver_cfg, "adaptive_dt_histogram", False) else "0",
                        )
                    )
                    not in ("0", "", "false", "False"),
                    max_rigid_contact=_sap_contact_per_world,
                    max_triangle_pairs=_sap_tri_pairs,
                    max_iterations=int(solver_cfg.sap_solver_iterations),
                    contact_preset_variant=_sap_preset,
                    line_search_variant=_sap_line_search,
                    contact_tau_d=float(solver_cfg.sap_contact_tau_d),
                    solve_precision=_sap_solve_precision,
                )
            # Fixed-step SAP: Newton's CollisionPipeline feeds SapContacts each step
            # (converted in _step_solver).
            sap_model = sap_model_from_newton(model)
            # SolverSAP takes the four precision knobs directly where
            # SolverSAPAdaptive takes one solve_precision string and expands it;
            # expand it the same way here so the same shell variable selects the
            # same contact-solve arithmetic on both arms. fp64 passes NO
            # overrides, leaving the preset's own defaults untouched.
            _sap_precision_kwargs: dict[str, str] = {}
            # The adaptive arm couples its optimality target to the solve
            # precision (a target below the fp32 residual floor can never be met
            # by any iteration budget); the fixed arm must resolve the SAME
            # target by the SAME rule, or the two arms stop solving the same
            # problem the moment fp32 is selected.
            _sap_optimality_rel_tol = 1.0e-8
            if _sap_solve_precision.strip().lower() in ("fp32", "f32"):
                _sap_precision_kwargs = {
                    "free_motion_solve_precision": "fp32",
                    "contact_solve_precision": "fp32",
                    "contact_linear_solve_precision": "fp32",
                    "sap_contact_weight_precision": "fp32",
                }
                from newton._src.solvers.sap.solver_sap_adaptive import _FP32_OPTIMALITY_K

                _sap_optimality_rel_tol = max(1.0e-8, float(_FP32_OPTIMALITY_K) * float(np.finfo(np.float32).eps))
            return SolverSAP(
                sap_model,
                max_rigid_contact=_sap_contact_per_world,
                max_iterations=int(solver_cfg.sap_solver_iterations),
                contact_tau_d=float(solver_cfg.sap_contact_tau_d),
                contact_preset_variant=_sap_preset,
                line_search_variant=_sap_line_search,
                # INNER-SOLVE TOLERANCES, matched to SolverSAPAdaptive's pinned
                # values. Left at SolverSAP's ctor defaults the fixed arm would
                # accept a contact-solve residual 100x looser (optimality_rel_tol
                # 1e-6) and could exit on a cost plateau (1e-30/1e-15) where the
                # adaptive arm structurally cannot -- so a difference between the
                # arms would no longer isolate timestepping, which is the only
                # thing the comparison is allowed to be about. The cost early
                # exit is disabled by zeroing both cost tolerances, matching the
                # adaptive arm's construction.
                optimality_rel_tol=_sap_optimality_rel_tol,
                cost_abs_tol=0.0,
                cost_rel_tol=0.0,
                **_sap_precision_kwargs,
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
            dt_inner_min=float(_env("NEWTON_ADAPTIVE_DT_MIN", getattr(solver_cfg, "adaptive_dt_min", 1e-12))),
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
        cls._icf = backend == "icf"
        cls._diverged_pending = None
        cls._sap_failed = None
        cls._sap_world_active = None
        cls._sap_strict = False
        if cls._adaptive:
            cls._adaptive_frame = 0
            # Sized to the per-env worlds only: the trailing global-entities
            # slot of the (world_count + 1,) reset masks has no env to
            # terminate, so it has no pending bit.
            cls._diverged_pending = wp.zeros(int(model.world_count), dtype=wp.int32, device=PhysicsManager._device)
        elif backend == "sap":
            # Fixed-step SAP gets the SEPARABLE half of the adaptive arm's
            # containment: per-world failure detection, latch, state freeze,
            # world isolation and the same termination signal. It does NOT get
            # the dt shrink-retry -- a smaller step on rejection is what
            # adaptivity IS, and handing it to the fixed arm would delete the
            # comparison rather than make it fair.
            _n = int(model.world_count)
            cls._diverged_pending = wp.zeros(_n, dtype=wp.int32, device=PhysicsManager._device)
            cls._sap_failed = wp.zeros(_n, dtype=wp.int32, device=PhysicsManager._device)
            cls._sap_world_active = wp.ones(_n, dtype=wp.int32, device=PhysicsManager._device)
            cls._sap_strict = os.environ.get("NEWTON_SAP_CONTAINMENT", "1") == "0"
        elif backend == "icf":
            # MDP parity for the fixed/adaptive ICF pair. The adaptive arm gets a
            # `physics_diverged` pathway from its own divergence latch; without a
            # mask here the fixed arm would have no such pathway at all, and the
            # two arms would differ in their termination set as well as in their
            # stepping scheme. See _certify_fixed_icf_solve for what the bit means
            # and how the two thresholds still differ.
            cls._diverged_pending = wp.zeros(int(model.world_count), dtype=wp.int32, device=PhysicsManager._device)

        if cls._sap:
            if cls._adaptive:
                NewtonManager._sap_model = None
                NewtonManager._needs_collision_pipeline = False
                logger.info(
                    "NewtonMJWarpManager: SolverSAPAdaptive (SAP step-doubling; per-world adaptive dt; "
                    "solve precision %s; solver-internal substep-body CUDA-graph capture, "
                    "set NEWTON_SAP_ADAPTIVE_GRAPH=0 to disable)",
                    getattr(NewtonManager._solver, "solve_precision", "fp64"),
                )
            else:
                # SolverSAP stores the SapModel built in _create_solver as its .model.
                NewtonManager._sap_model = getattr(NewtonManager._solver, "model", None)
                NewtonManager._needs_collision_pipeline = True
                logger.info("NewtonMJWarpManager: SolverSAP (fixed-step convex contact; CUDA graph disabled)")
            return

        if cls._adaptive and cls._icf:
            logger.info(
                "NewtonMJWarpManager: SolverICFAdaptive (ICF step-doubling; per-world adaptive dt; "
                "consumes the manager's CollisionPipeline contacts, frozen across the inner substeps "
                "of one boundary)"
            )
        elif cls._adaptive:
            logger.info(
                "NewtonMJWarpManager: SolverMuJoCoAdaptive (adaptive step-doubling; solver-internal "
                "per-iteration CUDA-graph replay, set NEWTON_MJ_ADAPTIVE_GRAPH=0 to disable)"
            )
        # Both ICF arms fall through to the cfg latch, so they consume the same
        # manager-owned contact set on the same collide cadence -- the property
        # that keeps the fixed/adaptive ICF pair a single-variable contrast.
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
                # SAP-adaptive owns its inner per-world adaptive-dt loop + its own contacts;
                # updates state_0. step() is the boundary call (Newton signature: state_in,
                # state_out, control, contacts, dt); the substep body is CUDA-graph-captured
                # INSIDE the solver and replayed per iteration, so the manager must NOT also
                # wrap the call in its own capture.
                cls._solver.step(state_0, state_1, control, contacts, substep_dt)
                # The reset below consumes the solver's divergence latch as a
                # controller reset mask AND clears it, so the latch is first
                # accumulated into the pending mask the env-side termination
                # term reads via get_diverged_env_mask -- otherwise the latch
                # is gone before any termination term can see it.
                if cls._diverged_pending is not None:
                    wp.launch(
                        _accumulate_diverged_pending,
                        dim=cls._diverged_pending.shape[0],
                        inputs=[cls._solver.diverged, cls._diverged_pending],
                        device=cls._diverged_pending.device,
                    )
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
                cls._solver.step(s0, s0, c, sc, substep_dt, world_active=cls._sap_world_active)
                cls._certify_fixed_sap_solve()
            return
        # MuJoCo fixed and adaptive share the stock 5-positional call; for the adaptive
        # solver step() is the boundary call (it owns its inner step-doubling loop + its
        # own contacts).
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        if cls._icf and not cls._adaptive:
            cls._certify_fixed_icf_solve()
        if cls._adaptive and cls._diverged_pending is not None:
            # Same pending accumulation as the SAP-adaptive branch, so both
            # adaptive backends report divergence through the one
            # manager-owned mask the termination term reads (the
            # MuJoCo-adaptive latch itself persists until a reset clears it,
            # but no termination term reads solver internals directly).
            wp.launch(
                _accumulate_diverged_pending,
                dim=cls._diverged_pending.shape[0],
                inputs=[cls._solver.diverged, cls._diverged_pending],
                device=cls._diverged_pending.device,
            )

    @classmethod
    def _certify_fixed_sap_solve(cls) -> None:
        """Consume the fixed-step SAP arm's per-env convergence result.

        The solve's own status flag is not usable for this: ``SolverSAP`` reports
        ``last_converged`` from a result the contact solve constructs with the
        literal ``True``, so it carries no information. The decision lives per-env
        in ``contact_solve.converged_env``, which is the array the adaptive solver
        folds into its solve-ok state and turns into a rejection -- so both arms
        certify the same quantity by the same arithmetic, and neither commits a
        solve that never converged without saying so.

        Containment (default) latches, freezes and reports on device with no host
        sync. Strict mode (``NEWTON_SAP_CONTAINMENT=0``, the same gate the
        adaptive solver reads) raises instead, which costs one host read per
        substep and is therefore opt-in.
        """
        if cls._diverged_pending is None or cls._sap_world_active is None:
            return
        contact_solve = getattr(cls._solver, "contact_solve", None)
        converged_env = getattr(contact_solve, "converged_env", None)
        if converged_env is None:
            return
        n = cls._diverged_pending.shape[0]
        if converged_env.shape[0] != n:
            raise RuntimeError(
                "NewtonMJWarpManager: fixed-step SAP convergence certificate expects one "
                f"contact_solve env per world, got {converged_env.shape[0]} for {n} worlds."
            )
        wp.launch(
            _latch_sap_solve_failure,
            dim=n,
            inputs=[converged_env, cls._sap_world_active, cls._sap_failed, cls._diverged_pending],
            device=cls._diverged_pending.device,
        )
        if cls._sap_strict and int(cls._diverged_pending.numpy().sum()) > 0:
            raise RuntimeError(
                "SolverSAP inner SAP solve failed to converge to "
                f"optimality_rel_tol={float(cls._solver.optimality_rel_tol):.3e}."
            )

    @classmethod
    def _certify_fixed_icf_solve(cls) -> None:
        """Consume the fixed-step ICF arm's per-env convergence result.

        ``SolverICF.last_converged`` is not usable for this: it is host-side
        bookkeeping, not the per-env decision. That decision lives in
        ``contact_solve.converged_env``, which is the array the adaptive ICF arm
        folds into its own solve-ok state, so both arms certify the same quantity
        by the same arithmetic.

        The two arms' divergence thresholds are NOT identical and must not be
        reported as such: the adaptive arm rejects a failed solve and retries at a
        smaller dt, latching ``diverged`` only when it is still failing at the dt
        floor, while the fixed arm has no retry and latches on the first failure.
        What this makes equal is the EXISTENCE of the pathway, not its trigger point.
        """
        if cls._diverged_pending is None:
            return
        contact_solve = getattr(cls._solver, "contact_solve", None)
        converged_env = getattr(contact_solve, "converged_env", None)
        if converged_env is None:
            return
        n = cls._diverged_pending.shape[0]
        if converged_env.shape[0] != n:
            raise RuntimeError(
                "NewtonMJWarpManager: fixed-step ICF convergence certificate expects one "
                f"contact_solve env per world, got {converged_env.shape[0]} for {n} worlds."
            )
        wp.launch(
            _latch_icf_solve_failure,
            dim=n,
            inputs=[converged_env, cls._diverged_pending],
            device=cls._diverged_pending.device,
        )

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """Adaptive: march the whole control period in one boundary call.

        The adaptive solver (MuJoCo or SAP step-doubling) is itself the substepper
        (error-controlled inner dt), and control is constant across the decimation tick
        (actuators run once per tick, before this call), so one boundary call per tick
        suffices. Fixed-step paths keep the stock per-substep loop.
        """
        # NEWTON_ADAPTIVE_SINGLE_BOUNDARY=0 routes the adaptive solvers through the
        # stock per-substep loop instead: shorter boundaries mean injected contacts
        # are re-detected num_substeps times per tick, bounding how long the march
        # integrates against a frozen contact set.
        if cls._adaptive and os.environ.get("NEWTON_ADAPTIVE_SINGLE_BOUNDARY", "1") == "1":
            cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt * cls._num_substeps)
            cls._state_0.clear_forces()
            return
        super()._run_solver_substeps(contacts)

    @classmethod
    def _apply_fixed_sap_pipeline_overrides(cls) -> None:
        """Resolve the manager collision pipeline the way the SAP-adaptive arm resolves its own.

        The fixed-step SAP arm does not own a pipeline; it consumes the
        manager's, which is built from the task's authored
        :class:`NewtonCollisionPipelineCfg`. The adaptive arm builds its pipeline
        itself and resolves two things there that the authored cfg does not
        carry, so without this the two arms run different collision pipelines
        for reasons that have nothing to do with timestepping:

        * ``NEWTON_SAP_DETERMINISTIC`` — canonical post-narrow-phase contact
          sort. Reaching one arm and not the other makes a determinism
          comparison between the arms meaningless.
        * the triangle-pair budget — sized to the scene rather than left at
          whatever the cfg authored for a different world count.

        Both are applied to a COPY, gated on the fixed SAP arm: the cfg object
        is shared with the MuJoCo backend, whose established trajectories a
        determinism flip would move.
        """
        if not (cls._sap and not cls._adaptive):
            return
        ccfg = getattr(cls, "_collision_cfg", None)
        if ccfg is None:
            return
        deterministic = os.environ.get("NEWTON_SAP_DETERMINISTIC", "0") == "1"
        world_count = max(int(getattr(cls._model, "world_count", 1)), 1) if cls._model is not None else 1
        pairs = _sap_triangle_pair_budget(int(ccfg.max_triangle_pairs), world_count)
        if deterministic:
            pairs = _clamp_deterministic_triangle_pairs(pairs)
        if bool(ccfg.deterministic) == deterministic and int(ccfg.max_triangle_pairs) == pairs:
            return
        resolved = copy.deepcopy(ccfg)
        resolved.deterministic = deterministic
        resolved.max_triangle_pairs = pairs
        NewtonManager._collision_cfg = resolved
        logger.info(
            "NewtonMJWarpManager: fixed-step SAP collision pipeline resolved to deterministic=%s, "
            "max_triangle_pairs=%d (task authored %d) to match the SAP-adaptive arm's own pipeline.",
            deterministic,
            pairs,
            int(ccfg.max_triangle_pairs),
        )

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
            cls._apply_fixed_sap_pipeline_overrides()
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
          contact-solve warm-start is one global flag, cleared via
          ``reset_runtime_state_masked()`` when the mask flags any world.  With
          staggered per-env resets, untouched envs pay a small re-convergence
          cost, measured to be dynamically negligible (see
          ``test_mimic_state_seam.py``).  The manager's own per-world containment
          state — the solve-failure latch, the participation mask and the pending
          divergence bit — is cleared here for the reset worlds only.

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
            if cls._sap_failed is not None:
                # A reset world's failure latch is consumed here for the same
                # reason the adaptive arm's is: the env rebuilt the world's
                # state, so the failing solve is over and the world must both
                # rejoin the active set and stop re-terminating a fresh episode.
                wp.launch(
                    _release_sap_failed,
                    dim=cls._sap_failed.shape[0],
                    inputs=[world_mask, cls._sap_failed, cls._sap_world_active],
                    device=cls._sap_failed.device,
                )
                wp.launch(
                    _clear_diverged_pending,
                    dim=cls._diverged_pending.shape[0],
                    inputs=[world_mask, cls._diverged_pending],
                    device=cls._diverged_pending.device,
                )
            # Device-side equivalent of "reset the warm start if any world
            # reset": the host test it replaces was a full device sync on every
            # reset boundary, and most boundaries flag no world at all.
            cls._solver.reset_runtime_state_masked(local_mask)
            return
        if cls._adaptive:
            if cls._diverged_pending is not None:
                # A reset world's pending divergence is consumed here: the env
                # rebuilds the world's state, so the diverged episode the mask
                # reported is over and the fresh episode starts clean.
                wp.launch(
                    _clear_diverged_pending,
                    dim=cls._diverged_pending.shape[0],
                    inputs=[world_mask, cls._diverged_pending],
                    device=cls._diverged_pending.device,
                )
            cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)
            return
        if cls._icf and cls._diverged_pending is not None:
            # Same consumption rule as the other arms: the env rebuilt this
            # world's state, so the solve that failed is no longer the state
            # being stepped and the fresh episode must not re-terminate on it.
            wp.launch(
                _clear_diverged_pending,
                dim=cls._diverged_pending.shape[0],
                inputs=[world_mask, cls._diverged_pending],
                device=cls._diverged_pending.device,
            )
        if getattr(cls._solver, "use_mujoco_cpu", False) and not local_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def invalid_env_mask(cls):
        """Envs whose world the adaptive march abandoned short of the boundary.

        With a quantile stop the march ends once the active set has fallen to
        its cutoff rather than waiting for the last straggler, so the remaining
        worlds never reached the step boundary. Their state is mid-step and is
        not a transition, so the env layer resets them -- for every task, with
        no task-side term to declare.

        ``None`` when no quantile stop is engaged on the active solver.
        """
        mask = getattr(cls._solver, "boundary_cut_mask", None)
        if mask is None:
            return None
        import warp as wp

        out = wp.to_torch(mask) != 0
        cls._solver.clear_boundary_cuts()
        return out

    @classmethod
    def get_diverged_env_mask(cls):
        """Per-env divergence mask accumulated since each env's last reset.

        Returns an int32 torch view (zero-copy) of the pending mask, nonzero
        where the env's world latched the adaptive solver's ``diverged`` flag
        in some boundary since that env last reset: a NaN/divergent state, or
        (SAP containment) an inner solve still failing at the dt floor. A
        latched world held its last committed finite state while its clock
        skipped to the boundary, so no state-space check can detect it -- this
        mask is the only signal the env layer gets, and a termination term
        must consume it for the world to be recovered by an env reset (which
        clears the env's bit via :meth:`_reset_solver_internals`).

        On the fixed-step SAP arm the same mask carries the inner solve's
        convergence certificate: a world whose contact solve did not converge is
        latched, frozen and reported here, so the ``physics_diverged`` term ends
        that episode instead of training on a solve that never converged. The
        fixed-step ICF arm reports the same certificate without the freeze
        (``SolverICF.step`` takes no participation mask), which is what gives the
        fixed and adaptive ICF arms the same termination pathway. The MuJoCo fixed
        arm allocates no mask and still returns ``None``.

        Returns ``None`` when the active solver has no divergence signal.
        """
        if cls._diverged_pending is None:
            return None
        return wp.to_torch(cls._diverged_pending)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active solver configuration supports CUDA graph capture.

        MANAGER-level capture stays OFF for SAP (owns its capture internally) and ON for
        both ICF arms (ICF allocates nothing per step and the adaptive ICF march records as
        a conditional while-node rather than its own capture). For the
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
        if cls._icf:
            # Both ICF arms are manager-capturable. The adaptive one records its
            # data-dependent boundary march as a ``wp.capture_while`` conditional
            # node and opens no capture of its own, and ICF allocates nothing per
            # step, so the gate below (written for mujoco_warp's per-step scratch
            # allocations) does not apply. Without this the adaptive ICF arm would
            # run launch-by-launch against a captured fixed ICF arm and any wall-time
            # comparison between them would be measuring launch overhead.
            return True
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
            # Demand + engagement counters exist on the SAP-adaptive solver
            # only; appended AFTER cumulative_substeps so existing parsers
            # keep matching. cumulative_accepted is the schedule-invariant
            # per-world work axis (accepted substeps), which makes matched-
            # demand wall comparisons readable from this file alone.
            extra = ""
            acc_fn = getattr(cls._solver, "cumulative_accepted_steps", None)
            if acc_fn is not None:
                extra += f" cumulative_accepted={int(acc_fn())}"
            eng_fn = getattr(cls._solver, "runahead_engagement", None)
            if eng_fn is not None:
                ra_cross, ra_fires = eng_fn()
                extra += f" ra_cross={int(ra_cross)} ra_fires={int(ra_fires)}"
            path = os.environ.get("NEWTON_ADAPTIVE_LOG", "/tmp/newton_adaptive.log")
            with open(path, "a") as f:
                f.write(
                    f"frame={cls._adaptive_frame} inner_dt[min={dt.min():.3e} max={dt.max():.3e} "
                    f"spread={float(dt.max() - dt.min()):.3e}] cumulative_substeps={subs}{extra}\n"
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
        if cfg is not None and cfg.debug_mode and not cls._sap and not cls._icf:  # type: ignore[union-attr]
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
