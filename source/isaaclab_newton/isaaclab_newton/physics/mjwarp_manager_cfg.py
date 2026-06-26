# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class MJWarpSolverCfg(NewtonSolverCfg):
    """Configuration for MuJoCo Warp solver-related parameters.

    These parameters are used to configure the MuJoCo Warp solver. For more information, see the
    `MuJoCo Warp documentation`_.

    .. _MuJoCo Warp documentation: https://github.com/google-deepmind/mujoco_warp
    """

    class_type: type[NewtonManager] | str = "{DIR}.mjwarp_manager:NewtonMJWarpManager"
    """Manager class for the MuJoCo Warp solver."""

    solver_type: str = "mujoco_warp"
    """Solver type. Can be "mujoco_warp"."""

    njmax: int = 300
    """Number of constraints per environment (world)."""

    nconmax: int | None = None
    """Number of contact points per environment (world)."""

    iterations: int = 100
    """Number of solver iterations."""

    ls_iterations: int = 50
    """Number of line search iterations for the solver."""

    solver: str = "newton"
    """Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants."""

    integrator: str = "euler"
    """Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants."""

    use_mujoco_cpu: bool = False
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`."""

    disable_contacts: bool = False
    """Whether to disable contact computation in MuJoCo."""

    default_actuator_gear: float | None = None
    """Default gear ratio for all actuators."""

    actuator_gears: dict[str, float] | None = None
    """Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`."""

    update_data_interval: int = 1
    """Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.

    If 0, Data is never updated after initialization.
    """

    save_to_mjcf: str | None = None
    """Optional path to save the generated MJCF model file.

    If None, the MJCF model is not saved.
    """

    impratio: float = 1.0
    """Frictional-to-normal constraint impedance ratio."""

    cone: str = "pyramidal"
    """The type of contact friction cone. Can be "pyramidal" or "elliptic"."""

    ccd_iterations: int = 35
    """Maximum iterations for convex collision detection (GJK/EPA).

    Increase this if you see warnings about ``opt.ccd_iterations`` needing to be increased,
    which typically occurs with complex collision geometries (e.g. multi-finger hands).
    """

    ls_parallel: bool = False
    """Deprecated parallel line search option.

    Setting this to ``True`` emits a :class:`DeprecationWarning` and is ignored.
    MuJoCo Warp is dropping support for parallel line search; Isaac Lab uses
    iterative line search for performance.
    """

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's internal contact solver.

    If ``True`` (default), MuJoCo handles collision detection and contact resolution internally.
    If ``False``, Newton's :class:`CollisionPipeline` is used instead.  A default pipeline
    (``broad_phase="explicit"``) is created automatically when :attr:`NewtonCfg.collision_cfg`
    is ``None``.  Set :attr:`NewtonCfg.collision_cfg` to a :class:`NewtonCollisionPipelineCfg`
    to customize pipeline parameters (broad phase, contact limits, hydroelastic, etc.).

    .. note::
        Setting ``collision_cfg`` while ``use_mujoco_contacts=True`` raises
        :class:`ValueError` because the two collision modes are mutually exclusive.
    """

    tolerance: float = 1e-6
    """Solver convergence tolerance for the constraint residual.

    The solver iterates until the residual drops below this threshold or
    ``iterations`` is reached.  Lower values give more precise constraint
    satisfaction at the cost of more iterations.  MuJoCo default is ``1e-8``;
    Newton default is ``1e-6``.
    """

    adaptive: bool = False
    """Use the error-controlled adaptive solver :class:`~newton.solvers.SolverMuJoCoAdaptive` (step-doubling)
    instead of the fixed-step :class:`~newton.solvers.SolverMuJoCo`.

    When ``True``, the manager drives the solver via ``step_dt`` once per substep (the solver owns its inner
    dt loop and its own contact pipeline) and CUDA-graph capture is disabled (the per-frame substep count is
    data-dependent). The remaining MuJoCo-Warp fields above are forwarded to the underlying solver.
    """

    adaptive_tol: float = 1e-3
    """Adaptive only: inf-norm ``joint_q`` error tolerance per world [m or rad]."""

    adaptive_dt_mode: str = "per_world"
    """Adaptive only: ``"per_world"`` (each env adapts its own dt) or ``"global"`` (shared worst-case dt)."""

    adaptive_dt_init: float = 0.01
    """Adaptive only: initial inner timestep [s]. Set below ``sim.dt`` to give the controller room to subdivide."""

    adaptive_dt_min: float = 1e-6
    """Adaptive only: minimum inner timestep [s] (must be < ``adaptive_dt_init``)."""

    adaptive_tiling: str = "ragged"
    """Adaptive only: ``"ragged"`` (legacy adaptive dt with a clamped remainder landing) or
    ``"even"`` (Fix 4: choose substep count N from the carried ideal_dt, then tile each control
    interval with a uniform inner dt = dt_outer/N -- removes the within-interval raggedness while
    keeping adaptivity across control steps). Default ``"ragged"`` for backward-compat."""

    adaptive_max_substeps: int = 256
    """Adaptive ``"even"`` tiling only: hard upper bound on the substep count N per control interval.
    Bounds worst-case work when a world's ideal_dt collapses to ``adaptive_dt_min`` (uncapped N would
    be ~dt_outer/dt_min ~ 1e4 and effectively hang the per-world fixed loop). Default 256 only bounds
    runaway; normal motion needs N ~ 1-60."""

    backend: str = "mujoco"
    """Inner physics backend: ``"mujoco"`` (MuJoCo-Warp, default) or ``"sap"`` (the vendored convex SAP
    contact solver :class:`~newton.solvers.SolverSAP`). Overridable at runtime via ``NEWTON_SOLVER`` or
    ``NEWTON_SAP=1``. With ``"sap"`` the MuJoCo-Warp fields above are ignored; use the ``sap_*`` fields."""

    sap_adaptive: bool = False
    """SAP only: drive the error-controlled step-doubling controller over SAP
    (:class:`~newton.solvers.SolverSAPAdaptive`, even+global tiling) instead of fixed-step
    :class:`~newton.solvers.SolverSAP`. Overridable via ``NEWTON_SAP_ADAPTIVE=1``. Fixed-step SAP is the
    RL-ideal (stable+consistent+fast); the adaptive layer adds error control for accuracy/dataset-gen."""

    sap_max_rigid_contact: int = 128
    """SAP only: per-world rigid-contact capacity (contact storage = this * num_envs)."""

    sap_solver_iterations: int = 30
    """SAP only: max SAP contact-solve iterations. Must be high enough to converge at the chosen dt
    (adaptive asserts convergence for a consistent step-doubling error estimate)."""

    sap_contact_preset: str = "approx32"
    """SAP only: contact/Jacobian precision preset (``"approx32"`` | ``"approx64"`` | ``"drake"``).

    Default ``"approx32"`` (f32 Jacobians + f32 contact linear solve): ~1.5x faster than ``"drake"``
    (f64) on contact scenes with no measurable loss of step-doubling stability. Set ``"drake"`` for
    full-f64 reference accuracy."""

    sap_line_search: str = "armijo_decay"
    """SAP only: line-search variant (``"monotone_decay"`` | ``"armijo_decay"`` | ``"exact_root"``)."""

    sap_contact_tau_d: float = 0.01
    """SAP only: fallback dissipation time scale [s]."""

    def __post_init__(self):
        if self.ls_parallel:
            warnings.warn(
                "MJWarpSolverCfg.ls_parallel is deprecated and ignored. "
                "Isaac Lab uses iterative line search for performance because "
                "MuJoCo Warp is dropping parallel line search support. Tune "
                "MJWarpSolverCfg.ls_iterations instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            self.ls_parallel = False
