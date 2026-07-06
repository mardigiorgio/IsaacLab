Added
^^^^^

* Added regression tests for the mimic state-restore seam (``scene.reset_to`` mid-episode,
  staggered per-env resets) across all four MJWarp solver modes.

Changed
^^^^^^^

* Documented the fixed-step SAP global contact warm-start clear on env resets in
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg`.

Fixed
^^^^^

* Fixed a ``cudaErrorIllegalAddress`` crash on repeated environment resets under the Newton
  MuJoCo-Warp backend (e.g. the mimic annotation replay loop's repeated ``sim.reset()`` calls).
  :meth:`~isaaclab_newton.physics.NewtonManager.reset` now drops references to the previous CUDA
  graph, model, state, solver, actuator adapter, and vendored SAP model and forces a garbage
  collection and device sync before rebuilding, instead of relying on the cyclic garbage
  collector to free them at an arbitrary later point.
