Fixed
^^^^^

* Fixed :attr:`~isaaclab.sim.schemas.RigidBodyBaseCfg.disable_gravity` being silently ignored
  per body on the Newton MuJoCo-Warp backend (fixed-step and adaptive). It is now mapped onto
  MuJoCo-Warp's per-body ``gravcomp`` mechanism before the solver is constructed, so gravity-free
  bodies (e.g. ``FRANKA_PANDA_HIGH_PD_CFG``'s arm) no longer sag under gravity. The SAP backend
  has no per-body gravity mechanism; a scene with ``disable_gravity`` bodies now logs an
  actionable warning on SAP instead of silently sagging those bodies.
