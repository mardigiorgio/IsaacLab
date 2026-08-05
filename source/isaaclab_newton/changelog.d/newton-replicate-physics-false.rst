Fixed
^^^^^

* Fixed Newton environments with ``replicate_physics=False`` (e.g. the stack-cube Mimic
  tasks) failing at solver construction with every body assigned to the global world:
  :class:`~isaaclab_newton.cloner.NewtonReplicateContext` now declares
  ``builds_physics_model`` so scene replication keeps building the Newton model even when
  physics replication is disabled.
