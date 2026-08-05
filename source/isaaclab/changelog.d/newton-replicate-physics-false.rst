Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.replicate_session.replicate` dropping physics replication
  contexts that author the backend's physics model when ``replicate_physics=False``.
  Contexts declaring ``builds_physics_model = True`` (e.g. Newton's) are now kept, since
  such backends have no other way to obtain a physics model for ``/World/envs/env_<id>``
  scenes.
