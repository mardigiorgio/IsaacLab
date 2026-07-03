Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonShapeCfg.ke` and
  :attr:`~isaaclab_newton.physics.NewtonShapeCfg.kd` default contact
  stiffness/damping fields, forwarded to Newton's ``ShapeConfig``. On the
  MuJoCo-Warp backend these convert to contact ``solref`` (both must be
  positive); the defaults reproduce MuJoCo's default compliance, so existing
  environments are unaffected.
