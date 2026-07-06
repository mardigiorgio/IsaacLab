Changed
^^^^^^^

* Changed :attr:`~isaaclab_newton.physics.NewtonCfg.enforce_velocity_limit` to default ``True``,
  rate-limiting implicit-actuator joint position targets to each joint's configured drive velocity
  limit (:attr:`~isaaclab_newton.assets.ArticulationData.joint_vel_limits`), matching PhysX's
  native drive behavior. Newton's MuJoCo-Warp solver explicitly drops the joint velocity limit
  and the vendored SAP solver has no velocity-limit concept at all, so a stiff implicit-PD joint
  commanded a large position step could otherwise move several times faster than its configured
  limit. Set :attr:`~isaaclab_newton.physics.NewtonCfg.enforce_velocity_limit` to ``False`` to
  restore the previous (unclamped) behavior.
