Added
^^^^^

* Added the ``IsaacContrib-Pick-Cube-G1-v0`` skeleton task: fixed-base G1 at the lab table
  with a DexCube, joint-space actions, placeholder rewards, an RSL-RL PPO starting config,
  and a ``newton_mjwarp`` physics preset.

Fixed
^^^^^

* Fixed the physics-preset application in the environment test utilities: ``physics_preset_name``
  was silently ignored (tests ran PhysX regardless); Newton preset tests now genuinely exercise
  the Newton backend.
