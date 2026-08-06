Fixed
^^^^^

* Fixed zero-success mimic data generation on the Newton MuJoCo backend for the stack tasks
  (``Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`` and siblings). The ``newton_mjwarp`` preset's
  contact stiffness (``ke=1e6/kd=2000``) authored a MuJoCo contact ``solref`` time constant of
  1 ms at a 5 ms substep — an unstable finger-cube contact whose grip chatter spun grasped
  cubes until they were ballistically ejected. The preset now uses ``ke=4e4/kd=400`` at
  ``num_substeps=4`` (the ``solref`` stability boundary), the stacking cubes author
  ``mjc:condim=6`` with raised Newton torsional friction so pinched cubes cannot spin out of
  the grasp, and the cube reset height no longer interpenetrates the table under Newton's
  collision geometry (PhysX behavior is unchanged: the ``mjc:*``/``newton:*`` attributes are
  ignored by PhysX, the UsdPhysics material values mirror the DexCube asset's own, and cubes
  settle to the same rest pose after the reset drop).
