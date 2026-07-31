Changed
^^^^^^^

* **Breaking:** Reworked ``IsaacContrib-Pick-Cube-G1-v0``'s config to the minimal
  pick-and-hold structure (single clipped joint action term with the full waist
  enabled, dense reach/close/lift/carry rewards, success termination at a fixed
  chest hold point, no domain randomization) and removed the untracked
  ``g1_dish_rack`` task. Users of the previous reward/termination term names
  should re-resolve their configs against the new ``G1PickCubeEnvCfg``.
