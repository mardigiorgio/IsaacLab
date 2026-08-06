Changed
^^^^^^^

* Changed the ``IsaacContrib-Lift-Spatula-G1-v0`` physics preset default from
  PhysX to Newton MJWarp and removed the PhysX preset: the task's
  shape-filtered contact recipe cannot be expressed under PhysX, so the PhysX
  preset always failed at environment construction. Select the backend
  explicitly with ``presets=newton_mjwarp`` (unchanged) or rely on the default.
