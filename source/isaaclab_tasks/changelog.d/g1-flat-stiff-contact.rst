Changed
^^^^^^^

* Changed the ``Isaac-Velocity-Flat-G1-v0`` Newton preset to author stiff
  foot-ground contact (``ke=1e6, kd=2000`` -> solref 1 ms) instead of
  inheriting MuJoCo's default 20 ms compliance, which let feet transiently
  sink 1-2 cm at footfall. Policies trained on the previous compliance can
  be retrained with the same command; no API change.
