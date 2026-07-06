Fixed
^^^^^

* Fixed :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.max_num_failures` being read nowhere in
  the datagen pipeline. :func:`~isaaclab_mimic.datagen.generation.env_loop` now terminates a
  ``generation_guarantee=True`` run once ``num_failures`` reaches
  :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.max_num_failures`, instead of retrying forever
  on a task that never succeeds. Runs with ``generation_guarantee=False`` are unaffected; their
  attempt count already bounds the run.
