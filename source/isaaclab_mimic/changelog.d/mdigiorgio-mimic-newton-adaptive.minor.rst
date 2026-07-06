Added
^^^^^

* Added :mod:`~isaaclab_mimic.datagen.mimic_recorders` with the datagen-info and
  subtask-signal recorder terms (moved out of the annotation script) for reuse at teleop
  record time.
* Added ``--physics_preset`` / ``--solver`` support to the annotation, dataset generation, and
  teleop recording scripts, ``--retries`` with a yield report to annotation, and
  ``--record_subtask_signals`` to teleop recording, enabling the full mimic pipeline on the
  Newton backend.
* Added an end-to-end Newton smoke test and a datagen fidelity/throughput benchmark across
  the four MJWarp solver modes.

Fixed
^^^^^

* Fixed :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.max_num_failures` being read nowhere in
  the datagen pipeline. :func:`~isaaclab_mimic.datagen.generation.env_loop` now terminates a
  ``generation_guarantee=True`` run once ``num_failures`` reaches
  :attr:`~isaaclab.envs.mimic_env_cfg.DataGenConfig.max_num_failures`, instead of retrying forever
  on a task that never succeeds. Runs with ``generation_guarantee=False`` are unaffected; their
  attempt count already bounds the run.
* Fixed the annotation retry loop treating any solver error as a retryable failed attempt,
  which masked CUDA illegal-memory-access and out-of-memory errors instead of aborting the run.
  Only the documented adaptive-solver convergence-failure signature is now retried; every other
  error propagates.
