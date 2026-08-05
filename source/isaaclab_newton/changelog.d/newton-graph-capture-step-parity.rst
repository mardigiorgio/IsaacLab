Fixed
^^^^^

* Fixed state writes being silently discarded under a Kit visualizer when the MuJoCo-Warp solver
  ran with ``update_data_interval`` greater than 1. The pre-capture warm-up step advanced the
  solver's internal step counter, so the CUDA graph was recorded on the opposite parity and the
  Newton-state to MuJoCo ``qpos`` sync was baked out of the replayed graph. Every subsequent
  teleport -- including :func:`~isaaclab.envs.mdp.reset_scene_to_default`,
  :func:`~isaaclab.envs.mdp.reset_root_state_uniform` and
  :func:`~isaaclab.envs.mdp.reset_joints_by_offset` -- was then dropped, while the readback still
  reported success because it views the write buffer. The warm-up now preserves the step counter.
