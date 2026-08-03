Changed
^^^^^^^

* Changed the ``IsaacContrib-Lift-Spatula-G1-v0`` reward set to the ungated lift recipe.
  ``reach``, ``lift`` and ``track`` replace ``fingers_to_handle``, ``contact_count``,
  ``lifted`` and ``hold_at_point``; none of them gate on opposed thumb/finger contact,
  which is a pinch signature the intended claw grip does not produce. Migration: the
  retired functions remain exported from the task's ``mdp`` module and can be restored
  by name.
* Changed the ``IsaacContrib-Lift-Spatula-G1-v0`` PPO config to cap the policy standard
  deviation at 0.5 (``std_range=(0.15, 0.5)``, ``init_std=0.4``) with ``entropy_coef=0.0``,
  and shortened the horizon to ``gamma=0.98`` over 1500 iterations.

Added
^^^^^

* Added ``object_lifted``, ``track_carry_point`` and ``joint_vel_l2_clamped`` to the
  ``IsaacContrib-Lift-Spatula-G1-v0`` task's ``mdp`` module, and made
  ``fingers_to_handle``'s contact gate optional via ``contact_threshold=None``.
