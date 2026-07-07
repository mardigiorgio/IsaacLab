Added
^^^^^

* Added staged pick-up rewards to ``IsaacContrib-Pick-Cube-G1-v0``: coarse and
  fine palm reaching, finger closing gated to palm-near-cube, dense lift
  progress, and a hold bonus at the success height, replacing the placeholder
  wrist-distance shaping. New MDP terms
  :func:`~isaaclab_tasks.contrib.g1_pick_cube.mdp.palms_to_object_vector`,
  :func:`~isaaclab_tasks.contrib.g1_pick_cube.mdp.palms_to_cube_distance_reward`,
  :func:`~isaaclab_tasks.contrib.g1_pick_cube.mdp.fingers_closed_near_cube` and
  :func:`~isaaclab_tasks.contrib.g1_pick_cube.mdp.object_lift_progress`.

Changed
^^^^^^^

* **Breaking:** Changed the G1 pick-cube base pose to a squat (pelvis lowered
  to 0.55 m, legs posed bent and held by stiff implicit actuators) and moved
  the cube spawn to the robot-side table edge. A reachability study showed the
  standing pose cannot bring a palm closer than ~0.26 m to the cube, so
  policies trained against the old geometry were unable to reach it. Removed
  ``ee_to_cube_distance_reward`` in favor of the palm-based reward terms
  (migrate by using ``palms_to_cube_distance_reward`` with the palm body
  names).
* **Breaking:** Removed the ``success`` termination from the G1 pick-cube task
  and renamed the ``lifted`` reward term to ``lifted_hold``: terminating on
  lift while paying dense height rewards teaches hovering below the threshold.
  Detect success by monitoring the ``lifted_hold`` reward term (positive while
  the cube is held above the success height).
* Changed the G1 pick-cube arm and hand actuators to hardware-plausible effort
  limits (60 N·m arms, 5 N·m fingers) so arbitrary policy actions cannot
  destabilize the solver, and clipped the raw joint-position actions. Restore
  the asset values by overriding ``scene.robot.actuators`` and
  ``actions.upper_body.clip`` in a derived environment config.

Fixed
^^^^^

* Fixed the G1 pick-cube environment never re-applying the robot joint init
  state on reset: on the Newton backend the articulation restarted every
  episode at the zero pose instead of the authored default pose.
