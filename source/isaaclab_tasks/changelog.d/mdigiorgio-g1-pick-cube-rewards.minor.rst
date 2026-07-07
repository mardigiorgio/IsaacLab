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

* **Breaking:** Changed the G1 pick-cube task geometry: the lab table is now
  at its real hip height (the asset previously encoded a missed inch-to-meter
  conversion and sat at the G1's shins), the cube is a standard Rubik's-cube
  size (5.7 cm), and the legs are held posed by stiff implicit actuators.
  Removed ``ee_to_cube_distance_reward`` in favor of the palm-based reward
  terms (migrate by using ``palms_to_cube_distance_reward`` with the palm
  body names).
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
