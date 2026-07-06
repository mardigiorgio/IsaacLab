Changed
^^^^^^^

* Changed the stack tasks' ``newton_mjwarp`` physics preset to author stiff
  cube/gripper contact (``ke=1e6, kd=2000`` -> solref (0.001 s, 1.0)) instead
  of inheriting MuJoCo-Warp's default 20 ms compliance. At the stacking
  gripper's ~40 N pinch force the previous compliance let the finger pads
  sink ``F/ke`` ~= 1.6 cm into a cube, so a nominally closed grasp could
  still let the cube squirt out. Arm and finger PD gains are unchanged;
  Newton's per-body gravity compensation and drive velocity-limit clamping
  now let the stock high-PD gains hold as originally authored.
