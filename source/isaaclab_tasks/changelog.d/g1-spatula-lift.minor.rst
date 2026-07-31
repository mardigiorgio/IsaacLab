Added
^^^^^

* Added contrib task ``IsaacContrib-Lift-Spatula-G1-v0``: a free-standing Unitree G1
  balancing at the lab table picks up an LBM wooden spatula BY THE HANDLE with the right
  TriHand on a READY→GO command and holds it at a fixed objective. The spatula asset is
  authored with separate handle/blade collision prims so shape-filtered contact sensors
  gate all grasp rewards on handle contact (dexsuite-lift reward structure), the curated
  reset-state (grasp map) event is enabled to bootstrap grasp learning, and falls,
  trunk-on-table leaning, and behind-the-body reaches terminate.
