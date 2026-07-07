Fixed
^^^^^

* Fixed the lab table height: the working surface was authored at 0.289 m —
  the real table's 28.9 in with a missed unit conversion — putting the
  tabletop at the G1's shins. ``LAB_TABLE_HEIGHT`` is now 0.734 m (28.9 in);
  consumers that derive positions from it pick up the correction
  automatically.
