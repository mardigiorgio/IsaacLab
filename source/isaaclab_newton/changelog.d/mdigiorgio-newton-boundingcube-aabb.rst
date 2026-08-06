Fixed
^^^^^

* Fixed USD ``physics:approximation = "boundingCube"`` colliders importing as inertia-aligned
  bounding boxes in the Newton cloner. For meshes with near-isotropic inertia (e.g. the nucleus
  block assets used by the stack tasks), the inertia-OBB fit's principal axes are numerically
  degenerate, so the imported box came out arbitrarily rotated (~6 deg lean at rest) and up to
  ~10% inflated — raising the blocks' rest height and their stacked center-to-center height past
  the stack tasks' success tolerance. The collider is now the mesh's local-frame axis-aligned
  bounding box, matching PhysX's ``boundingCube`` semantics.
