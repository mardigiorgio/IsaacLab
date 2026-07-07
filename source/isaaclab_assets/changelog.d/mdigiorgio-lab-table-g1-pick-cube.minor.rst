Added
^^^^^

* Added :mod:`~isaaclab_assets.props.lab_table` with :func:`~isaaclab_assets.props.lab_table.lab_table_cfgs`,
  a procedural factory building a real lab table as five static-collision cuboids (one top slab
  and four legs). Exposes the measured geometry as :data:`~isaaclab_assets.props.lab_table.LAB_TABLE_LENGTH`,
  :data:`~isaaclab_assets.props.lab_table.LAB_TABLE_WIDTH`, :data:`~isaaclab_assets.props.lab_table.LAB_TABLE_HEIGHT`,
  :data:`~isaaclab_assets.props.lab_table.LAB_TABLE_TOP_THICKNESS`, and
  :data:`~isaaclab_assets.props.lab_table.LAB_TABLE_LEG_SECTION` for reuse by tasks that need to
  place assets relative to the tabletop.
