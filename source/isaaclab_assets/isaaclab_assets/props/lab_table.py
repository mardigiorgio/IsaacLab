# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural model of the physical lab table.

Frame convention: the factory's ``prim_path`` root sits at the table footprint
center ON THE FLOOR, +Z up, with the long axis along +X. The working surface
is the plane z = :data:`LAB_TABLE_HEIGHT`.

The table is intentionally static collision geometry only (the real table does
not move); contact stiffness is authored by the consuming task's physics
preset, not here.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

LAB_TABLE_LENGTH = 1.825
"""Tabletop length along +X [m]."""

LAB_TABLE_WIDTH = 0.61
"""Tabletop width along +Y [m]."""

LAB_TABLE_HEIGHT = 0.289
"""Height of the top working surface above the floor [m]."""

LAB_TABLE_TOP_THICKNESS = 0.03
"""Top slab thickness [m] (slab spans z in [0.259, 0.289])."""

LAB_TABLE_LEG_SECTION = 0.05
"""Square leg cross-section side [m]. Approximate — adjust when measured."""


def lab_table_cfgs(prim_path: str) -> dict[str, AssetBaseCfg]:
    """Build the five static-collision cuboids composing the lab table.

    Args:
        prim_path: Root prim path for the table (e.g. ``"{ENV_REGEX_NS}/LabTable"``).
            Parts are spawned as children of this path.

    Returns:
        Mapping of scene-entity name to config: ``table_top`` and
        ``table_leg_0`` .. ``table_leg_3`` (legs flush with the footprint
        corners). Assign each entry as an attribute of an
        :class:`~isaaclab.scene.InteractiveSceneCfg` (see the g1_pick_cube
        scene for the intended usage).
    """
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
    collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    leg_height = LAB_TABLE_HEIGHT - LAB_TABLE_TOP_THICKNESS

    cfgs: dict[str, AssetBaseCfg] = {
        "table_top": AssetBaseCfg(
            prim_path=f"{prim_path}/top",
            spawn=sim_utils.CuboidCfg(
                size=(LAB_TABLE_LENGTH, LAB_TABLE_WIDTH, LAB_TABLE_TOP_THICKNESS),
                collision_props=collision,
                visual_material=visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, LAB_TABLE_HEIGHT - LAB_TABLE_TOP_THICKNESS / 2)),
        )
    }
    half_len = LAB_TABLE_LENGTH / 2 - LAB_TABLE_LEG_SECTION / 2
    half_wid = LAB_TABLE_WIDTH / 2 - LAB_TABLE_LEG_SECTION / 2
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        cfgs[f"table_leg_{i}"] = AssetBaseCfg(
            prim_path=f"{prim_path}/leg_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(LAB_TABLE_LEG_SECTION, LAB_TABLE_LEG_SECTION, leg_height),
                collision_props=collision,
                visual_material=visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(sx * half_len, sy * half_wid, leg_height / 2)),
        )
    return cfgs
