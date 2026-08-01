# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the procedural lab table asset factory."""

from isaaclab.app import AppLauncher

# launch omniverse app (isaaclab imports require Kit)
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from isaaclab_assets.props.lab_table import (
    LAB_TABLE_HEIGHT,
    LAB_TABLE_LEG_SECTION,
    LAB_TABLE_LENGTH,
    LAB_TABLE_TOP_THICKNESS,
    LAB_TABLE_WIDTH,
    lab_table_cfgs,
)


def test_constants_match_real_table():
    # measured: 122 cm long, 30 in wide, 7 cm thick top, tabletop 83 cm off the floor
    assert pytest.approx(1.22) == LAB_TABLE_LENGTH
    assert pytest.approx(0.762) == LAB_TABLE_WIDTH
    assert pytest.approx(0.83) == LAB_TABLE_HEIGHT
    assert pytest.approx(0.07) == LAB_TABLE_TOP_THICKNESS


def test_factory_returns_five_static_parts():
    cfgs = lab_table_cfgs("{ENV_REGEX_NS}/LabTable")
    assert set(cfgs) == {"table_top", "table_leg_0", "table_leg_1", "table_leg_2", "table_leg_3"}
    for name, cfg in cfgs.items():
        assert cfg.prim_path.startswith("{ENV_REGEX_NS}/LabTable/"), name
        # static: collision enabled, no rigid-body physics
        assert cfg.spawn.collision_props is not None, name
        assert getattr(cfg.spawn, "rigid_props", None) is None, name


def test_top_slab_geometry():
    cfgs = lab_table_cfgs("/World/LabTable")
    top = cfgs["table_top"]
    assert top.spawn.size == pytest.approx((1.22, 0.762, 0.07))
    # top face at LAB_TABLE_HEIGHT -> center half a slab below it
    assert top.init_state.pos[2] == pytest.approx(LAB_TABLE_HEIGHT - 0.035)


def test_legs_flush_with_corners():
    cfgs = lab_table_cfgs("/World/LabTable")
    half_len = 1.22 / 2 - LAB_TABLE_LEG_SECTION / 2
    half_wid = 0.762 / 2 - LAB_TABLE_LEG_SECTION / 2
    leg_height = LAB_TABLE_HEIGHT - LAB_TABLE_TOP_THICKNESS
    expected_xy = {(sx * half_len, sy * half_wid) for sx in (-1, 1) for sy in (-1, 1)}
    actual_xy = set()
    for i in range(4):
        leg = cfgs[f"table_leg_{i}"]
        assert leg.spawn.size == pytest.approx((0.05, 0.05, leg_height))
        assert leg.init_state.pos[2] == pytest.approx(leg_height / 2)
        actual_xy.add((round(leg.init_state.pos[0], 6), round(leg.init_state.pos[1], 6)))
    assert actual_xy == {(round(x, 6), round(y, 6)) for x, y in expected_xy}
