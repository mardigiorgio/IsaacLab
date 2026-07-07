# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the G1 pick-cube skeleton env (PhysX + Newton presets)."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

TASK = "IsaacContrib-Pick-Cube-G1-v0"


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_random_actions_smoke(physics_preset_name):
    """Env creates and survives 20 random-action steps with valid signals on both backends."""
    _run_environments(TASK, "cuda", 2, num_steps=20, physics_preset_name=physics_preset_name)


def test_cube_settles_on_tabletop():
    """The table collision geometry is right: a spawned cube rests at tabletop + half edge."""
    from isaaclab_tasks.contrib.g1_pick_cube.g1_pick_cube_env_cfg import CUBE_REST_Z

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=2)
    env = gym.make(TASK, cfg=env_cfg)
    try:
        env.reset()
        zero_actions = torch.zeros((2, env.unwrapped.action_space.shape[1]), device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(30):  # settle
                env.step(zero_actions)
        cube = env.unwrapped.scene["cube"]
        rest_z = cube.data.root_pos_w.torch[:, 2] - env.unwrapped.scene.env_origins[:, 2]
        # 1 cm tolerance: on the table, not inside it, not on the floor
        assert torch.all(torch.abs(rest_z - CUBE_REST_Z) < 0.01), f"cube rest z = {rest_z.tolist()}"
    finally:
        env.close()
