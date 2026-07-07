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

from contextlib import contextmanager

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

TASK = "IsaacContrib-Pick-Cube-G1-v0"


@contextmanager
def _settled_env(physics_preset_name, num_envs=2, steps=30):
    """Create the env on the given preset, settle it with zero actions, and yield it."""
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=num_envs)
    # parse_env_cfg collapses PresetCfg wrappers to their default (PhysX), so a
    # named preset must be re-applied from the raw registry config afterwards.
    if physics_preset_name is not None:
        env_cfg = apply_physics_preset(env_cfg, TASK, physics_preset_name)
    env = gym.make(TASK, cfg=env_cfg)
    try:
        env.reset()
        zero_actions = torch.zeros((num_envs, env.unwrapped.action_space.shape[1]), device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(steps):
                env.step(zero_actions)
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_random_actions_smoke(physics_preset_name):
    """Env creates and survives 20 random-action steps with valid signals on both backends."""
    _run_environments(TASK, "cuda", 2, num_steps=20, physics_preset_name=physics_preset_name)


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_squat_init_pose_is_stable(physics_preset_name):
    """The squat base pose settles without exploding: joints stay finite and within limits."""
    with _settled_env(physics_preset_name) as env:
        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos.torch
        limits = robot.data.joint_pos_limits.torch
        assert torch.isfinite(joint_pos).all(), "joint positions are not finite after settling"
        assert torch.all(joint_pos > limits[..., 0] - 0.1), "joint positions below lower limits"
        assert torch.all(joint_pos < limits[..., 1] + 0.1), "joint positions above upper limits"


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_cube_settles_on_tabletop(physics_preset_name):
    """The table collision geometry is right: a spawned cube rests at tabletop + half edge."""
    from isaaclab_tasks.contrib.g1_pick_cube.g1_pick_cube_env_cfg import CUBE_REST_Z

    with _settled_env(physics_preset_name) as env:
        cube = env.unwrapped.scene["cube"]
        rest_z = cube.data.root_pos_w.torch[:, 2] - env.unwrapped.scene.env_origins[:, 2]
        # 1 cm tolerance: on the table, not inside it, not on the floor
        assert torch.all(torch.abs(rest_z - CUBE_REST_Z) < 0.01), f"cube rest z = {rest_z.tolist()}"


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_palm_obs_and_rewards_use_palm_bodies(physics_preset_name):
    """SceneEntityCfgs resolve to the two palms: 6-dim palm obs, palm-based reaching reward.

    Guards against the silent-failure mode where SceneEntityCfg defaults in
    term functions are never resolved by the managers (body_ids stays
    slice(None) and every robot body is used).
    """
    with _settled_env(physics_preset_name, steps=5) as env:
        obs_manager = env.unwrapped.observation_manager
        dims = dict(zip(obs_manager.active_terms["policy"], obs_manager.group_obs_term_dim["policy"]))
        assert dims["palm_to_cube"] == (6,), f"palm_to_cube obs dim = {dims['palm_to_cube']}, expected (6,)"
        reaching_cfg = env.unwrapped.reward_manager.get_term_cfg("reaching")
        body_ids = reaching_cfg.params["robot_cfg"].body_ids
        assert isinstance(body_ids, list) and len(body_ids) == 2, f"reaching resolved body_ids = {body_ids}"
