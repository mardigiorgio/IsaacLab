# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the G1 spatula-lift env (PhysX + Newton presets)."""

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

TASK = "IsaacContrib-Lift-Spatula-G1-v0"


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
def test_init_pose_is_stable(physics_preset_name):
    """The init pose settles without exploding: joints stay finite and within limits."""
    with _settled_env(physics_preset_name) as env:
        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos.torch
        limits = robot.data.joint_pos_limits.torch
        assert torch.isfinite(joint_pos).all(), "joint positions are not finite after settling"
        assert torch.all(joint_pos > limits[..., 0] - 0.1), "joint positions below lower limits"
        assert torch.all(joint_pos < limits[..., 1] + 0.1), "joint positions above upper limits"


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_spatula_settles_on_tabletop(physics_preset_name):
    """The spatula rests on the table near its deterministic spawn, not inside or below it."""
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import LAB_TABLE_HEIGHT, SPATULA_SPAWN_POS

    with _settled_env(physics_preset_name, steps=60) as env:
        spatula = env.unwrapped.scene["spatula"]
        pos = spatula.data.root_pos_w.torch - env.unwrapped.scene.env_origins
        assert torch.all(pos[:, 2] > LAB_TABLE_HEIGHT - 0.01), f"spatula below tabletop: z = {pos[:, 2].tolist()}"
        assert torch.all(pos[:, 2] < LAB_TABLE_HEIGHT + 0.08), f"spatula not settled: z = {pos[:, 2].tolist()}"
        xy_drift = torch.linalg.vector_norm(pos[:, :2] - torch.tensor(SPATULA_SPAWN_POS[:2], device=pos.device), dim=1)
        assert torch.all(xy_drift < 0.05), f"spatula drifted in xy: {xy_drift.tolist()}"


@pytest.mark.parametrize("physics_preset_name", [None, "newton_mjwarp"], ids=["physx", "newton_mjwarp"])
def test_reach_obs_and_rewards_use_right_hand_bodies(physics_preset_name):
    """SceneEntityCfgs resolve: 3-dim palm obs and the right palm.

    Guards against the silent-failure mode where SceneEntityCfg defaults in
    term functions are never resolved by the managers (body_ids stays
    slice(None) and every robot body is used).
    """
    with _settled_env(physics_preset_name, steps=5) as env:
        obs_manager = env.unwrapped.observation_manager
        dims = dict(zip(obs_manager.active_terms["policy"], obs_manager.group_obs_term_dim["policy"]))
        assert dims["palm_to_handle"] == (3,), f"palm_to_handle obs dim = {dims['palm_to_handle']}, expected (3,)"
        reach_cfg = env.unwrapped.reward_manager.get_term_cfg("reach")
        body_ids = reach_cfg.params["bodies_cfg"].body_ids
        assert isinstance(body_ids, list) and len(body_ids) == 4, f"reach bodies_cfg resolved body_ids = {body_ids}"


@pytest.mark.parametrize("physics_preset_name", ["newton_mjwarp"], ids=["newton_mjwarp"])
def test_new_reward_functions_are_finite_and_ungated(physics_preset_name):
    """The three new reward terms compute finite [num_envs] tensors with no contact gate.

    At rest the spatula is on the table, so lift/track must read exactly 0.0;
    reach must be strictly positive despite zero contact force, which is the
    whole point of making the gate optional.
    """
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.contrib.g1_spatula_lift import mdp
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
        ACTUATED_JOINT_NAMES,
        CARRY_POINT,
        FINGERTIP_BODY_NAMES,
        HANDLE_GRASP_OFFSET_B,
        REST_SPATULA_Z,
    )

    with _settled_env(physics_preset_name, num_envs=2, steps=30) as env:
        uenv = env.unwrapped

        lifted = mdp.object_lifted(uenv, rest_height=REST_SPATULA_Z, minimal_offset=0.03)
        assert lifted.shape == (2,), f"object_lifted shape {lifted.shape}"
        assert torch.isfinite(lifted).all()
        assert torch.all(lifted == 0.0), f"resting spatula reads lifted: {lifted.tolist()}"

        track = mdp.track_carry_point(
            uenv,
            carry_point=CARRY_POINT,
            stds=(0.30, 0.05),
            weights=(0.5, 0.5),
            rest_height=REST_SPATULA_Z,
            minimal_offset=0.03,
        )
        assert track.shape == (2,) and torch.isfinite(track).all()
        assert torch.all(track == 0.0), f"resting spatula earns track income: {track.tolist()}"

        vel_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)
        vel_cfg.resolve(uenv.scene)
        jv = mdp.joint_vel_l2_clamped(uenv, asset_cfg=vel_cfg)
        assert jv.shape == (2,) and torch.isfinite(jv).all()
        assert torch.all(jv >= 0.0) and torch.all(jv <= 1000.0)

        bodies_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link", *FINGERTIP_BODY_NAMES])
        bodies_cfg.resolve(uenv.scene)
        reach = mdp.fingers_to_handle(
            uenv,
            std=0.4,
            contact_threshold=None,
            grasp_offset_b=HANDLE_GRASP_OFFSET_B,
            bodies_cfg=bodies_cfg,
        )
        assert reach.shape == (2,) and torch.isfinite(reach).all()
        assert torch.all(reach > 0.0), f"ungated reach paid nothing without contact: {reach.tolist()}"


@pytest.mark.parametrize("physics_preset_name", ["newton_mjwarp"], ids=["newton_mjwarp"])
def test_reward_and_curriculum_terms_agree(physics_preset_name):
    """Every curriculum term names a live reward term, and no task term is contact-gated.

    modify_reward_weight resolves term_name in __init__, so a stale name is a
    hard failure at env construction rather than a silent no-op. This pins that
    contract, and pins that the claw recipe's task terms take no contact
    threshold.
    """
    with _settled_env(physics_preset_name, steps=5) as env:
        uenv = env.unwrapped
        reward_terms = set(uenv.reward_manager.active_terms)
        assert {"reach", "lift", "track"} <= reward_terms, f"missing task terms: {reward_terms}"
        assert "contact_count" not in reward_terms, "the ungated touch bootstrap was not removed"
        assert "action_l2" not in reward_terms, "action_l2 was not replaced by joint_vel_l2"

        # CurriculumManager exposes no get_term_cfg: names come off the manager
        # (proving the term registered), params off the public env cfg.
        curriculum_terms = set(uenv.curriculum_manager.active_terms)
        for name in ("action_rate", "joint_vel"):
            assert name in curriculum_terms, f"curriculum term '{name}' is not active: {curriculum_terms}"
            params = getattr(uenv.cfg.curriculum, name).params
            assert params["term_name"] in reward_terms, (
                f"curriculum '{name}' targets '{params['term_name']}', not a live reward term"
            )
            assert params["num_steps"] == 6000, f"curriculum '{name}' num_steps = {params['num_steps']}"

        for name in ("reach", "lift", "track"):
            params = uenv.reward_manager.get_term_cfg(name).params
            assert params.get("contact_threshold") is None, f"'{name}' is still contact-gated"


def test_ppo_cfg_caps_policy_std():
    """The std CAP is the fix for the blade spiral; entropy bonus is off.

    Run 2026-08-01 ballooned std to 0.98 inside std_range=(0.12, 1.0), the arm
    dove at the object and 94% of episodes died on blade contact. rsl_rl 5.4.1
    clamps std_param to std_range on every update, so the cap is enforced by
    the library. init_std must sit strictly below the cap or the policy starts
    pinned to it. This test needs no simulator.
    """
    from isaaclab_tasks.contrib.g1_spatula_lift.agents.rsl_rl_ppo_cfg import G1SpatulaLiftPPORunnerCfg

    cfg = G1SpatulaLiftPPORunnerCfg()
    lo, hi = cfg.actor.distribution_cfg.std_range
    assert (lo, hi) == (0.15, 0.5), f"std_range = {(lo, hi)}"
    assert lo < cfg.actor.distribution_cfg.init_std < hi, (
        f"init_std {cfg.actor.distribution_cfg.init_std} is not strictly inside {(lo, hi)}"
    )
    assert cfg.algorithm.entropy_coef == 0.0
    assert cfg.algorithm.learning_rate == 1.0e-4
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.gamma == 0.98
    assert cfg.max_iterations == 1500
