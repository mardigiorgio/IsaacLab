# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fast logic-level test for the ``max_num_failures`` termination wiring in ``env_loop``.

``datagen_config.max_num_failures`` is defined on every ``MimicEnvCfg`` (default 25, every mimic
env cfg pins it to 25) but was never read by the datagen pipeline: a ``generation_guarantee=True``
run (the franka stack default) that never succeeds looped forever, bounded only by an external
subprocess timeout (see the Task 10b report / ``test_generate_dataset_franka_newton.py``). This
test drives ``isaaclab_mimic.datagen.generation.env_loop`` directly with a minimal fake env --
no ``AppLauncher``, no real simulation, no ``DataGenerator`` -- to check the new termination
branch in isolation. It fakes attempt bookkeeping by writing straight to the module-level
``num_success`` / ``num_failures`` / ``num_attempts`` counters that ``run_data_generator`` would
otherwise update from a real generation attempt; ``env_loop`` doesn't care how they got there.
"""

import asyncio

import torch

from isaaclab_mimic.datagen import generation


class _FakeSim:
    """Stands in for ``env.sim``; ``is_stopped`` is only consulted if a test doesn't break earlier."""

    def __init__(self, stopped_after: int = 1):
        self._stopped_after = stopped_after
        self._calls = 0

    def is_stopped(self) -> bool:
        self._calls += 1
        return self._calls >= self._stopped_after


class _FakeActionSpace:
    shape = (1,)


class _FakeDataGenConfig:
    def __init__(self, generation_guarantee: bool, generation_num_trials: int, max_num_failures: int):
        self.generation_guarantee = generation_guarantee
        self.generation_num_trials = generation_num_trials
        self.max_num_failures = max_num_failures


class _FakeCfg:
    def __init__(self, datagen_config: _FakeDataGenConfig):
        self.datagen_config = datagen_config


class _FakeEnv:
    """Minimal stand-in exposing only the attributes ``env_loop`` touches."""

    def __init__(self, datagen_config: _FakeDataGenConfig, stopped_after: int = 1):
        self.num_envs = 1
        self.device = "cpu"
        self.action_space = _FakeActionSpace()
        self.cfg = _FakeCfg(datagen_config)
        self.sim = _FakeSim(stopped_after=stopped_after)
        self.step_calls = 0

    def step(self, actions: torch.Tensor) -> None:
        self.step_calls += 1

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


def _reset_counters(num_success: int, num_failures: int, num_attempts: int) -> None:
    """Seed the module-level attempt counters ``env_loop`` reads (normally set by ``run_data_generator``)."""
    generation.num_success = num_success
    generation.num_failures = num_failures
    generation.num_attempts = num_attempts


def _run_one_iteration(env: _FakeEnv) -> None:
    """Drive ``env_loop`` with a single pre-queued action so it evaluates one termination check.

    Pre-filling ``env_action_queue`` with exactly ``env.num_envs`` items means the inner
    "wait for actions" loop is a no-op, so ``env_loop`` proceeds straight to ``env.step`` and then
    the attempt-counter check on its very first pass -- no async generator task needed.
    """
    asyncio_event_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(asyncio_event_loop)
        env_reset_queue = asyncio.Queue()
        env_action_queue = asyncio.Queue()
        env_action_queue.put_nowait((0, torch.zeros(env.action_space.shape)))

        generation.env_loop(
            env=env,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            shared_datagen_info_pool=None,
            asyncio_event_loop=asyncio_event_loop,
            data_gen_tasks=None,
        )
    finally:
        asyncio_event_loop.close()
        asyncio.set_event_loop(None)


def test_env_loop_terminates_on_max_num_failures(capsys):
    """A zero-success, generation_guarantee=True run stops once num_failures hits the cap."""
    cfg = _FakeDataGenConfig(generation_guarantee=True, generation_num_trials=1000, max_num_failures=3)
    env = _FakeEnv(cfg)
    _reset_counters(num_success=0, num_failures=3, num_attempts=3)

    _run_one_iteration(env)

    assert env.step_calls == 1, "env_loop should have taken exactly one step before terminating"
    out = capsys.readouterr().out
    assert "Reached max_num_failures (3)" in out


def test_env_loop_does_not_terminate_before_max_num_failures(capsys):
    """Below the failure cap, env_loop keeps going (falls through to the sim.is_stopped() exit)."""
    cfg = _FakeDataGenConfig(generation_guarantee=True, generation_num_trials=1000, max_num_failures=3)
    env = _FakeEnv(cfg, stopped_after=1)
    _reset_counters(num_success=0, num_failures=2, num_attempts=2)

    _run_one_iteration(env)

    out = capsys.readouterr().out
    assert "Reached max_num_failures" not in out


def test_env_loop_ignores_max_num_failures_without_guarantee(capsys):
    """With generation_guarantee=False, attempts already bound the run; the failure cap is a no-op."""
    cfg = _FakeDataGenConfig(generation_guarantee=False, generation_num_trials=1000, max_num_failures=3)
    env = _FakeEnv(cfg, stopped_after=1)
    _reset_counters(num_success=0, num_failures=5, num_attempts=5)

    _run_one_iteration(env)

    out = capsys.readouterr().out
    assert "Reached max_num_failures" not in out
