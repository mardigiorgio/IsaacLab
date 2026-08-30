# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic tests of the flip's 7-stage FSM (flip_fsm_core.FlipFsm7). Pure torch; no sim."""

import torch

from isaaclab_tasks.contrib.trossen_mug_flip.flip_fsm_core import (
    COMPLETE,
    HANDLE_GRASPED,
    HANDLE_RELEASED,
    MILESTONE_BONUS,
    MUG_LIFTED,
    MUG_PLACED,
    MUG_UPRIGHT,
    NEED,
    FlipFsm7,
)

N = 4
DEV = "cpu"


def mk():
    return FlipFsm7(N, DEV)


def step(f, **kw):
    """Step with everything false/zero except the given signals."""
    d = dict(
        up=torch.full((N,), -1.0),
        valid_grasp=torch.zeros(N, dtype=torch.bool),
        lifted=torch.zeros(N, dtype=torch.bool),
        lifted_hold=torch.zeros(N, dtype=torch.bool),
        placed=torch.zeros(N, dtype=torch.bool),
        placed_hold=torch.zeros(N, dtype=torch.bool),
        released=torch.ones(N, dtype=torch.bool),
        no_contact=torch.ones(N, dtype=torch.bool),
        reach_prog=torch.zeros(N),
        lift_prog=torch.zeros(N),
        upright_prog=torch.zeros(N),
        place_prog=torch.zeros(N),
        retreat_prog=torch.zeros(N),
    )
    for k, v in kw.items():
        d[k] = torch.full((N,), float(v)) if k.endswith("_prog") or k == "up" else torch.full((N,), bool(v), dtype=torch.bool)
    return f.step(**d)


T = torch.ones(N, dtype=torch.bool)
F = torch.zeros(N, dtype=torch.bool)


def run(f, frames, **kw):
    out = None
    for _ in range(frames):
        out = step(f, **kw)
    return out


def to_grasped(f):
    return run(f, NEED[0] + 1, valid_grasp=True, released=False)


def to_lifted(f):
    to_grasped(f)
    return run(f, NEED[1] + 1, valid_grasp=True, released=False, lifted=True, lifted_hold=True)


def to_upright(f):
    to_lifted(f)
    return run(f, NEED[2] + 1, valid_grasp=True, released=False, lifted=True, lifted_hold=True, up=0.9)


def to_placed(f):
    to_upright(f)
    # mid-flip release: mug ballistic, lands upright on the table (latch carries validity)
    return run(f, NEED[3] + 1, up=0.95, placed=True, placed_hold=True)


def to_released(f):
    to_placed(f)
    return run(f, NEED[4] + 1, up=0.95, placed=True, placed_hold=True)


def test_1_upright_without_grasp_never_completes():
    f = mk()
    out = run(f, 200, up=1.0, placed=True, placed_hold=True)  # mug righted by a push
    assert int(out["stage"].max()) == 0 and not out["success"].any()


def test_2_body_touch_is_not_a_grasp():
    # valid_grasp is computed upstream as (both pads on handle) & (no pad on body);
    # the FSM must not advance while it is false.
    f = mk()
    out = run(f, 40, valid_grasp=False, released=False, up=0.2)
    assert int(out["stage"].max()) == 0 and not f.grasp_latch.any()


def test_3_one_pad_is_insufficient():
    # same upstream contract: one-pad contact -> valid_grasp False -> no advance
    f = mk()
    out = run(f, 40, valid_grasp=False, released=False)
    assert int(out["stage"].max()) == 0


def test_4_grasp_advances_after_persistence():
    f = mk()
    out = run(f, NEED[0] - 1, valid_grasp=True, released=False)
    assert int(out["stage"].max()) == 0  # one frame short of the window's close
    out = step(f, valid_grasp=True, released=False)
    assert (out["stage"] == HANDLE_GRASPED).all() and out["milestone_reward"].allclose(torch.full((N,), MILESTONE_BONUS[0]))


def test_5_lift_without_grasp_earns_nothing():
    f = mk()
    out = run(f, 30, lifted=True, lifted_hold=True, lift_prog=1.0, upright_prog=1.0, up=0.9)
    assert int(out["stage"].max()) == 0
    assert float(out["progress_reward"].sum()) == 0.0  # reach bucket is 0; lift/upright gated off


def test_6_full_grasp_lift_upright_sequence():
    f = mk()
    assert (to_grasped(f)["stage"] == HANDLE_GRASPED).all()
    assert (to_lifted(f)["stage"] == MUG_LIFTED).all()
    assert (to_upright(f)["stage"] == MUG_UPRIGHT).all()


def test_7_placed_without_release_does_not_complete():
    f = mk()
    to_placed(f)
    out = run(f, 100, up=0.95, placed=True, placed_hold=True, released=False, no_contact=False)
    assert (out["stage"] == MUG_PLACED).all() and not out["success"].any()


def test_8_release_then_instability_does_not_complete():
    f = mk()
    to_released(f)
    out = run(f, 100, up=0.2)  # mug tips over after the release
    assert not out["success"].any() and int(out["stage"].max()) < MUG_PLACED


def test_9_stable_release_30_frames_completes():
    f = mk()
    out = to_released(f)
    assert (out["stage"] == HANDLE_RELEASED).all()
    out = run(f, NEED[5], up=0.95, placed=True, placed_hold=True)
    assert (out["stage"] == COMPLETE).all() and out["success"].all()


def test_10_milestones_pay_once():
    f = mk()
    to_grasped(f)
    total = torch.zeros(N)
    # drop the mug (regress to 0), regrasp: no second grasp milestone
    run(f, 3)
    for _ in range(NEED[0] + 5):
        total += step(f, valid_grasp=True, released=False)["milestone_reward"]
    assert float(total.sum()) == 0.0


def test_11_reset_clears_everything():
    f = mk()
    to_released(f)
    f.reset(torch.arange(N))
    assert int(f.stage.sum()) == 0 and not f.grasp_latch.any() and not f.awarded.any()
    assert float(f.best_prog.sum()) == 0.0 and int(f.persist.sum()) == 0 and not f.success.any()


def test_12_completion_requires_grasp_history():
    # fuzz: random predicate streams with valid_grasp forced False can never complete
    g = torch.Generator().manual_seed(7)
    f = mk()
    for _ in range(500):
        out = f.step(
            up=torch.rand(N, generator=g) * 2 - 1,
            valid_grasp=F,
            lifted=torch.rand(N, generator=g) > 0.5,
            lifted_hold=torch.rand(N, generator=g) > 0.5,
            placed=torch.rand(N, generator=g) > 0.5,
            placed_hold=torch.rand(N, generator=g) > 0.5,
            released=torch.rand(N, generator=g) > 0.5,
            no_contact=torch.rand(N, generator=g) > 0.5,
            reach_prog=torch.rand(N, generator=g),
            lift_prog=torch.rand(N, generator=g),
            upright_prog=torch.rand(N, generator=g),
            place_prog=torch.rand(N, generator=g),
            retreat_prog=torch.rand(N, generator=g),
        )
        assert not out["success"].any() and float(out["success_no_grasp"].sum()) == 0.0
        assert int(out["stage"].max()) == 0


def test_13_ratchet_never_repays():
    f = mk()
    step(f, reach_prog=0.1)  # post-reset baseline step (seeded progress, unpaid)
    r1 = step(f, reach_prog=0.8)["progress_reward"]
    r2 = step(f, reach_prog=0.3)["progress_reward"]  # moved away
    r3 = step(f, reach_prog=0.8)["progress_reward"]  # came back
    assert float(r1.sum()) > 0 and float(r2.sum()) == 0.0 and float(r3.sum()) == 0.0


def test_14_seeded_progress_is_not_paid():
    # a fresh (post-reset) env's first step baselines the ratchet to the seeded
    # state's progress: high starting progress earns nothing; improvement does
    f = mk()
    r1 = step(f, reach_prog=0.9)["progress_reward"]
    assert float(r1.sum()) == 0.0  # the seeded 0.9 is the baseline, not income
    r2 = step(f, reach_prog=0.95)["progress_reward"]
    assert float(r2.sum()) > 0.0  # genuine improvement beyond the seed pays
