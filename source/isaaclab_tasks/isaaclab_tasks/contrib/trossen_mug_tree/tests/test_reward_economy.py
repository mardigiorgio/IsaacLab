# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anti-farming proofs for the mug-hang staged economy (pure core, no sim).

Run: python -m pytest test_reward_economy.py -q   (or plain python: __main__ runs all)

Each test drives :class:`HangFsm` with a synthetic 150-step exploit trajectory
and asserts its return is below the completing trajectory's.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hang_fsm_core import (  # noqa: E402
    RATCHET_W,
    FINISH_FRAMES,
    GAMMA,
    MILESTONE_BONUS,
    PHI_MAX,
    SUCCESS_BONUS,
    SUPPORT_FRAMES,
    FsmInputs,
    HangFsm,
)

T = 150  # episode length [control steps]


def _inputs(**kw):
    base = dict(
        held=False, lifted=False, threaded=False, supported=False, released=True, arm_ok=False,
        reach_prog=0.5, lift_prog=0.0, insert_prog=0.0, release_prog=0.0, retreat_prog=0.0,
    )
    base.update(kw)
    out = {}
    for k, v in base.items():
        if isinstance(v, bool):
            out[k] = torch.tensor([v])
        else:
            out[k] = torch.tensor([float(v)])
    return FsmInputs(**out)


def _run(seq):
    """seq: iterable of FsmInputs; returns (undiscounted return, success_step or None, fsm)."""
    fsm = HangFsm(1, "cpu")
    fsm.reset(slice(None))
    total = 0.0
    success_at = None
    for t, x in enumerate(seq):
        out = fsm.step(x)
        total += float(out["milestone_reward"][0] + out["shaping"][0])
        if bool(out["success"][0]) and success_at is None:
            success_at = t
            total += SUCCESS_BONUS
            break
    return total, success_at, fsm


def traj_complete():
    """The intended behavior: grasp -> lift -> insert -> release -> retreat."""
    seq = []
    for _ in range(10):
        seq.append(_inputs(reach_prog=0.8))
    for _ in range(5):
        seq.append(_inputs(held=True, released=False, lift_prog=0.2))
    for _ in range(5):
        seq.append(_inputs(held=True, lifted=True, released=False, lift_prog=1.0))
    for _ in range(6):
        seq.append(_inputs(held=True, lifted=True, threaded=True, released=False, insert_prog=0.9))
    for _ in range(SUPPORT_FRAMES + 2):
        seq.append(_inputs(threaded=True, supported=True, released=True, release_prog=0.9))
    for _ in range(FINISH_FRAMES + 2):
        seq.append(_inputs(threaded=True, supported=True, released=True, arm_ok=True, retreat_prog=1.0))
    return seq


def test_completion_dominates_hover_near_branch():
    hover = [_inputs(held=True, lifted=True, released=False, insert_prog=0.95, reach_prog=1.0) for _ in range(T)]
    r_hover, s, _ = _run(hover)
    r_done, s_done, _ = _run(traj_complete())
    assert s is None and s_done is not None
    assert r_done > r_hover + 50, (r_done, r_hover)


def test_completion_dominates_persistent_contact():
    touch = [_inputs(held=True, released=False, reach_prog=1.0) for _ in range(T)]
    r_touch, s, _ = _run(touch)
    r_done, _, _ = _run(traj_complete())
    assert s is None and r_done > r_touch + 50, (r_done, r_touch)


def test_threshold_oscillation_nets_nothing():
    seq = []
    for k in range(T):
        if k % 2 == 0:
            seq.append(_inputs(held=True, released=False, reach_prog=1.0))
        else:
            seq.append(_inputs(held=False, released=True, reach_prog=0.2))
    r_osc, s, fsm = _run(seq)
    assert s is None
    # grasp milestone at most once, PBRS telescopes: return bounded by bonus + PHI_MAX
    assert r_osc <= MILESTONE_BONUS[0] + RATCHET_W[0] + RATCHET_W[1] + PHI_MAX + 1e-3, r_osc
    r_done, _, _ = _run(traj_complete())
    assert r_done > r_osc + 50


def test_repeated_grasping_awards_once():
    seq = []
    for _ in range(5):
        for _ in range(6):
            seq.append(_inputs(held=True, released=False))
        for _ in range(6):
            seq.append(_inputs(held=False, released=True))
    _, _, fsm = _run(seq)
    assert int(fsm.awarded[:, 0].sum()) == 1
    assert int(fsm.regressions[0]) >= 4  # every drop counted


def test_timeout_below_completion():
    # best possible non-completing episode: all milestones, park at max potential
    seq = traj_complete()[:-FINISH_FRAMES - 2]  # stop just short of the finish window
    seq += [_inputs(threaded=True, supported=True, released=True, retreat_prog=0.99)] * (T - len(seq))
    r_almost, s, _ = _run(seq)
    r_done, _, _ = _run(traj_complete())
    assert s is None
    assert r_almost <= sum(MILESTONE_BONUS) + sum(RATCHET_W) + PHI_MAX + 1e-3, r_almost
    assert r_done > r_almost + 50, (r_done, r_almost)


def test_inequality_constants():
    assert sum(MILESTONE_BONUS) + sum(RATCHET_W) + PHI_MAX < SUCCESS_BONUS


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"ALL {len(fns)} PASSED")
