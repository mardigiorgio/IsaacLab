# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The flip's authoritative 7-stage FSM (2026-08-29 redesign).

Stages::

    0 APPROACH_HANDLE -> 1 HANDLE_GRASPED -> 2 MUG_LIFTED -> 3 MUG_UPRIGHT
      -> 4 MUG_PLACED -> 5 HANDLE_RELEASED -> 6 COMPLETE

Every rewarded transition corresponds to the intended sequence; only stage 6
activates ``task_complete``. Income = one-shot milestone bonuses on the
transitions + a positive-only, per-stage-bucket progress ratchet (episode-best
deltas: oscillation re-earns nothing) + the terminal bonus. No signed PBRS
term: the previous economy paid the same progress twice (shaping AND ratchet).

Grasp-history latch (measured, placed2 model_450/600): the only set-down that
lands upright on this rig is a timed RELEASE mid-flip -- the mug swings off
the pinch and rights itself as it falls (11.7% naive from harvested pre-toss
states; 13 scripted lowerings x 4 squeeze depths land 0%). So:

* ``grasp_latch`` sets when the mug crosses HORIZONTAL (up > ``LATCH_COS`` = 0)
  while VALIDLY grasped -- half the flip must happen in-hand. It is the
  grasp-history evidence: rotation/placement credit and COMPLETE are
  impossible without it (``success_no_grasp`` asserts 0 in tests).
* Stage 3+ validity rides the latch, not the live grasp, because the mug is
  ballistic between release and touchdown. Stage 4 (PLACED) checks the mug's
  state on the table; stage 5 requires the pads free of the mug.

The regression rule follows the hang's: the deepest stage whose validity
predicate holds is the stage; advancement predicates are persisted.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------- stages
APPROACH_HANDLE, HANDLE_GRASPED, MUG_LIFTED, MUG_UPRIGHT, MUG_PLACED, HANDLE_RELEASED, COMPLETE = range(7)

# one-shot milestone bonuses on transitions 0->1 .. 4->5 (spec 2026-08-29)
# grasp raise reverted (2026-08-30, fsm7m): the approach-heavy variant (grasp 5,
# reach w4, wider hover shares, live no_pinch-90) collapsed the held chain --
# lift-bank completion 90% -> 0% by iter 600, the documented far-start decay
# mode. fsm7k's values are the measured optimum.
# approach incomes at LEGACY scale (2026-08-30): the legacy economy solved
# home->pinch on this scene at 96-99% in 1000 iters with pinch milestone 5 and
# reach ratchet 10; FSM7's 2/1 left the home policy idle (0% from home, user-
# visible). Ordering: 33 + 20 = 53 < 90.
MILESTONE_BONUS = (5.0, 4.0, 8.0, 8.0, 8.0)
# positive-only progress buckets: approach, lift, upright, place, retreat
# upright 20 -> 4 (2026-08-29, controlled comparison): fsm7d (w=4) broke out to
# 9.8% completions by iter 358 and was still climbing; fsm7e/f/g (w=20, with
# and without drop fines / seed baselining) all oscillated at 2-6% for 400
# iters. The heavy upright gradient floods the batch with doomed flip attempts
# (mug_fallen 31-43% vs 14%) and starves the toss refinement that actually
# completes. The rotation chain rides RSI (rotpath/toss banks), not gradient.
PROGRESS_W = (10.0, 2.0, 4.0, 3.0, 1.0)  # reach 1 -> 10 (legacy RATCHET_W[0]); see MILESTONE_BONUS note
# stage -> progress bucket (stages 3 and 4 both earn the placement bucket)
STAGE_BUCKET = (0, 1, 2, 3, 3, 4, 4)
# terminal bonus: ~3x the milestone sum (30), and > milestones + full ratchet (41)
SUCCESS_BONUS = 90.0

# persistence windows [control frames at 30 Hz] for transitions 0->1 .. 5->6
NEED = (12, 3, 3, 12, 3, 30)

# thresholds (measured unless noted)
LATCH_COS = 0.0        # past horizontal while grasped (placed2 successes release at up ~0.1)
UPRIGHT_IN_HAND_COS = 0.5   # static held up-cos tops out at 0.66 (probe_flip_low_pinch_chain)
UPRIGHT_HOLD_COS = 0.35     # hysteresis: a settling flip dips to 0.48-0.49 (probe swing trace)
UPRIGHT_COS = 0.87     # on-table upright (30 deg) -- family convention

_ASSERT = sum(MILESTONE_BONUS) + sum(PROGRESS_W)
assert _ASSERT < SUCCESS_BONUS, f"pre-completion max {_ASSERT} >= terminal {SUCCESS_BONUS}"


class FlipFsm7:
    """Vectorized 7-stage FSM. All buffers live on ``device``; no per-env loops."""

    def __init__(self, num_envs: int, device):
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.persist = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.awarded = torch.zeros(num_envs, 5, dtype=torch.bool, device=device)
        self.grasp_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.best_prog = torch.zeros(num_envs, 5, device=device)
        self.regressions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.latch_seeded = torch.zeros(num_envs, dtype=torch.bool, device=device)  # latch came from a bank seed
        self._w = torch.tensor(PROGRESS_W, device=device)
        self._ms = torch.tensor(MILESTONE_BONUS, device=device)
        self._need = torch.tensor(NEED, dtype=torch.long, device=device)
        self._bucket = torch.tensor(STAGE_BUCKET, dtype=torch.long, device=device)

    def reset(self, env_ids):
        self.stage[env_ids] = 0
        self.persist[env_ids] = 0
        self.awarded[env_ids] = False
        self.grasp_latch[env_ids] = False
        self.best_prog[env_ids] = 0.0
        self.regressions[env_ids] = 0
        self.success[env_ids] = False
        self.fresh[env_ids] = True
        self.latch_seeded[env_ids] = False

    def step(
        self,
        *,
        up: torch.Tensor,            # mug up-cosine [-1, 1]
        valid_grasp: torch.Tensor,   # both pads on the HANDLE, neither on the body
        lifted: torch.Tensor,        # z > z0 + lift_height
        lifted_hold: torch.Tensor,   # hysteresis (0.4x) for carry validity
        placed: torch.Tensor,        # upright & on table & calm (lin+ang)
        placed_hold: torch.Tensor,   # upright & on table (no calm: settling wobble)
        released: torch.Tensor,      # pad-handle force below threshold
        no_contact: torch.Tensor,    # NO pad force on the mug at all (body included)
        reach_prog: torch.Tensor,    # [0,1] pads -> handle grasp point
        lift_prog: torch.Tensor,     # [0,1] height above rest
        upright_prog: torch.Tensor,  # [0,1] (up+1)/2
        place_prog: torch.Tensor,    # [0,1] upright-gated descent to the table
        retreat_prog: torch.Tensor,  # [0,1] EE distance from the placed mug
    ) -> dict[str, torch.Tensor]:
        # -------- grasp-history latch (rotation credit only through the handle)
        self.grasp_latch |= (up > LATCH_COS) & valid_grasp
        latch = self.grasp_latch
        upright_in_hand = (up > UPRIGHT_IN_HAND_COS) & latch

        # -------- regression: deepest stage whose validity predicate holds
        v1 = valid_grasp | latch
        v2 = (valid_grasp & lifted_hold) | latch
        v3 = latch & (up > UPRIGHT_HOLD_COS)
        v4 = latch & placed_hold
        v5 = v4 & released
        stage = self.stage
        target = torch.zeros_like(stage)
        for k, vk in ((1, v1), (2, v2), (3, v3), (4, v4), (5, v5)):
            target = torch.where((stage >= k) & vk, torch.full_like(stage, k), target)
        target = torch.where(stage == COMPLETE, stage, target)  # success is terminal
        regressed = target < stage
        self.regressions += regressed.long()
        self.persist = torch.where(regressed, torch.zeros_like(self.persist), self.persist)
        stage = target

        # -------- advancement: next stage's predicate, persisted NEED[stage] frames
        adv_pred = torch.zeros_like(valid_grasp)
        adv_pred = torch.where(stage == APPROACH_HANDLE, valid_grasp, adv_pred)
        adv_pred = torch.where(stage == HANDLE_GRASPED, valid_grasp & lifted, adv_pred)
        adv_pred = torch.where(stage == MUG_LIFTED, upright_in_hand, adv_pred)
        adv_pred = torch.where(stage == MUG_UPRIGHT, placed, adv_pred)
        adv_pred = torch.where(stage == MUG_PLACED, released & placed_hold, adv_pred)
        adv_pred = torch.where(stage == HANDLE_RELEASED, no_contact & placed, adv_pred)
        adv_pred &= stage < COMPLETE
        self.persist = torch.where(adv_pred, self.persist + 1, torch.zeros_like(self.persist))
        advance = adv_pred & (self.persist >= self._need.gather(0, stage.clamp(0, 5)))

        new_ms = torch.zeros_like(self.awarded)
        for k in range(5):
            hit = advance & (stage == k) & ~self.awarded[:, k]
            new_ms[:, k] = hit
            self.awarded[:, k] |= hit
        trans = torch.zeros(stage.shape[0], 6, dtype=torch.bool, device=stage.device)
        for k in range(6):
            trans[:, k] = advance & (stage == k)
        stage = torch.where(advance, stage + 1, stage)
        self.persist = torch.where(advance, torch.zeros_like(self.persist), self.persist)
        self.stage = stage
        self.success = stage == COMPLETE

        # -------- positive-only per-bucket progress ratchet (episode-best delta)
        lift_g = lift_prog * ((valid_grasp | latch).float())
        upright_g = upright_prog * (((valid_grasp & lifted_hold) | latch).float())
        place_g = place_prog * latch.float()
        retreat_g = retreat_prog * (latch & placed_hold).float()
        prog = torch.stack([reach_prog, lift_g, upright_g, place_g, retreat_g], dim=1).clamp(0.0, 1.0)
        # Baseline the first post-reset step (2026-08-29, fsm7e/f): a bank start
        # seeded mid-task otherwise collects its own seeded progress as free
        # ratchet income on step 1 (20 x 0.3 = 6 raw at up -0.4), re-armed by
        # every truncation-reset -- the rotate-drop farming loop. Only progress
        # BEYOND the start state pays.
        if self.fresh.any():
            self.best_prog = torch.where(self.fresh.unsqueeze(1), prog, self.best_prog)
            self.fresh &= False
        bucket = self._bucket.gather(0, stage.clamp(0, 6))
        prog_now = prog.gather(1, bucket.unsqueeze(1)).squeeze(1)
        best = self.best_prog.gather(1, bucket.unsqueeze(1)).squeeze(1)
        gain = (prog_now - best).clamp(min=0.0)
        progress_r = self._w.gather(0, bucket) * gain
        self.best_prog.scatter_(1, bucket.unsqueeze(1), torch.maximum(best, prog_now).unsqueeze(1))

        milestone_r = (new_ms.float() * self._ms).sum(dim=1)
        return {
            "stage": stage.float(),
            # cloned: the curriculum reads env._fsm at reset time, after reset()
            # may have cleared these buffers for the finishing envs
            "success": self.success.clone(),
            "milestone_reward": milestone_r,
            "progress_reward": progress_r,
            "regressed": regressed.float(),
            "new_milestones": new_ms,
            "transitions": trans,
            "grasp_latch": latch.clone(),
            "earned_grasp": (latch & ~self.latch_seeded).clone(),  # latch achieved in-episode, not seeded
            # no_pinch_by reads this: stage>=1 covers seeded starts (their grasp
            # milestone is never awarded), awarded[0] covers pinch-then-drop.
            "pinched_ever": ((stage >= 1) | self.awarded[:, 0] | latch).clone(),
            "valid_grasp": valid_grasp,
            "success_no_grasp": (self.success & ~latch).float(),  # MUST stay 0
        }
