# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The mug-hang task's staged reward economy, as PURE tensor math.

No Isaac Lab imports: the sim-side term (``mdp.hang_fsm``) computes the
physical predicates and hands them here; the unit tests drive this module
directly with synthetic trajectories. One source of truth for stages,
milestones, potentials, and the anti-farming arithmetic.

Stages::

    0 APPROACH -> 1 GRASPED -> 2 CARRY -> 3 INSERTED -> 4 PLACED -> success

Advancement requires the stage's physical predicate to hold for its
persistence window; losing the predicate REGRESSES the stage (milestones are
never re-awarded: one-shot flags are separate from the current stage).

Income = one-shot milestone bonuses + strict potential-based shaping
(gamma * Phi(s') - Phi(s), Ng et al. 1999) + a single success bonus paid by
the termination. Phi = STAGE_C * stage + phi_stage(s), phi in [0, STAGE_C],
so the shaping return telescopes to at most PHI_MAX and holding, oscillating,
or re-contacting cannot accumulate.

Anti-farming inequality (asserted in the tests)::

    max pre-completion return  =  sum(MILESTONE_BONUS) + sum(RATCHET_W) + PHI_MAX
                               =  55 + 80 + 20 = 155  <  SUCCESS_BONUS = 200
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- constants
STAGE_C = 4.0  # potential per stage rung (2 -> 4, 2026-08-27: stage-0 gradient starved discovery)
PHI_MAX = 5 * STAGE_C  # Phi at best pre-terminal state (stage 4, phi full)
MILESTONE_BONUS = (5.0, 5.0, 10.0, 15.0, 20.0)  # grasp, lift, insert, release, RETREAT (arm at finish while placed)
# Per-stage progress RATCHET budgets (2026-08-27): strict PBRS alone was ~two
# orders too weak for cold-start discovery (arm never reached the mug by iter
# 300). A ratchet pays W_k per unit of NEW episode-best stage progress --
# dense along the first approach, zero the second time, bounded by W_k.
RATCHET_W = (10.0, 10.0, 15.0, 15.0, 30.0)  # retreat 10 -> 30 (2026-08-28): the swing to the finish pose must out-earn staying placed
SUCCESS_BONUS = 200.0  # > milestones 55 + ratchets 80 + PHI_MAX 20 = 155
GAMMA = 0.99  # must match the agent's discount for strict PBRS

# persistence windows [frames at 30 Hz]
GRASP_FRAMES = 3
LIFT_FRAMES = 3
INSERT_FRAMES = 3
SUPPORT_FRAMES = 12  # 0.4 s
FINISH_FRAMES = 3  # 0.1 s (6 -> 3, 2026-08-28: reach the bonus before the stage-4 PBRS drift dominates)

APPROACH, GRASPED, CARRY, INSERTED, PLACED = 0, 1, 2, 3, 4


@dataclass
class FsmInputs:
    """Physical predicates and bounded progress scalars, one value per env."""

    held: torch.Tensor  # bool: both pads press the mug
    lifted: torch.Tensor  # bool: mug clear of the table
    threaded: torch.Tensor  # bool: branch axis through the handle loop
    supported: torch.Tensor  # bool: threaded & tree contact & calm
    released: torch.Tensor  # bool: no pad force on the mug
    arm_ok: torch.Tensor  # bool: arm within tolerance of the finish pose
    reach_prog: torch.Tensor  # [0,1] approach progress (pads -> mug)
    lift_prog: torch.Tensor  # [0,1] height progress
    insert_prog: torch.Tensor  # [0,1] loop-to-branch, face-on weighted
    release_prog: torch.Tensor  # [0,1] gripper-open x support persistence
    retreat_prog: torch.Tensor  # [0,1] arm-to-finish progress
    lifted_hold: torch.Tensor | None = None  # optional lower-threshold 'still lifted' for CARRY validity (hysteresis)


class HangFsm:
    """Latched per-env stage machine. All buffers torch, device-resident."""

    def __init__(self, num_envs: int, device, ratchet_w=None, milestone_bonus=None, carry_requires_lifted=False):
        # Per-task economy (2026-08-28): the flip needs a heavier ROTATE ratchet
        # (partial rotation must out-earn the shaping loss of the drop that
        # follows it); the hang keeps the module defaults. The anti-farming
        # inequality is asserted for whatever tuples are used.
        self.ratchet_w = tuple(RATCHET_W if ratchet_w is None else ratchet_w)
        self.milestone_bonus = tuple(MILESTONE_BONUS if milestone_bonus is None else milestone_bonus)
        # carry_requires_lifted (flip, 2026-08-28): CARRY stays valid only while the
        # object is still lifted, so letting the lift sag back to the table regresses
        # the stage. The hang keeps the default (its carry target is off the table).
        self.carry_requires_lifted = bool(carry_requires_lifted)
        assert len(self.ratchet_w) == 5 and len(self.milestone_bonus) == 5
        assert sum(self.milestone_bonus) + sum(self.ratchet_w) + PHI_MAX < SUCCESS_BONUS, (
            f"max pre-completion return {sum(self.milestone_bonus) + sum(self.ratchet_w) + PHI_MAX} >= success {SUCCESS_BONUS}"
        )
        z = lambda dt: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
        self.stage = z(torch.long)
        self.awarded = torch.zeros(num_envs, 5, dtype=torch.bool, device=device)
        self.persist = z(torch.long)  # frames the NEXT stage's predicate has held
        self.finish_count = z(torch.long)
        self.regressions = z(torch.long)
        self.prev_phi = z(torch.float32)
        self.has_prev = z(torch.bool)
        self.best_prog = torch.zeros(num_envs, 5, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        self.stage[env_ids] = 0
        self.awarded[env_ids] = False
        self.persist[env_ids] = 0
        self.finish_count[env_ids] = 0
        self.regressions[env_ids] = 0
        self.prev_phi[env_ids] = 0.0
        self.has_prev[env_ids] = False
        self.best_prog[env_ids] = 0.0

    # ------------------------------------------------------------------ core
    def step(self, x: FsmInputs) -> dict[str, torch.Tensor]:
        stage = self.stage

        # -------- regression: the CURRENT stage's own predicate must hold.
        # Losing it falls back to the deepest stage whose predicate holds.
        ok1 = x.held | x.threaded  # GRASPED remains valid while held (or already threaded)
        still_lifted = x.lifted if x.lifted_hold is None else x.lifted_hold
        ok2 = ((x.held & still_lifted) if self.carry_requires_lifted else x.held) | x.threaded  # CARRY likewise
        ok3 = x.threaded  # INSERTED requires the loop on the branch
        ok4 = x.supported  # PLACED requires live support
        target = torch.zeros_like(stage)
        target = torch.where((stage >= 1) & ok1, torch.ones_like(stage), target)
        target = torch.where((stage >= 2) & ok2, torch.full_like(stage, 2), target)
        target = torch.where((stage >= 3) & ok3, torch.full_like(stage, 3), target)
        target = torch.where((stage >= 4) & ok4, torch.full_like(stage, 4), target)
        regressed = target < stage
        self.regressions += regressed.long()
        self.persist = torch.where(regressed, torch.zeros_like(self.persist), self.persist)
        stage = target

        # -------- advancement: next stage's predicate, persisted.
        adv_pred = torch.zeros_like(x.held)
        adv_pred = torch.where(stage == APPROACH, x.held, adv_pred)
        adv_pred = torch.where(stage == GRASPED, x.held & x.lifted, adv_pred)
        adv_pred = torch.where(stage == CARRY, x.threaded, adv_pred)
        adv_pred = torch.where(stage == INSERTED, x.supported & x.released, adv_pred)
        self.persist = torch.where(adv_pred, self.persist + 1, torch.zeros_like(self.persist))
        need = torch.full_like(self.persist, GRASP_FRAMES)
        need = torch.where(stage == GRASPED, torch.full_like(need, LIFT_FRAMES), need)
        need = torch.where(stage == CARRY, torch.full_like(need, INSERT_FRAMES), need)
        need = torch.where(stage == INSERTED, torch.full_like(need, SUPPORT_FRAMES), need)
        advance = (stage < PLACED) & (self.persist >= need)
        new_ms = torch.zeros_like(self.awarded)
        for k in range(4):
            hit = advance & (stage == k) & ~self.awarded[:, k]
            new_ms[:, k] = hit
            self.awarded[:, k] |= hit
        stage = torch.where(advance, stage + 1, stage)
        self.persist = torch.where(advance, torch.zeros_like(self.persist), self.persist)
        self.stage = stage

        # -------- success: PLACED, still supported+released, arm parked, persisted.
        done_pred = (stage == PLACED) & x.supported & x.released & x.arm_ok
        self.finish_count = torch.where(done_pred, self.finish_count + 1, torch.zeros_like(self.finish_count))
        success = self.finish_count >= FINISH_FRAMES
        # 5th milestone (2026-08-28): the RETREAT itself, one-shot, on the first
        # frame the arm is at the finish pose while placed -- the stage-4 PBRS
        # drift is negative (~-0.2/step at the top of the potential), so without
        # a bonus that fires BEFORE the success window, holding placed was
        # strictly cheaper than moving.
        hit5 = done_pred & ~self.awarded[:, 4]
        new_ms[:, 4] = hit5
        self.awarded[:, 4] |= hit5

        # -------- potential and strict PBRS shaping.
        prog = torch.stack([x.reach_prog, x.lift_prog, x.insert_prog, x.release_prog, x.retreat_prog], dim=1)
        phi_stage = prog.gather(1, stage.unsqueeze(1).clamp(0, 4)).squeeze(1).clamp(0.0, 1.0) * STAGE_C
        phi = STAGE_C * stage.float() + phi_stage
        # Phi(s') - Phi(s), NOT gamma*Phi(s') - Phi(s) (2026-08-28): the gamma
        # form drifts by (gamma-1)*Phi per step -- measured -0.18/step at PLACED,
        # -15 to -18 per episode -- which taxes exactly the persistence the FSM
        # demands and made "hold placed, never retreat" the optimum. The plain
        # difference telescopes to zero at rest and to Phi(end)-Phi(start) over
        # an episode, so staying is free and only progress pays; the policy-
        # invariance guarantee of strict PBRS is traded for a rest-neutral
        # potential, and the milestone/success ordering is unchanged (tests).
        shaping = torch.where(self.has_prev, phi - self.prev_phi, torch.zeros_like(phi))
        self.prev_phi = phi
        self.has_prev |= True

        # -------- per-stage progress ratchet: dense on FIRST progress only.
        # best_prog is per (env, stage) and never resets within an episode, so
        # oscillating or re-approaching re-earns nothing; total per stage <= W_k.
        prog_now = prog.gather(1, stage.unsqueeze(1).clamp(0, 4)).squeeze(1).clamp(0.0, 1.0)
        best = self.best_prog.gather(1, stage.unsqueeze(1)).squeeze(1)
        gain = (prog_now - best).clamp(min=0.0)
        w = torch.tensor(self.ratchet_w, device=phi.device).gather(0, stage.clamp(0, 4))
        ratchet_r = w * gain
        self.best_prog.scatter_(1, stage.unsqueeze(1), torch.maximum(best, prog_now).unsqueeze(1))

        milestone_r = (new_ms.float() * torch.tensor(self.milestone_bonus, device=phi.device)).sum(dim=1)
        milestone_r = milestone_r + ratchet_r
        return {
            "success": success,
            "milestone_reward": milestone_r,
            "ratchet": ratchet_r,
            "shaping": shaping,
            "stage": stage,
            "new_milestones": new_ms,
            "regressed": regressed,
            "phi": phi,
        }
