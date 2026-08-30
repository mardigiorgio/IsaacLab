# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real setup audit for the slide and flip deployment variants.

Pure config, no GPU. Asserts the deployment placement contract:

1. Both scenes spawn the mug on the SHARED tape-measure mark
   (REAL_SETUP.md): identical x/y root position. z and orientation differ
   BY DESIGN — slide upright at the brief's reference pose, flip inverted.
2. The S2R variants randomize exactly the slight-shift set and nothing
   else: mug ±1 cm x/y and ±10° yaw on the mark (z pinned), arm home
   ±0.03 rad, flip's banks off. Rewards, actions, gates and stepping stay
   byte-identical to the campaign task.
3. The campaign cfgs themselves remain zero-jitter (the tape-measure
   protocol the launched runs were audited under).

Run from the IsaacLab root:
    CUDA_VISIBLE_DEVICES= ~/Documents/code/icra2027/.venv/bin/python \
        scripts/probes/probe_s2r_setup.py
"""

from __future__ import annotations

import math

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


MUG_MARK_XY = (-0.02, 0.0)
SLIDE_UPRIGHT_POS = (-0.02, 0.0, 0.021)  # REAL_SETUP.md sim reference
FLIP_INVERTED_ROT = (0.70710678, 0.70710678, 0.0, 0.0)
XY_SHIFT = 0.01
YAW_SHIFT = 0.17
ARM_SHIFT = 0.03
FLIP_BANKS = (
    "reset_arm_grasp_bank",
    "reset_arm_lift_bank",
    "reset_arm_rotate_bank",
    "reset_arm_hover_bank",
    "reset_arm_home_via_bank",
)


def _zero_jitter(env) -> bool:
    pr = env.events.reset_object_position.params["pose_range"]
    return all(tuple(v) == (0.0, 0.0) for v in pr.values())


def _s2r_jitter(env, name: str) -> None:
    pr = env.events.reset_object_position.params["pose_range"]
    check(tuple(pr["x"]) == (-XY_SHIFT, XY_SHIFT), f"{name}: mug x shift != ±{XY_SHIFT}")
    check(tuple(pr["y"]) == (-XY_SHIFT, XY_SHIFT), f"{name}: mug y shift != ±{XY_SHIFT}")
    check(tuple(pr["z"]) == (0.0, 0.0), f"{name}: mug z not pinned")
    check(tuple(pr["yaw"]) == (-YAW_SHIFT, YAW_SHIFT), f"{name}: mug yaw shift != ±{YAW_SHIFT}")
    check(set(pr) == {"x", "y", "z", "yaw"}, f"{name}: unexpected pose_range keys {set(pr)}")


def _weights(env) -> dict:
    return {k: v.weight for k, v in vars(env.rewards).items() if hasattr(v, "weight")}


def main() -> int:
    from isaaclab_tasks.contrib.trossen_mug_flip.trossen_mug_flip_env_cfg import TrossenMugFlipEnvCfg
    from isaaclab_tasks.contrib.trossen_mug_flip.trossen_sim2real_cfg import TrossenMugFlipS2REnvCfg
    from isaaclab_tasks.contrib.trossen_mug_slide.trossen_mug_slide_env_cfg import TrossenMugSlideEnvCfg
    from isaaclab_tasks.contrib.trossen_mug_slide.trossen_sim2real_cfg import TrossenMugSlideS2REnvCfg

    slide, flip = TrossenMugSlideEnvCfg(), TrossenMugFlipEnvCfg()
    s2r_slide, s2r_flip = TrossenMugSlideS2REnvCfg(), TrossenMugFlipS2REnvCfg()

    # ---- 1. the shared mark ----
    sp, fp = slide.scene.object.init_state, flip.scene.object.init_state
    check(tuple(sp.pos[:2]) == MUG_MARK_XY, f"slide mug xy {tuple(sp.pos[:2])} != mark {MUG_MARK_XY}")
    check(tuple(fp.pos[:2]) == MUG_MARK_XY, f"flip mug xy {tuple(fp.pos[:2])} != mark {MUG_MARK_XY}")
    check(
        all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(sp.pos, SLIDE_UPRIGHT_POS)),
        f"slide mug pos {tuple(sp.pos)} != brief reference {SLIDE_UPRIGHT_POS}",
    )
    check(
        all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(fp.rot, FLIP_INVERTED_ROT)),
        f"flip mug rot {tuple(fp.rot)} != inverted spawn {FLIP_INVERTED_ROT}",
    )
    # S2R must not move the spawn itself, only jitter around it
    for name, base, s2r in (("slide", slide, s2r_slide), ("flip", flip, s2r_flip)):
        check(
            tuple(base.scene.object.init_state.pos) == tuple(s2r.scene.object.init_state.pos),
            f"{name}: S2R moved the mug spawn",
        )

    # ---- 2. the S2R shift set ----
    _s2r_jitter(s2r_slide, "slide-S2R")
    _s2r_jitter(s2r_flip, "flip-S2R")
    arm = s2r_slide.events.randomize_arm_start.params
    check(tuple(arm["position_range"]) == (-ARM_SHIFT, ARM_SHIFT), "slide-S2R: arm home shift != ±0.03")
    check(tuple(arm["velocity_range"]) == (0.0, 0.0), "slide-S2R: arm start velocity nonzero")
    check(
        flip.events.reset_arm_grasp_bank.params.get("home_noise") == ARM_SHIFT,
        "flip: authored home_noise != ±0.03 (the S2R relies on it)",
    )
    for b in FLIP_BANKS:
        check(
            getattr(s2r_flip.events, b).params["bank_fraction"] == 0.0,
            f"flip-S2R: {b} still banked",
        )
        check(
            getattr(flip.events, b).params["bank_fraction"] > 0.0,
            f"flip campaign cfg: {b} unexpectedly zero (S2R leaked into base?)",
        )

    # ---- everything else byte-identical to the campaign task ----
    for name, base, s2r in (("slide", slide, s2r_slide), ("flip", flip, s2r_flip)):
        check(_weights(base) == _weights(s2r), f"{name}: S2R changed reward weights")
        check(
            (base.sim.dt, base.decimation) == (s2r.sim.dt, s2r.decimation),
            f"{name}: S2R changed the step",
        )
        check(
            base.actions.arm_action.scale == s2r.actions.arm_action.scale
            and base.actions.gripper_action.scale == s2r.actions.gripper_action.scale,
            f"{name}: S2R changed action scales",
        )
        cmd_b, cmd_s = base.commands.object_pose, s2r.commands.object_pose
        check(
            (cmd_b.position_success_threshold, cmd_b.orientation_success_threshold)
            == (cmd_s.position_success_threshold, cmd_s.orientation_success_threshold),
            f"{name}: S2R changed the success gates",
        )

    # ---- 3. the campaign cfgs stay tape-measure ----
    check(_zero_jitter(slide), "slide campaign cfg: spawn jitter nonzero")
    check(_zero_jitter(flip), "flip campaign cfg: spawn jitter nonzero")

    # ---- gym ids resolve lazily ----
    import gymnasium as gym

    for gid in ("IsaacContrib-Slide-Mug-Trossen-S2R-v0", "IsaacContrib-Flip-Mug-Trossen-S2R-v0"):
        check(gym.spec(gid) is not None, f"gym id missing: {gid}")

    if FAILURES:
        print("S2R AUDIT FAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("S2R AUDIT PASSED: shared mark, slight-shift set, campaign untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
