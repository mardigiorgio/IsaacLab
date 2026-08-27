# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Campaign coefficient audit: print every experiment-relevant coefficient of
the four campaign tasks and ASSERT the cross-task invariants that make the
adaptive-vs-fixed comparison airtight.

Pure config instantiation - no simulator, no GPU. Run it before every
campaign launch; a failed assertion is a broken invariant, not a style issue.

USAGE (single line, from the IsaacLab root)
  ~/Documents/code/IsaacLabRubato/.venv/bin/python scripts/probes/probe_campaign_coefficients.py
"""

from __future__ import annotations

import math

# The K ladder: (env.sim.dt override, decimation override); K3 is each task's
# authored default. Every rung must preserve the 30 Hz control boundary.
K_LADDER = {"K1": (1 / 30, 1), "K2": (1 / 60, 2), "K3": (None, None)}
CONTROL_HZ = 30.0
SEED = 42
NUM_ENVS = 2048
CONTACT_CAP = {"slide": 1024, "lift": 1024, "flip": 1024, "plate": 8192}


def main() -> int:
    from isaaclab_tasks.contrib.trossen_mug_flip.trossen_mug_flip_env_cfg import TrossenMugFlipEnvCfg
    from isaaclab_tasks.contrib.trossen_mug_lift.agents.rsl_rl_ppo_cfg import TrossenMugLiftPPORunnerCfg
    from isaaclab_tasks.contrib.trossen_mug_flip.agents.rsl_rl_ppo_cfg import TrossenMugFlipPPORunnerCfg
    from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import TrossenMugLiftEnvCfg
    from isaaclab_tasks.contrib.trossen_mug_slide.agents.rsl_rl_ppo_cfg import TrossenMugSlidePPORunnerCfg
    from isaaclab_tasks.contrib.trossen_mug_slide.trossen_mug_slide_env_cfg import TrossenMugSlideEnvCfg
    from isaaclab_tasks.contrib.trossen_plate_rack.agents.rsl_rl_ppo_cfg import TrossenPlatePickPPORunnerCfg
    from isaaclab_tasks.contrib.trossen_plate_rack.trossen_plate_rack_env_cfg import TrossenPlatePickEnvCfg

    tasks = {
        "slide": (TrossenMugSlideEnvCfg(), TrossenMugSlidePPORunnerCfg()),
        "lift": (TrossenMugLiftEnvCfg(), TrossenMugLiftPPORunnerCfg()),
        "plate": (TrossenPlatePickEnvCfg(), TrossenPlatePickPPORunnerCfg()),
        "flip": (TrossenMugFlipEnvCfg(), TrossenMugFlipPPORunnerCfg()),
    }

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    print(f"campaign constants: seed={SEED} num_envs={NUM_ENVS} contact caps={CONTACT_CAP}")
    print(f"K ladder: K1 dt=1/30 dec=1, K2 dt=1/60 dec=2, K3 = task default; control {CONTROL_HZ} Hz throughout\n")

    for name, (env, agent) in tasks.items():
        cmd = env.commands.object_pose
        dist = agent.actor.distribution_cfg
        control_dt = env.sim.dt * env.decimation
        print(f"=== {name} ===")
        print(f"  control: dt={env.sim.dt:.6f} x dec {env.decimation} = {1/control_dt:.1f} Hz, episode {env.episode_length_s}s")
        print(f"  actions: arm scale={env.actions.arm_action.scale}  gripper={type(env.actions.gripper_action).__name__}"
              + (f" scale={env.actions.gripper_action.scale}" if hasattr(env.actions.gripper_action, 'scale') else ""))
        print(f"  explore: init_std={dist.init_std} range={dist.std_range}  entropy={agent.algorithm.entropy_coef}")
        print(f"  success: {cmd.class_type.__name__} pos<{cmd.position_success_threshold} tilt<{round(cmd.orientation_success_threshold, 4)}")
        rew = {k: v.weight for k, v in vars(env.rewards).items() if hasattr(v, "weight")}
        print(f"  rewards: {rew}")
        events = [k for k in vars(env.events) if not k.startswith("_")]
        print(f"  events:  {events}\n")

        # ---- invariants ----
        check(abs(1 / control_dt - CONTROL_HZ) < 1e-6, f"{name}: control rate {1/control_dt} != {CONTROL_HZ} Hz")
        for k, (dt, dec) in K_LADDER.items():
            if dt is not None:
                check(abs(dt * dec - 1 / CONTROL_HZ) < 1e-9, f"{name}: {k} override breaks the 30 Hz boundary")
        check(cmd.position_success_threshold == 0.05, f"{name}: success pos gate != 0.05")
        check(abs(cmd.orientation_success_threshold - math.acos(0.87)) < 1e-6, f"{name}: success tilt gate != acos(0.87)")
        check(env.rewards.early_termination.weight == -50, f"{name}: divergence fine != -50")
        check("physics_diverged" in vars(env.terminations), f"{name}: physics_diverged termination missing")
        dr = [k for k in events if "material" in k or "mass" in k or "com" in k or "nudge" in k]
        check(not dr, f"{name}: DR terms present in campaign cfg: {dr}")
        jitter = env.events.reset_object_position.params["pose_range"]
        check(all(v == (0.0, 0.0) for v in jitter.values()), f"{name}: object spawn jitter nonzero: {jitter}")

    # cross-task: the mug tasks share one tape-measured spawn spot
    sx, sy = tasks["slide"][0].scene.object.init_state.pos[:2]
    for t in ("lift", "flip"):
        check(tuple(tasks[t][0].scene.object.init_state.pos[:2]) == (sx, sy), f"{t}: mug spawn differs from slide")

    # Per-task recipes, by ruling (2026-08-27): the lift keeps its TESTED
    # pinch recipe ("lift literally works, we tested it"); wide-search
    # settings belong to the discovery tasks. Uniformity across tasks was
    # measured to BREAK the lift; these are the sanctioned values.
    WIDE_ARM = {
        "follower_left_joint_[0-2]": 0.5,
        "follower_left_joint_3": 1.0,
        "follower_left_joint_[4-5]": 1.5,
    }
    EXPECT = {
        "lift": {"std": (0.5, (0.05, 1.5)), "arm": 0.1, "grip": 0.05},
        "slide": {"std": (1.5, (0.05, 3.0)), "arm": WIDE_ARM, "grip": 0.15},
        "plate": {"std": (1.5, (0.05, 3.0)), "arm": WIDE_ARM, "grip": 0.15},
        "flip": {"std": (1.0, (0.05, 2.5)), "arm": WIDE_ARM, "grip": 0.15},
    }
    for t, (env, agent) in tasks.items():
        exp = EXPECT[t]
        d = agent.actor.distribution_cfg
        check(
            d.init_std == exp["std"][0] and tuple(d.std_range) == exp["std"][1],
            f"{t}: exploration differs from its sanctioned recipe",
        )
        check(
            type(env.actions.gripper_action).__name__ == "JointPositionActionCfg",
            f"{t}: gripper is not position-controlled",
        )
        check(env.actions.gripper_action.scale == exp["grip"], f"{t}: gripper scale != {exp['grip']}")
        check(env.actions.arm_action.scale == exp["arm"], f"{t}: arm scale differs from its sanctioned recipe")

    # lift starts pinned: zero home scatter, exact bank poses
    lift_ev = tasks["lift"][0].events
    check(lift_ev.randomize_arm_start.params["position_range"] == (0.0, 0.0), "lift: home-start scatter nonzero")
    check(lift_ev.reset_arm_grasp_bank.params["noise"] == 0.0, "lift: grasp-bank noise nonzero")

    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("AUDIT PASSED: all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
