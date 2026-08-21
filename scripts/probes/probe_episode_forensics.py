# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-truth episode forensics for the mug lift: what did each env DO.

Rolls the trained policy from a checkpoint and records per-step, per-env
STATE — start type, worst finger-pad-to-mug distance, pad contact, mug
displacement and height — then reports counted facts per start type:
approached / touched / clamped / lifted / knocked, with distances. No
aggregate curves, no interpretation between the sim and the report.

Also cross-checks one logged-metric pathway against raw state (contact
steps counted from forces vs the reward term's own gate), so the
instruments themselves get audited.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_episode_forensics.py --checkpoint logs/rsl_rl/trossen_mug_lift/<run>/model_100.pt
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=150)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> int:
    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import filter_unsupported_rsl_rl_kwargs
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        apply_solver_choice(env_cfg, "icf")
        env_cfg.observations.policy.enable_corruption = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(
            wrapped, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=env.unwrapped.device
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        u = env.unwrapped
        scene = u.scene
        obj = scene["object"]
        robot = scene["robot"]
        pad_ids, _ = robot.find_bodies("follower_left_gripper_.*")
        arm_ids, arm_names = robot.find_joints("follower_left_joint_[0-5]")

        obs, _ = wrapped.reset()
        # classify the start each env actually got, from the realized state
        q0 = robot.data.joint_pos.torch[:, arm_ids].clone()
        from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import GRASP_BANK_POSE

        bank_q = torch.tensor([GRASP_BANK_POSE[n] for n in arm_names], device=u.device)
        car_ids, _ = robot.find_joints("follower_left_.*_carriage_joint")
        car0 = robot.data.joint_pos.torch[:, car_ids].mean(dim=1)
        dist_to_bank = (q0 - bank_q).abs().max(dim=1).values
        start_type = torch.where(
            dist_to_bank < 0.05,
            torch.where(car0 < 0.01, torch.tensor(2, device=u.device), torch.tensor(1, device=u.device)),
            torch.tensor(0, device=u.device),
        )  # 0 = home/random, 1 = pre-grasp, 2 = grasped

        n = args_cli.num_envs
        min_pad_dist = torch.full((n,), float("inf"), device=u.device)
        contact_steps = torch.zeros(n, device=u.device)
        both_pad_steps = torch.zeros(n, device=u.device)
        gate_steps = torch.zeros(n, device=u.device)
        max_mug_rise = torch.zeros(n, device=u.device)
        max_mug_drift = torch.zeros(n, device=u.device)
        spawn = obj.data.root_pos_w.torch.clone()

        from isaaclab_tasks.contrib.trossen_mug_lift.mdp import _pad_force_mags, opposed_grasp

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, _, _, _ = wrapped.step(actions)
                pad_pos = robot.data.body_pos_w.torch[:, pad_ids]
                d = torch.linalg.vector_norm(pad_pos - obj.data.root_pos_w.torch[:, None, :], dim=-1).max(dim=1).values
                min_pad_dist = torch.minimum(min_pad_dist, d)
                forces = _pad_force_mags(u, "pad_object_contact")
                contact_steps += (forces.max(dim=1).values > 0.01).float()
                both_pad_steps += (forces > 0.01).all(dim=1).float()
                gate_steps += opposed_grasp(u, "pad_object_contact", 0.01).float()
                pos = obj.data.root_pos_w.torch
                max_mug_rise = torch.maximum(max_mug_rise, pos[:, 2] - spawn[:, 2])
                max_mug_drift = torch.maximum(
                    max_mug_drift, torch.linalg.vector_norm(pos[:, :2] - spawn[:, :2], dim=1)
                )

        names = {0: "home/random", 1: "pre-grasp", 2: "grasped"}
        for t in (0, 1, 2):
            m = start_type == t
            c = int(m.sum())
            if c == 0:
                print(f"[{names[t]:11s}] 0 envs")
                continue
            approached = int((min_pad_dist[m] < 0.10).sum())
            touched = int((contact_steps[m] > 0).sum())
            clamped = int((both_pad_steps[m] > 5).sum())
            lifted = int((max_mug_rise[m] > 0.04).sum())
            knocked = int(((max_mug_drift[m] > 0.05) & (max_mug_rise[m] < 0.04)).sum())
            print(
                f"[{names[t]:11s}] {c:2d} envs | approached<10cm {approached:2d} | touched {touched:2d}"
                f" | clamped>5steps {clamped:2d} | lifted>4cm {lifted:2d} | knocked {knocked:2d}"
                f" | median min-pad-dist {min_pad_dist[m].median()*1000:6.1f} mm"
            )
        # instrument audit: the reward gate vs raw forces must agree
        agree = (gate_steps == both_pad_steps).all()
        print(f"[audit] opposed_grasp gate vs raw both-pad forces: {'AGREE' if agree else 'DISAGREE'}"
              f" (gate {gate_steps.sum():.0f} vs raw {both_pad_steps.sum():.0f} total steps)")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
