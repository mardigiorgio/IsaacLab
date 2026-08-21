# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold the pre-grasp pose statically for visual inspection.

Re-writes the bank pose into the arm's joint state every control step, so the
viewer shows the exact authored configuration indefinitely — no PD yank, no
policy motion. Fly the camera around it at leisure.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_hold_pose.py --viz newton
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--steps", type=int, default=100000)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> int:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import GRASP_BANK_POSE
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 2
        apply_solver_choice(env_cfg, "icf")
        env_cfg.episode_length_s = 1.0e6  # no resets while inspecting

        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        ids, names = robot.find_joints(list(GRASP_BANK_POSE.keys()), preserve_order=True)
        target = torch.tensor(
            [GRASP_BANK_POSE[n] for n in names], device=u.device, dtype=torch.float32
        ).unsqueeze(0).repeat(u.num_envs, 1)
        zero_vel = torch.zeros_like(target)
        zero_act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                robot.write_joint_position_to_sim_index(position=target, joint_ids=ids)
                robot.write_joint_velocity_to_sim_index(velocity=zero_vel, joint_ids=ids)
                env.step(zero_act)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
