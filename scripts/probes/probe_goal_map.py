# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the slide goal's command-to-env mapping and solve for a target.

Evaluates the commanded goal's WORLD position at two command values, fits
the affine map per axis, and prints the command that lands the goal at a
requested env-frame position — replacing frame-chain inference with a
measurement.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_goal_map.py --target_y -0.5696
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Slide-Mug-Trossen-v0")
parser.add_argument("--target_y", type=float, default=-0.5696, help="Desired goal env-frame y [m].")
parser.add_argument("--target_x", type=float, default=-0.020, help="Desired goal env-frame x [m].")

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

    from isaaclab.utils.math import combine_frame_transforms
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 2
        apply_solver_choice(env_cfg, "icf")

        def goal_env_for(cmd_x: float, cmd_y: float) -> torch.Tensor:
            env_cfg.commands.object_pose.ranges.pos_x = (cmd_x, cmd_x)
            env_cfg.commands.object_pose.ranges.pos_y = (cmd_y, cmd_y)
            env = gym.make(args_cli.task, cfg=env_cfg)
            env.reset()
            u = env.unwrapped
            robot = u.scene["robot"]
            command = u.command_manager.get_command("object_pose")
            des_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
            )
            out = (des_pos_w - u.scene.env_origins)[0].cpu().clone()
            obj = (u.scene["object"].data.root_pos_w.torch - u.scene.env_origins)[0].cpu().clone()
            env.close()
            return out, obj

        g1, mug = goal_env_for(0.0, 0.0)
        g2, _ = goal_env_for(0.2, 0.1)

        # Anchor pass: where does the ARM actually stand, and where is the slab?
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        ids, names = robot.find_bodies(["follower_left_base_link", "follower_left_link_6"])
        pos = (robot.data.body_pos_w.torch[:, ids] - u.scene.env_origins[:, None, :])[0].cpu()
        for n, p in zip(names, pos):
            print(f"[probe] {n:<28}: {p.tolist()}")
        env.close()

    dx = (g2 - g1) / torch.tensor([0.2, 0.2, 1.0])  # per-axis response, cmd_x step 0.2, cmd_y step 0.1 handled below
    # env response per unit cmd_x and cmd_y (finite differences on separate axes)
    print(f"[probe] mug env pos           : {mug.tolist()}")
    print(f"[probe] goal env @(cmd 0,0)   : {g1.tolist()}")
    print(f"[probe] goal env @(cmd .2,.1) : {g2.tolist()}")
    # Solve assuming axis-aligned map: env = g1 + J * cmd, J diagonal-ish 2x2.
    j_xx = (g2[0] - g1[0]) / 0.2
    j_yx = (g2[1] - g1[1]) / 0.2
    print(f"[probe] d(env)/d(cmd_x) ~ ({j_xx:.3f}, {j_yx:.3f}) [mixed with cmd_y step]")
    # Direct solve on the dominant axis toward target:
    ty, tx = args_cli.target_y, args_cli.target_x
    if abs(j_yx) > abs(j_xx):
        need = (ty - float(g1[1])) / float(j_yx)
        print(f"[probe] to land goal env_y={ty}: set cmd pos_x ~= {need:.4f} (cmd_y controls env_x)")
    else:
        need = (tx - float(g1[0])) / float(j_xx)
        print(f"[probe] to land goal env_x={tx}: set cmd pos_x ~= {need:.4f} (check other axis for y)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
