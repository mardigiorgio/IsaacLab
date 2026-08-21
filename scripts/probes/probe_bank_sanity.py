# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that grasp-bank reset starts do not disturb the mug.

Rolls the env with EVERY episode drawn from the grasp bank and zero actions —
the worst case for the home-ward PD pull the bank start begins with — and
reports, per control step, the mug's displacement from spawn and the peak pad
contact force. The bank is safe when the retreating arm leaves the mug within
millimeters of its spawn; batting shows up as centimeter drift or force
spikes in the first few steps.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_bank_sanity.py --num_envs 64 --steps 20
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)

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

    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        apply_solver_choice(env_cfg, "icf")
        # The grasp-bank event is not currently wired in the task cfg; when it
        # is, force every reset through the bank — that is the case under test.
        bank = getattr(env_cfg.events, "reset_arm_grasp_bank", None)
        if bank is not None:
            bank.params["bank_fraction"] = 1.0
        env_cfg.observations.policy.enable_corruption = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        obj = scene["object"]
        robot = scene["robot"]
        pad_ids, _ = robot.find_bodies("follower_left_gripper_.*")
        env.reset()
        spawn = (obj.data.root_pos_w.torch - scene.env_origins).clone()
        pad_pos = robot.data.body_pos_w.torch[:, pad_ids]
        pad_dist = torch.linalg.vector_norm(pad_pos - obj.data.root_pos_w.torch[:, None, :], dim=-1)
        print(
            f"[reset] pad-origin-to-mug: mean {pad_dist.mean() * 1000:6.1f} mm"
            f"  min {pad_dist.min() * 1000:6.1f} mm  max {pad_dist.max() * 1000:6.1f} mm"
        )
        # Inside-the-mug test: pad origin's radial distance from the mug's
        # vertical axis vs the rim radius, and its height vs the rim plane.
        mug_root = obj.data.root_pos_w.torch[:, None, :]
        rel = pad_pos - mug_root
        radial = torch.linalg.vector_norm(rel[..., :2], dim=-1)
        height = rel[..., 2]
        for k in range(pad_pos.shape[1]):
            r = radial[0, k] * 1000
            h = height[0, k] * 1000
            where = "INSIDE-CAVITY" if (r < 40 and 0 < h < 97) else ("ABOVE-RIM" if r < 40 else "OUTSIDE-WALL")
            print(f"[reset] pad{k}: radial {r:6.1f} mm  height {h:6.1f} mm  -> {where}")
        zero = torch.zeros(args_cli.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
        with torch.inference_mode():
            for step in range(args_cli.steps):
                env.step(zero)
                pos = obj.data.root_pos_w.torch - scene.env_origins
                drift = torch.linalg.vector_norm(pos[:, :2] - spawn[:, :2], dim=1)
                dz = pos[:, 2] - spawn[:, 2]
                forces = scene.sensors["pad_object_contact"].data.force_matrix_w
                fmax = 0.0
                if forces is not None:
                    fmax = float(torch.linalg.vector_norm(forces.torch.sum(dim=2), dim=-1).nan_to_num(0.0).max())
                # TCP = the finger-midpoint grasp frame, the honest "how far
                # from a grasp" gauge (pad body origins sit at the carriage
                # mount, ~10 cm behind the fingertips). The frame-transformer
                # sensor only populates after a step, so it is read here.
                tcp = scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
                tcp_dist = torch.linalg.vector_norm(tcp - obj.data.root_pos_w.torch, dim=-1)
                print(
                    f"[step {step:02d}] drift mean {drift.mean() * 1000:7.2f} mm  max {drift.max() * 1000:7.2f} mm"
                    f"  dz max {dz.abs().max() * 1000:7.2f} mm  pad force max {fmax:8.2f} N"
                    f"  tcp-mug mean {tcp_dist.mean() * 1000:6.1f} mm"
                )
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
