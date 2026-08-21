# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can the banked pre-grasp physically lift the mug? Scripted close-and-raise.

Policy-free existence proof: every env resets to GRASP_BANK_POSE, the gripper
closes for a settling interval, then the shoulder raises while the wrist pose
is held. If the pinch holds, the mug tracks the gripper upward; if the mug
never leaves the table (or slips out while the arm rises), the failure is in
the CONTACT, not the reward function, and no reward tuning can teach the lift.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_scripted_grasp.py --num_envs 64
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--close_steps", type=int, default=15, help="steps of pure gripper close before raising")
parser.add_argument("--raise_steps", type=int, default=45, help="steps of shoulder raise after the close")
parser.add_argument("--raise_rad", type=float, default=0.4, help="total j1 raise [rad] ramped over raise_steps")

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
        apply_solver_choice(env_cfg, "icf")
        bank = getattr(env_cfg.events, "reset_arm_grasp_bank", None)
        if bank is not None:
            bank.params["bank_fraction"] = 1.0
            bank.params["noise"] = 0.0
        env_cfg.observations.policy.enable_corruption = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        scene = u.scene
        obj = scene["object"]
        env.reset()
        spawn_z = obj.data.root_pos_w.torch[:, 2].clone()

        # Action layout: arm term (6 joints, scaled relative targets) then the
        # binary gripper term (1 action: >=0 open, <0 close).
        arm_dim = u.action_manager.get_term("arm_action").action_dim
        total_dim = u.action_manager.total_action_dim
        act = torch.zeros(args_cli.num_envs, total_dim, device=u.device)
        arm_scale = float(u.action_manager.get_term("arm_action").cfg.scale)
        # j1 is the shoulder pitch: index of follower_left_joint_1 in the term.
        term = u.action_manager.get_term("arm_action")
        j1_col = term._joint_names.index("follower_left_joint_1")

        n_steps = args_cli.close_steps + args_cli.raise_steps
        with torch.inference_mode():
            for step in range(n_steps):
                act[:, arm_dim:] = -1.0  # close command every step
                if step >= args_cli.close_steps:
                    ramp = (step - args_cli.close_steps + 1) / args_cli.raise_steps
                    act[:, j1_col] = -args_cli.raise_rad * ramp / arm_scale
                env.step(act)
                z = obj.data.root_pos_w.torch[:, 2] - spawn_z
                forces = scene.sensors["pad_object_contact"].data.force_matrix_w
                fmax = 0.0
                if forces is not None:
                    fmax = float(torch.linalg.vector_norm(forces.torch.sum(dim=2), dim=-1).nan_to_num(0.0).max())
                tcp = scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
                tcp_mug = torch.linalg.vector_norm(tcp - obj.data.root_pos_w.torch, dim=-1)
                car_ids, _ = u.scene["robot"].find_joints("follower_left_.*_carriage_joint")
                car = u.scene["robot"].data.joint_pos.torch[:, car_ids]
                if step % 5 == 4 or step == n_steps - 1:
                    lifted = float((z > 0.04).float().mean())
                    print(
                        f"[step {step:02d}] mug dz mean {z.mean()*1000:7.1f} mm  max {z.max()*1000:7.1f} mm"
                        f"  lifted>4cm {lifted*100:5.1f}%  pad force max {fmax:8.2f} N"
                        f"  tcp-mug mean {tcp_mug.mean()*1000:6.1f} mm"
                        f"  carriage mean {car.mean()*1000:6.2f} mm"
                    )
        held = float((z > 0.04).float().mean())
        print(f"[verdict] {held*100:.1f}% of envs hold the mug above 4 cm at the end of the raise")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
