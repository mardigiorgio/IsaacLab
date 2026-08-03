# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open the spatula scene with a LIVE articulation that does not drift.

The two failure modes this avoids:

* Physics stopped -> USD joints are not enforced, so dragging a link detaches it.
* ``env.step()`` -> the action manager runs, and a zero action on a
  ``RelativeJointPositionActionCfg`` sets each PD target to the joint's CURRENT
  position. There is no restoring force, so the arm sags on its own.

So: step the SIMULATION directly (never the env), and set the PD targets ONCE to
the reset pose. Physics is live, joints are connected, the arm holds still, and
nothing here overwrites joint targets afterwards — so edits made in the GUI
(Drive > Target Position) take effect and stay.

Run::

    ./isaaclab.sh -p .../assets/hold_scene.py --viz kit
"""

import warp.fem  # noqa: F401  isort: skip  (Kit ships warp 1.13; site-packages is 1.14)

import argparse

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

TASK = "IsaacContrib-Lift-Spatula-G1-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    env_cfg.events.reset_grasp_map = None
    env_cfg.events.randomize_right_arm = None
    env_cfg.events.randomize_spatula_pose = None
    env_cfg.terminations.blade_contact = None
    env_cfg.terminations.spatula_dropped = None
    env_cfg.terminations.robot_exploded = None
    env_cfg.rewards.blade_penalty = None
    env_cfg.episode_length_s = 1.0e6

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        env.reset()

        # Pin the PD targets to the reset pose ONCE. Never written again, so the
        # GUI owns them from here on.
        with torch.inference_mode():
            hold = robot.data.joint_pos.torch.clone()
            robot.set_joint_position_target_index(
                target=hold, joint_ids=torch.arange(robot.num_joints, device=uenv.device)
            )
            robot.write_data_to_sim()

        print("\n[hold] scene is live and held. Physics IS running, so joints are connected.")
        print("[hold] Nothing in this script writes joint targets from now on.")
        print("[hold] In the GUI: select a joint prim -> Drive > Target Position.")
        print("[hold]   right_wrist_pitch_joint  -> -25.8 deg   (palm toward the table)")
        print("[hold]   right_wrist_roll_joint   ->  +8.6 deg")
        print("[hold] Ctrl-C in this terminal to quit.\n")

        try:
            while True:
                # step the SIM, not the env: no action manager, no target rewrite
                uenv.sim.step(render=True)
                robot.update(uenv.sim.get_physics_dt())
        except KeyboardInterrupt:
            pass
        env.close()


if __name__ == "__main__":
    main()
