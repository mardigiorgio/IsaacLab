# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive joint posing for the G1 spatula claw, with a live viewer.

Type joint values at the prompt; the hand moves in the viewer immediately and
the fingertip-vs-handle geometry is printed after every change. ``save`` prints
a paste-ready ``DEFAULT_ARM_JOINT_POS`` block.

Run::

    ./isaaclab.sh -p .../assets/pose_fingers.py --viz kit presets=newton_mjwarp
    ./isaaclab.sh -p .../assets/pose_fingers.py --viz newton presets=newton_mjwarp

Commands (one per line)::

    index 0.8            both index joints to 0.8 rad
    middle 0.8           both middle joints
    index0 0.5           a single joint
    thumb0 0.6           thumb abduction (opposition)
    thumb1 -0.5          thumb flexion
    thumb2 -1.0          thumb distal
    fingers 0.8          index AND middle together
    arm -0.64 -0.14 -0.10 1.10 -1.20 0.50 -0.21    all 7 arm joints
    open                 all finger joints to 0
    p                    reprint geometry
    save                 print the config block to paste
    q                    quit
"""

# MUST precede Kit: Isaac Sim ships warp 1.13 in extscache whose fem/cache.py
# does `from warp._src.utils import warn`, removed in the site-packages 1.14.
# Binding warp.fem to 1.14 here means the later import hits the module cache
# instead of loading the 1.13 copy and dying on the missing symbol.
import warp.fem  # noqa: F401  isort: skip

import argparse

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--settle-steps", type=int, default=8, help="steps to run after each change")
parser.add_argument(
    "--freeze",
    action="store_true",
    help="do not step physics — render only, so the GUI's own joint authoring is not fought by this script",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import select
import sys

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
    FINGERTIP_BODY_NAMES,
    HANDLE_SEGMENT_P0_B,
    HANDLE_SEGMENT_P1_B,
)
from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import (
    _handle_segment_geometry,
    _sensor_peak_force,
    digit_handle_forces,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

from isaaclab_assets.props.lab_table import LAB_TABLE_HEIGHT

TASK = "IsaacContrib-Lift-Spatula-G1-v0"

ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HAND_JOINTS = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

# command word -> the joints it drives
ALIASES = {
    "index": ("right_hand_index_0_joint", "right_hand_index_1_joint"),
    "middle": ("right_hand_middle_0_joint", "right_hand_middle_1_joint"),
    "fingers": (
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ),
    "index0": ("right_hand_index_0_joint",),
    "index1": ("right_hand_index_1_joint",),
    "middle0": ("right_hand_middle_0_joint",),
    "middle1": ("right_hand_middle_1_joint",),
    "thumb0": ("right_hand_thumb_0_joint",),
    "thumb1": ("right_hand_thumb_1_joint",),
    "thumb2": ("right_hand_thumb_2_joint",),
}


def main():  # noqa: C901
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    # Deterministic: nothing may yank the pose away while hand-tuning it.
    # `reset_pregrasp` is the only randomizing term EventCfg still has --
    # `reset_grasp_map` / `randomize_*` were removed from the task, and assigning
    # them here only created new attributes nobody read.
    env_cfg.events.reset_pregrasp = None
    env_cfg.terminations.blade_contact = None
    env_cfg.terminations.spatula_dropped = None
    env_cfg.terminations.robot_exploded = None
    env_cfg.rewards.blade_penalty = None
    env_cfg.episode_length_s = 1.0e6
    env_cfg.viewer.eye = (0.42, -0.42, 0.98)
    env_cfg.viewer.lookat = (0.18, -0.17, 0.855)
    env_cfg.viewer.origin_type = "env"

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        device = uenv.device
        env.reset()

        names = list(robot.joint_names)
        limits = robot.data.joint_pos_limits.torch[0]
        idx = {n: names.index(n) for n in (*ARM_JOINTS, *HAND_JOINTS)}
        tips = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        tips.resolve(uenv.scene)
        zero = torch.zeros((1, uenv.action_space.shape[1]), device=device)
        env0 = torch.tensor([0], device=device)
        # start from the config's own default pose
        target = robot.data.joint_pos.torch[0].clone()

        def apply(steps):
            jp = target.unsqueeze(0)
            with torch.inference_mode():
                robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env0)
                for _ in range(steps):
                    env.step(zero)

        def geometry():
            dist, y = _handle_segment_geometry(
                uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips, SceneEntityCfg("spatula")
            )
            origin = uenv.scene.env_origins[0]
            print(f"  tabletop z = {LAB_TABLE_HEIGHT:.3f} m")
            sides = {}
            for c, i in enumerate(tips.body_ids):
                z = (robot.data.body_pos_w.torch[0, i, 2] - origin[2]).item()
                short = robot.body_names[i].replace("right_hand_", "").replace("_link", "")
                sides[short] = y[0, c].item()
                print(
                    f"  {short:<10s} dist {dist[0, c].item():.4f} m   side(y) {y[0, c].item():+.4f} m"
                    f"   z {z:.4f} ({z - LAB_TABLE_HEIGHT:+.4f} vs table)"
                )
            th = sides.get("thumb_2", 0.0)
            ix = sides.get("index_1", 0.0)
            md = sides.get("middle_1", 0.0)
            straddle = th * ix < 0 and th * md < 0
            print(f"  STRADDLE (thumb opposite BOTH fingers): {'YES' if straddle else 'no'}")
            # CONTACT is the trustworthy signal. The distances above are measured
            # from LINK ORIGINS, which sit at the knuckles — the pads reach
            # several cm further, so a 3-6 cm 'distance' can already be a pad
            # pressing. Force is unambiguous: nonzero means actually touching.
            f = digit_handle_forces(uenv)[0]
            blade = _sensor_peak_force(uenv, "hand_blade_contact")[0].item()
            print(
                f"  HANDLE FORCE [N]  thumb {f[0].item():7.3f}   index {f[1].item():7.3f}"
                f"   middle {f[2].item():7.3f}    | blade {blade:7.3f}"
            )
            touching = int(f[0] > 0.01) + int(f[1] > 0.01) + int(f[2] > 0.01)
            print(
                f"  DIGITS TOUCHING THE HANDLE: {touching}/3"
                + ("   <-- grasp candidate" if touching >= 2 and straddle else "")
            )

        def show_limits():
            print("  joint limits [rad]:")
            for n in HAND_JOINTS:
                lo, hi = limits[idx[n], 0].item(), limits[idx[n], 1].item()
                cur = target[idx[n]].item()
                print(f"    {n.replace('right_hand_', ''):<16s} [{lo:+.3f}, {hi:+.3f}]  now {cur:+.3f}")

        def save_block():
            print("\n# paste into DEFAULT_ARM_JOINT_POS in g1_spatula_lift_env_cfg.py")
            for n in (*ARM_JOINTS, *HAND_JOINTS):
                lo, hi = limits[idx[n], 0].item(), limits[idx[n], 1].item()
                v = min(max(target[idx[n]].item(), lo + 1e-3), hi - 1e-3)  # keep inside limits
                print(f'    "{n}": {v:.3f},')
            print()

        apply(args_cli.settle_steps)
        print("\n=== interactive finger posing ===")
        show_limits()
        geometry()
        print("\ncommands: index/middle/fingers/index0/index1/middle0/middle1/thumb0/thumb1/thumb2 <rad>")
        print("          arm <7 floats> | open | p | limits | save | q")
        sys.stdout.write("> ")
        sys.stdout.flush()

        running = True
        while running:
            # Keep the viewer live while waiting for input, and RE-ASSERT the
            # pose every step. A zero action does NOT hold this arm:
            # RelativeJointPositionAction sets target = action + CURRENT joint
            # pos, so with action 0 the PD target chases the joint and the arm
            # sags under gravity. Writing the state each step pins it still.
            # IDLE = RENDER ONLY. Never step physics here: stepping advances
            # gravity, and re-writing the pose each frame to compensate makes
            # the arm visibly jitter. Physics runs only inside apply(), when a
            # command actually changes something — so the arm sits dead still
            # between commands, and contact forces are still real because
            # apply() settles for --settle-steps before measuring.
            with torch.inference_mode():
                uenv.sim.render()
            if not select.select([sys.stdin], [], [], 0.0)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            parts = line.split()
            if not parts:
                sys.stdout.write("> ")
                sys.stdout.flush()
                continue
            cmd, rest = parts[0].lower(), parts[1:]
            try:
                if cmd in ("q", "quit", "exit"):
                    running = False
                elif cmd in ("p", "print"):
                    geometry()
                elif cmd == "limits":
                    show_limits()
                elif cmd == "save":
                    save_block()
                elif cmd == "open":
                    for n in HAND_JOINTS:
                        lo, hi = limits[idx[n], 0].item(), limits[idx[n], 1].item()
                        target[idx[n]] = min(max(0.0, lo), hi)
                    apply(args_cli.settle_steps)
                    geometry()
                elif cmd == "arm":
                    vals = [float(v) for v in rest]
                    if len(vals) != 7:
                        print("  arm needs 7 values (sh_pitch sh_roll sh_yaw elbow wr_roll wr_pitch wr_yaw)")
                    else:
                        for n, v in zip(ARM_JOINTS, vals):
                            target[idx[n]] = v
                        apply(args_cli.settle_steps)
                        geometry()
                elif cmd in ALIASES:
                    v = float(rest[0])
                    for n in ALIASES[cmd]:
                        lo, hi = limits[idx[n], 0].item(), limits[idx[n], 1].item()
                        cl = min(max(v, lo), hi)
                        if cl != v:
                            print(f"  {n}: {v:+.3f} clamped to {cl:+.3f} (limits [{lo:+.3f}, {hi:+.3f}])")
                        target[idx[n]] = cl
                    apply(args_cli.settle_steps)
                    geometry()
                else:
                    print(f"  unknown command {cmd!r}")
            except (ValueError, IndexError):
                print("  bad arguments — e.g. 'index 0.8' or 'arm -0.64 -0.14 -0.10 1.10 -1.20 0.50 -0.21'")
            if running:
                sys.stdout.write("> ")
                sys.stdout.flush()

        env.close()


if __name__ == "__main__":
    main()
