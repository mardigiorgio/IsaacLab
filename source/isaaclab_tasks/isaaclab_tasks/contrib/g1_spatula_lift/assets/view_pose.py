# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleport to a saved grasp-map stage, measure it, and render frames.

Visual confirmation for poses found by ``straddle_search.py``: printed geometry
proves the numbers, a rendered frame proves the pose is the claw you meant.

Run::

    ./isaaclab.sh -p .../assets/view_pose.py --pose /tmp/claw_cage.pt --frames-dir /tmp/frames
"""

import argparse

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--pose", required=True, help="path to a {'stages': [...]} file")
parser.add_argument("--frames-dir", default=None, help="write rendered PNGs here")
parser.add_argument("--settle-steps", type=int, default=6)
parser.add_argument("--camera", default="three_quarter", help="which framing to render")
parser.add_argument(
    "--sweep-close",
    action="store_true",
    help="sweep index/middle curl and thumb flexion from the loaded pose and report whether the tips converge on the handle",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

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
from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import _handle_segment_geometry
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

CAMERAS = {
    # along the handle axis: shows the thumb-vs-fingers straddle head on
    "along_handle": ((0.62, -0.18, 0.95), (0.10, -0.18, 0.86)),
    # across the handle: shows tip height against the tabletop
    "across_handle": ((0.16, -0.62, 0.95), (0.16, -0.14, 0.86)),
    # three-quarter view, matching how the hardware photo was taken
    "three_quarter": ((0.55, -0.60, 1.05), (0.14, -0.16, 0.86)),
}


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
    env_cfg.episode_length_s = 120.0

    stage = torch.load(args_cli.pose, weights_only=True)["stages"][0]
    render = args_cli.frames_dir is not None
    if render:
        # kitless runs expose no viewport_camera_controller; the framing must be
        # baked into the cfg before launch_simulation, as author_grasp_map does
        eye, lookat = CAMERAS[args_cli.camera]
        env_cfg.viewer.eye = eye
        env_cfg.viewer.lookat = lookat
        env_cfg.viewer.origin_type = "env"

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if render else None)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        spatula = uenv.scene["spatula"]
        device = uenv.device
        env.reset()

        joint_pos = stage["joint_pos"].to(device).unsqueeze(0)
        with torch.inference_mode():
            robot.write_joint_state_to_sim(
                joint_pos, torch.zeros_like(joint_pos), env_ids=torch.tensor([0], device=device)
            )
            zero = torch.zeros((1, uenv.action_space.shape[1]), device=device)
            for _ in range(args_cli.settle_steps):
                env.step(zero)

        names = list(robot.joint_names)
        print(f"[view] stage '{stage['name']}' arm joints [rad]:")
        for n in ARM_JOINTS:
            print(f"[view]   {n:<30s} {robot.data.joint_pos.torch[0, names.index(n)].item():+.3f}")
        print("[view] hand joints [rad]:")
        for n in names:
            if n.startswith("right_hand_"):
                print(f"[view]   {n:<30s} {robot.data.joint_pos.torch[0, names.index(n)].item():+.3f}")

        tips = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        tips.resolve(uenv.scene)
        dist, y = _handle_segment_geometry(
            uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips, SceneEntityCfg("spatula")
        )
        origin = uenv.scene.env_origins[0]
        print(f"[view] tabletop z = {LAB_TABLE_HEIGHT:.3f} m")
        for c, i in enumerate(tips.body_ids):
            pos = robot.data.body_pos_w.torch[0, i] - origin
            print(
                f"[view]   {robot.body_names[i]:<28s} seg-dist {dist[0, c].item():.4f} m"
                f"  local-y {y[0, c].item():+.4f} m  z {pos[2].item():.4f}"
                f"  ({pos[2].item() - LAB_TABLE_HEIGHT:+.4f} above table)"
            )
        spat = spatula.data.root_pos_w.torch[0] - origin
        print(f"[view] spatula root env-frame: {[round(v, 4) for v in spat.tolist()]}")

        if args_cli.sweep_close:
            sweep_close(env, uenv, robot, stage, device)

        if render:
            os.makedirs(args_cli.frames_dir, exist_ok=True)
            from PIL import Image

            # render the OPEN cage and, when the stage carries one, the CLOSED
            # grasp — they are different poses and only the pair shows the rake
            poses = [("open", stage["joint_pos"])]
            if "joint_pos_closed" in stage:
                poses.append(("closed", stage["joint_pos_closed"]))
            for tag, jp_cpu in poses:
                jp = jp_cpu.to(device).unsqueeze(0)
                with torch.inference_mode():
                    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=torch.tensor([0], device=device))
                    for _ in range(args_cli.settle_steps):
                        env.step(zero)
                frame = env.render()
                if frame is None:
                    print("[view] render returned None — no renderer attached")
                    break
                path = os.path.join(args_cli.frames_dir, f"{args_cli.camera}_{tag}.png")
                Image.fromarray(frame).save(path)
                print(f"[view] wrote {path}")

        env.close()


def sweep_close(env, uenv, robot, stage, device):
    """Close the digits from the loaded pose and report tip-to-handle distance.

    A cage is only useful if closing RAKES the handle in. This holds the arm at
    the found pose and drives index/middle curl (and thumb flexion) from open to
    fully closed, teleporting each fraction so the answer is kinematic. If the
    minimum distance grows instead of shrinking, the fingers sweep AWAY from the
    handle and the pose cages something it cannot pick up.
    """
    names = list(robot.joint_names)
    limits = robot.data.joint_pos_limits.torch[0]
    tips = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
    tips.resolve(uenv.scene)
    col = [robot.body_names[i] for i in tips.body_ids]
    ti, ii, mi = (
        col.index(n) for n in ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")
    )
    finger_ids = [names.index(n) for n in names if n.startswith("right_hand_") and ("index" in n or "middle" in n)]
    t1, t2 = names.index("right_hand_thumb_1_joint"), names.index("right_hand_thumb_2_joint")
    base = stage["joint_pos"].to(device)
    zero = torch.zeros((1, uenv.action_space.shape[1]), device=device)

    def interp(j, frac):
        return limits[j, 0].item() + frac * (limits[j, 1].item() - limits[j, 0].item())

    print("[view] closure sweep (arm held at the found pose):")
    print("[view]   fc  thumb_f | thumb d/y        index d/y        middle d/y")
    for fc in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        for tf in (0.5, 0.8):
            jp = base.clone().unsqueeze(0)
            for j in finger_ids:
                jp[0, j] = interp(j, fc)
            jp[0, t1] = interp(t1, tf)
            jp[0, t2] = interp(t2, tf)
            with torch.inference_mode():
                robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=torch.tensor([0], device=device))
                for _ in range(4):
                    env.step(zero)
                d, y = _handle_segment_geometry(
                    uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips, SceneEntityCfg("spatula")
                )
            print(
                f"[view]   {fc:.1f}   {tf:.1f}   | {d[0, ti].item():.3f}/{y[0, ti].item():+.3f}"
                f"      {d[0, ii].item():.3f}/{y[0, ii].item():+.3f}"
                f"      {d[0, mi].item():.3f}/{y[0, mi].item():+.3f}"
            )


if __name__ == "__main__":
    main()
