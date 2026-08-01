# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probes for the G1 spatula-lift task: names, settle pose, ready pose.

Runs kit-less on the Newton preset (the training backend) via
:func:`isaaclab.app.launch_simulation`::

    ./isaaclab.sh -p .../assets/probe_ready_pose.py --mode names
    ./isaaclab.sh -p .../assets/probe_ready_pose.py --mode settle
    ./isaaclab.sh -p .../assets/probe_ready_pose.py --mode pose

Modes:

* ``names``  — print the articulation body/joint names; gate: the TriHand
  distal links assumed by the env cfg must exist.
* ``settle`` — reset 3 times, run 2 s of zero actions each, print the settled
  spatula root pose (env frame, xyzw quat) and grasp-point height; gates:
  repeatability across resets and handle clearance above the table. Bake the
  printed values into ``SPATULA_SPAWN_POS/QUAT`` / ``REST_SPATULA_Z``.
* ``pose``   — run 2 s of zero actions and print the palm/fingertip/grasp
  geometry; iterate ``DEFAULT_ARM_JOINT_POS`` until palm→grasp < 0.12 m with
  the palm above the handle. Pair with ``--visualizer newton`` for a visual
  check.
"""

import argparse

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["names", "labels", "settle", "pose"], default="names")
parser.add_argument("--steps", type=int, default=120, help="zero-action control steps per settle run (60 Hz control)")
parser.add_argument(
    "--candidates",
    default=None,
    help=(
        "pose mode: ';'-separated candidate right-arm poses, each 7 comma-separated values"
        " (sh_pitch,sh_roll,sh_yaw,elbow,wr_roll,wr_pitch,wr_yaw). Each candidate is driven"
        " with relative actions and the resulting palm/tip geometry printed."
    ),
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
    HANDLE_GRASP_OFFSET_B,
    LAB_TABLE_HEIGHT,
    REST_SPATULA_Z,
)
from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import _handle_grasp_point_w
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

TASK = "IsaacContrib-Lift-Spatula-G1-v0"


def _settle(env, steps):
    zero = torch.zeros((1, env.unwrapped.action_space.shape[1]), device=env.unwrapped.device)
    with torch.inference_mode():
        for _ in range(steps):
            env.step(zero)


def _spatula_state(env):
    uenv = env.unwrapped
    spatula = uenv.scene["spatula"]
    origin = uenv.scene.env_origins[0]
    pos = (spatula.data.root_pos_w.torch[0] - origin).tolist()
    quat = spatula.data.root_quat_w.torch[0].tolist()
    grasp_w = _handle_grasp_point_w(uenv, HANDLE_GRASP_OFFSET_B, SceneEntityCfg("spatula"))[0]
    grasp = (grasp_w - origin).tolist()
    return pos, quat, grasp


def mode_names(env):
    robot = env.unwrapped.scene["robot"]
    print(f"[probe] {len(robot.body_names)} bodies: {robot.body_names}")
    print(f"[probe] {len(robot.joint_names)} joints: {robot.joint_names}")
    for name in ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"):
        status = "OK" if name in robot.body_names else "MISSING — fix the env-cfg fingertip links!"
        print(f"[probe] distal link {name}: {status}")


def mode_labels(env):
    """Print the Newton model body/shape labels the contact-sensor patterns must match."""
    from isaaclab_newton.physics.newton_manager import NewtonManager

    model = NewtonManager._model
    bodies = [b for b in list(model.body_label) if "env_0" in b or "/" not in b]
    shapes = [s for s in list(model.shape_label) if "env_0" in s or "/" not in s]
    print(f"[probe] env_0 body labels ({len(bodies)}):")
    for b in bodies:
        print(f"[probe]   body: {b}")
    print("[probe] env_0 shape labels containing Spatula/hand:")
    for s in shapes:
        if "Spatula" in s or "hand" in s:
            print(f"[probe]   shape: {s}")


def mode_settle(env):
    poses = []
    for run in range(3):
        env.reset()
        _settle(env, args_cli.steps)
        pos, quat, grasp = _spatula_state(env)
        poses.append(torch.tensor(pos + quat))
        print(f"[probe] run {run}: root pos {[round(v, 4) for v in pos]} quat(xyzw) {[round(v, 4) for v in quat]}")
        print(
            f"[probe] run {run}: grasp point {[round(v, 4) for v in grasp]}"
            f" (z - table = {grasp[2] - LAB_TABLE_HEIGHT:.4f}, cfg REST_SPATULA_Z - table ="
            f" {REST_SPATULA_Z - LAB_TABLE_HEIGHT:.4f})"
        )
    spread = (torch.stack(poses).max(dim=0).values - torch.stack(poses).min(dim=0).values).max().item()
    print(f"[probe] max pose spread across runs: {spread:.5f} (gate: < 0.005)")
    clearance = grasp[2] - LAB_TABLE_HEIGHT - 0.0035  # half the 7 mm handle thickness
    # ~5 mm at the baked spawn: no room for fingertips UNDER the handle — the
    # workable grasp is a pinch on the flat side faces / top of the handle
    print(f"[probe] handle underside clearance: {clearance:.4f} m (side-pinch grasp expected below ~0.01)")


_ARM_ORDER = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _print_hand_geometry(env, label=""):
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    palm_idx = robot.body_names.index("right_hand_palm_link")
    palm = (robot.data.body_pos_w.torch[0, palm_idx] - uenv.scene.env_origins[0]).tolist()
    _, _, grasp = _spatula_state(env)
    dist = sum((p - g) ** 2 for p, g in zip(palm, grasp)) ** 0.5
    print(f"[probe]{label} palm {[round(v, 4) for v in palm]}")
    print(f"[probe]{label} grasp point {[round(v, 4) for v in grasp]}")
    print(f"[probe]{label} palm->grasp distance {dist:.4f} m (target < 0.12, palm above handle: palm z > grasp z)")
    for tip in ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"):
        idx = robot.body_names.index(tip)
        tp = (robot.data.body_pos_w.torch[0, idx] - uenv.scene.env_origins[0]).tolist()
        print(f"[probe]{label} {tip}: {[round(v, 4) for v in tp]}")
    lidx = robot.body_names.index("left_hand_palm_link")
    lp = (robot.data.body_pos_w.torch[0, lidx] - uenv.scene.env_origins[0]).tolist()
    print(f"[probe]{label} left palm: {[round(v, 4) for v in lp]} (down-by-side: y < -0.35, z < 0.65)")
    pelvis = (robot.data.root_pos_w.torch[0] - uenv.scene.env_origins[0]).tolist()
    pquat = robot.data.root_quat_w.torch[0].tolist()
    sidx = robot.body_names.index("left_shoulder_pitch_link")
    sp = (robot.data.body_pos_w.torch[0, sidx] - uenv.scene.env_origins[0]).tolist()
    print(f"[probe]{label} pelvis {[round(v, 4) for v in pelvis]} quat(xyzw) {[round(v, 3) for v in pquat]}")
    print(f"[probe]{label} left shoulder {[round(v, 4) for v in sp]}")
    for jn in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"):
        jid = robot.joint_names.index(jn)
        default = robot.data.default_joint_pos.torch[0, jid].item()
        actual = robot.data.joint_pos.torch[0, jid].item()
        target = None
        for attr in ("joint_pos_target", "joint_target"):
            data = getattr(robot.data, attr, None)
            if data is not None:
                target = data.torch[0, jid].item()
                break
        print(f"[probe]{label} {jn}: default={default:.3f} actual={actual:.3f} target={target}")


def _drive_arm_to(env, targets: dict, steps: int = 90):
    """Drive the right arm to *targets* [rad] with proportional relative actions."""
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    try:
        action_term = uenv.action_manager.get_term("upper_body")
    except (KeyError, ValueError):
        raise SystemExit(
            "--candidates drives raw arm joints and needs the joint-space action variant;"
            " the env now uses task-space IK actions (candidate mode retired)"
        )
    act_joint_ids = action_term._joint_ids
    act_names = (
        list(robot.joint_names) if act_joint_ids == slice(None) else [robot.joint_names[i] for i in act_joint_ids]
    )
    slots = {name: act_names.index(name) for name in targets}
    action = torch.zeros((1, uenv.action_space.shape[1]), device=uenv.device)
    with torch.inference_mode():
        for _ in range(steps):
            for name, slot in slots.items():
                jid = robot.joint_names.index(name)
                err = targets[name] - robot.data.joint_pos.torch[0, jid].item()
                action[0, slot] = max(-1.0, min(1.0, err / 0.1))
            env.step(action)


def mode_pose(env):
    env.reset()
    _settle(env, args_cli.steps)
    _print_hand_geometry(env, " [ready]")
    if args_cli.candidates:
        for c_idx, cand in enumerate(args_cli.candidates.split(";")):
            values = [float(v) for v in cand.split(",")]
            targets = dict(zip(_ARM_ORDER, values))
            _drive_arm_to(env, targets)
            print(f"[probe] --- candidate {c_idx}: {[round(v, 3) for v in values]}")
            _print_hand_geometry(env, f" [cand{c_idx}]")


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    # parse_env_cfg collapses PresetCfg wrappers to their default (PhysX); the
    # Newton preset must be re-applied BEFORE launch_simulation so the scan
    # picks the kitless backend
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    # probes measure the NOMINAL reset — no curated grasp-map stage, no
    # reset randomization
    env_cfg.events.reset_grasp_map = None
    env_cfg.events.randomize_right_arm = None
    env_cfg.events.randomize_spatula_pose = None
    if args_cli.mode in ("names", "labels"):
        # these modes exist to DISCOVER the link names / model labels the
        # blade-contact sensor assumes, so strip the sensor and its dependents
        env_cfg.scene.hand_blade_contact = None
        env_cfg.terminations.blade_contact = None
        env_cfg.rewards.blade_penalty = None
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        try:
            env.reset()
            {
                "names": mode_names,
                "labels": mode_labels,
                "settle": mode_settle,
                "pose": mode_pose,
            }[args_cli.mode](env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
