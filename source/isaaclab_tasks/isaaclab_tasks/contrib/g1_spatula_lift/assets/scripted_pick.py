# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted-pick existence proof for the spatula-lift task.

Drives a fixed close-and-lift sequence through the task's own action term and
measures whether a REAL pick exists under the current contact physics:

* hold-phase per-digit force  <= 10x the spatula's 0.651 N weight
* release ejection speed      <  0.5 m/s within the first 50 ms of opening
* the spatula actually rises with the hand and stays gripped

Run on the Newton preset (kit-less), fixed-step or adaptive::

    ./isaaclab.sh -p .../assets/scripted_pick.py
    ./isaaclab.sh -p .../assets/scripted_pick.py --adaptive

The fixed-vs-adaptive pair doubles as a solver-fidelity A/B for the
newton-adaptive research: same trajectory, same gates, different integrator.
"""

import argparse
import os

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", action="store_true", help="run the MJWarp adaptive-timestep solver")
parser.add_argument("--curl", type=float, default=0.55, help="index/middle flexion fraction at close")
parser.add_argument("--thumb-curl", type=float, default=0.40, help="thumb flexion fraction at close")
parser.add_argument("--descend", type=float, default=0.020, help="palm descent before closing [m], via elbow/wrist")
parser.add_argument("--frames-dir", default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
    CONTACT_GATE_N,
    LIFT_SUCCESS_Z,
    REST_SPATULA_Z,
)
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
    FINGERTIP_BODY_NAMES,
    HANDLE_SEGMENT_P0_B,
    HANDLE_SEGMENT_P1_B,
)
from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import (
    _handle_segment_geometry,
    _opposed_handle_contact,
    _sensor_peak_force,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

TASK = "IsaacContrib-Lift-Spatula-G1-v0"
SPATULA_WEIGHT_N = 0.0664 * 9.81

ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
READY = (-0.60, -0.15, -0.10, 0.70, 0.30, -0.20, -0.30)
# descend: fold the elbow and drop the shoulder so the finger aperture
# reaches handle height, wrist pitching to keep the claw level (tuned for
# the 0.85 gantry mount — the forearm now clears the table edge)
DESCEND = (-0.59, -0.15, -0.10, 1.00, 0.30, -0.30, -0.30)
# lift: unfold the elbow and raise the shoulder, wrist compensating
LIFT = (-0.85, -0.15, -0.10, 0.75, 0.30, -0.15, -0.30)


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    if args_cli.adaptive:
        env_cfg.sim.physics.solver_cfg.adaptive = True
    # deterministic nominal start
    env_cfg.events.reset_grasp_map = None
    env_cfg.events.randomize_right_arm = None
    env_cfg.events.randomize_spatula_pose = None
    env_cfg.episode_length_s = 60.0
    env_cfg.viewer.eye = (0.9, -0.85, 1.15)
    env_cfg.viewer.lookat = (0.2, -0.18, 0.88)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.frames_dir else None)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        spatula = uenv.scene["spatula"]
        act_term = uenv.action_manager.get_term("upper_body")
        act_ids = act_term._joint_ids
        act_names = list(robot.joint_names) if act_ids == slice(None) else [robot.joint_names[i] for i in act_ids]
        arm_slots = {n: act_names.index(n) for n in ARM_JOINTS}
        finger_ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("right_hand_")]
        finger_slots = {robot.joint_names[i]: act_names.index(robot.joint_names[i]) for i in finger_ids}
        limits = robot.data.joint_pos_limits.torch[0]
        # flexion target = larger-|limit| bound scaled by curl (thumb sign-correct)
        finger_targets = {}
        for i in finger_ids:
            name = robot.joint_names[i]
            lo, hi = limits[i, 0].item(), limits[i, 1].item()
            bound = lo if abs(lo) > abs(hi) else hi
            frac = args_cli.thumb_curl if "thumb" in name else args_cli.curl
            finger_targets[name] = frac * bound

        log = {k: [] for k in ("phase", "thumb", "index", "middle", "spat_z", "spat_speed", "palm_z", "contact")}
        frame_count = [0]

        def snap(phase):
            log["phase"].append(phase)
            log["thumb"].append(_sensor_peak_force(uenv, "thumb_handle_contact")[0].item())
            log["index"].append(_sensor_peak_force(uenv, "index_handle_contact")[0].item())
            log["middle"].append(_sensor_peak_force(uenv, "middle_handle_contact")[0].item())
            log["contact"].append(bool(_opposed_handle_contact(uenv, CONTACT_GATE_N)[0].item()))
            log["spat_z"].append((spatula.data.root_pos_w.torch[0, 2] - uenv.scene.env_origins[0, 2]).item())
            log["spat_speed"].append(torch.linalg.vector_norm(spatula.data.root_lin_vel_w.torch[0]).item())
            palm_idx = robot.body_names.index("right_hand_palm_link")
            log["palm_z"].append((robot.data.body_pos_w.torch[0, palm_idx, 2] - uenv.scene.env_origins[0, 2]).item())

        tips_cfg = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        tips_cfg.resolve(uenv.scene)
        tip_names = [robot.body_names[i] for i in tips_cfg.body_ids]

        def print_geometry(tag):
            dist, y = _handle_segment_geometry(uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips_cfg, SceneEntityCfg("spatula"))
            parts = [f"{n.split('_')[2]}: d={dist[0, i]:.3f} y={y[0, i]:+.3f}" for i, n in enumerate(tip_names)]
            print(f"[scripted_pick] {tag} tips->segment: " + "  ".join(parts))
            fj = {robot.joint_names[i]: robot.data.joint_pos.torch[0, i].item() for i in finger_ids}
            tgts = {n: finger_targets[n] for n in fj}
            print(f"[scripted_pick] {tag} fingers actual/target: " + "  ".join(f"{n.split('right_hand_')[1]}: {v:+.2f}/{tgts[n]:+.2f}" for n, v in fj.items()))

        def save_frame(tag):
            if not args_cli.frames_dir:
                return
            frame = env.render()
            if frame is not None:
                from PIL import Image

                os.makedirs(args_cli.frames_dir, exist_ok=True)
                Image.fromarray(frame).save(os.path.join(args_cli.frames_dir, f"{frame_count[0]:02d}_{tag}.png"))
                frame_count[0] += 1

        def drive(arm_targets, finger_frac_targets, steps, phase):
            """Proportional relative actions toward joint targets (probe pattern)."""
            action = torch.zeros((1, uenv.action_space.shape[1]), device=uenv.device)
            with torch.inference_mode():
                for _ in range(steps):
                    for name, slot in arm_slots.items():
                        jid = robot.joint_names.index(name)
                        err = arm_targets[ARM_JOINTS.index(name)] - robot.data.joint_pos.torch[0, jid].item()
                        action[0, slot] = max(-1.0, min(1.0, err / 0.1))
                    for name, slot in finger_slots.items():
                        jid = robot.joint_names.index(name)
                        tgt = finger_frac_targets.get(name, 0.0)
                        err = tgt - robot.data.joint_pos.torch[0, jid].item()
                        action[0, slot] = max(-1.0, min(1.0, err / 0.2))
                    env.step(action)
                    snap(phase)

        palm_idx = robot.body_names.index("right_hand_palm_link")

        def servo_palm(target, hold_fingers, steps, phase, arm_base):
            """Closed-loop palm-position servo: diagonal gains on measured error.

            Signs calibrated from probe data: elbow fold moves the palm down
            and back; more-negative shoulder pitch moves it forward and up.
            The wrist targets stay at the arm_base values (claw attitude).
            """
            action = torch.zeros((1, uenv.action_space.shape[1]), device=uenv.device)
            tgt = torch.tensor(target, device=uenv.device)
            with torch.inference_mode():
                for _ in range(steps):
                    palm = robot.data.body_pos_w.torch[0, palm_idx] - uenv.scene.env_origins[0]
                    e = (tgt - palm).tolist()  # (ex, ey, ez)
                    # arm joints toward base pose, plus servo corrections
                    for name, slot in arm_slots.items():
                        jid = robot.joint_names.index(name)
                        base_err = arm_base[ARM_JOINTS.index(name)] - robot.data.joint_pos.torch[0, jid].item()
                        a = base_err / 0.1
                        if name == "right_elbow_joint":
                            a += -e[2] / 0.02  # too high -> fold more
                        elif name == "right_shoulder_pitch_joint":
                            a += -e[1] / 0.04  # short of the handle -> swing forward
                        elif name == "right_shoulder_yaw_joint":
                            a += -e[0] / 0.06
                        action[0, slot] = max(-1.0, min(1.0, a))
                    for name, slot in finger_slots.items():
                        jid = robot.joint_names.index(name)
                        err = hold_fingers.get(name, 0.0) - robot.data.joint_pos.torch[0, jid].item()
                        action[0, slot] = max(-1.0, min(1.0, err / 0.2))
                    env.step(action)
                    snap(phase)

        env.reset()
        open_hand = {n: 0.0 for n in finger_targets}
        drive(READY, open_hand, 30, "settle")
        save_frame("ready")
        # hover the palm so the open aperture brackets the handle: grasp point
        # +4.5 cm up, 1 cm back (tips land ahead of the palm)
        grasp_xyz = (0.248, -0.187, 0.890)
        servo_palm(grasp_xyz, open_hand, 90, "descend", READY)
        save_frame("descend")
        print_geometry("descend")
        servo_palm(grasp_xyz, finger_targets, 50, "close", READY)
        save_frame("close")
        print_geometry("close")
        drive(LIFT, finger_targets, 80, "lift")
        save_frame("lift")
        drive(LIFT, finger_targets, 60, "hold")
        save_frame("hold")
        drive(LIFT, open_hand, 3, "release3")  # first 50 ms after opening starts
        drive(LIFT, open_hand, 27, "release")
        save_frame("released")

        # ---- verdicts ----
        def sel(*phases):
            return [i for i, p in enumerate(log["phase"]) if p in phases]

        hold = sel("hold")
        rel3 = sel("release3")
        t = torch.tensor
        hold_forces = {d: t([log[d][i] for i in hold]) for d in ("thumb", "index", "middle")}
        hold_contact = [log["contact"][i] for i in hold]
        hold_z = t([log["spat_z"][i] for i in hold])
        palm_rise = log["palm_z"][hold[-1]] - log["palm_z"][sel("close")[0]]
        spat_rise = hold_z[-1].item() - REST_SPATULA_Z
        rel_speed = max(log["spat_speed"][i] for i in rel3) if rel3 else float("nan")
        peak_force = max(f.max().item() for f in hold_forces.values())

        print(f"\n[scripted_pick] solver = {'ADAPTIVE' if args_cli.adaptive else 'fixed-step'}")
        print(f"[scripted_pick] spatula weight = {SPATULA_WEIGHT_N:.3f} N; force gate = {10 * SPATULA_WEIGHT_N:.2f} N")
        for d in ("thumb", "index", "middle"):
            f = hold_forces[d]
            print(f"[scripted_pick] hold {d:6s} force [N]: mean {f.mean():.2f}  p95 {f.quantile(0.95):.2f}  max {f.max():.2f}")
        print(f"[scripted_pick] hold contact-gate fraction: {sum(hold_contact) / max(len(hold_contact), 1):.2f}")
        print(f"[scripted_pick] palm rise {palm_rise:+.3f} m, spatula rise {spat_rise:+.3f} m (carry target {LIFT_SUCCESS_Z - REST_SPATULA_Z:+.3f})")
        print(f"[scripted_pick] release speed in first 50 ms: {rel_speed:.3f} m/s (gate < 0.5)")
        lifted_ok = spat_rise > 0.10 and sum(hold_contact) / max(len(hold_contact), 1) > 0.8
        force_ok = peak_force <= 10 * SPATULA_WEIGHT_N
        release_ok = rel_speed < 0.5
        print(f"[scripted_pick] GATES: lift={'PASS' if lifted_ok else 'FAIL'}  force={'PASS' if force_ok else 'FAIL'}  release={'PASS' if release_ok else 'FAIL'}")
        env.close()


if __name__ == "__main__":
    main()
