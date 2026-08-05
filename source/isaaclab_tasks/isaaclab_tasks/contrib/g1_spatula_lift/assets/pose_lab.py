# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive pose lab: pose the hand in the GUI, read the joint values live.

Opens the spatula scene with the GUI, teleports the robot to a starting pose,
then runs in FREE-POSE mode — every step the PD target is rewritten to the
joint's own current position, so the arm and fingers stay wherever you drag
them instead of springing back. Drag joints in the viewport or type values into
the Property panel; the right-hand joint angles and fingertip heights print on a
timer so you can read off the pose you want.

Run::

    ./isaaclab.sh -p .../assets/pose_lab.py                    # start from the grasp map
    ./isaaclab.sh -p .../assets/pose_lab.py --pose dict        # start from PREGRASP_JOINT_POS
    ./isaaclab.sh -p .../assets/pose_lab.py --save /tmp/my_pose.pt

Press Ctrl+C to exit; with ``--save`` the last pose is written as a one-stage
grasp map, ready to drop in as ``grasp_map.pt``.
"""

import argparse
import sys

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--pose", default="map", help="starting pose: 'map' (grasp_map.pt), 'dict', or 'default'")
parser.add_argument("--save", default=None, help="write the final pose here as a {'stages': [...]} file")
parser.add_argument("--print-every", type=float, default=2.0, help="seconds between joint printouts")
parser.add_argument("--pose-file", default="/tmp/spatula_pose.json", help="edit this file to move the joints")
add_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# this tool is only useful with a window: default the visualizer to kit so the
# GUI opens without the caller having to remember --visualizer kit
if getattr(args_cli, "visualizer", None) in (None, "", "none"):
    args_cli.visualizer = "kit"
args_cli.headless = False
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "IsaacContrib-Lift-Spatula-G1-v0"
RIGHT_HAND = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def reset_spatula(env, eid, dev):
    """Teleport the spatula back to its spawn pose, at rest.

    Module-level so it can be imported and exercised directly by a test rather
    than only through the button, which swallows exceptions silently.
    """
    import torch as _t

    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
        SPATULA_SPAWN_POS,
        SPATULA_SPAWN_QUAT,
    )

    name = None
    for cand in ("spatula", "object"):
        try:
            obj = env.scene[cand]
            name = cand
            break
        except Exception:
            continue
    if name is None:
        raise KeyError(f"no spatula entity in scene; have {list(env.scene.keys())}")

    before = (obj.data.root_pos_w.torch[0] - env.scene.env_origins[0]).tolist()
    pose = _t.tensor([*SPATULA_SPAWN_POS, *SPATULA_SPAWN_QUAT], device=dev).unsqueeze(0)
    pose[:, :3] += env.scene.env_origins[:1]
    obj.write_root_pose_to_sim(pose, env_ids=eid)
    obj.write_root_velocity_to_sim(_t.zeros(1, 6, device=dev), env_ids=eid)
    obj.write_data_to_sim()
    env.scene.update(0.0)
    after = (obj.data.root_pos_w.torch[0] - env.scene.env_origins[0]).tolist()
    return name, before, after


def reset_arm(env, eid, dev, start_q):
    """Put the arm/hand back to the pose the lab started from.

    Writes the joint state AND the PD target together, then flushes, so the
    arm snaps back instead of being dragged there by the controller.
    """
    import torch as _t

    robot = env.scene["robot"]
    q = start_q.clone()
    robot.write_joint_state_to_sim(q, _t.zeros_like(q), env_ids=eid)
    robot.set_joint_position_target(q, env_ids=eid)
    robot.write_data_to_sim()
    return q


def main():  # noqa: C901
    """Open the GUI, hold the pose, and stream joint values.

    Complexity is over the linter default by one branch: this is an interactive
    authoring tool whose body is a UI layout plus an event loop, and splitting it
    would spread shared closure state across helpers for no readability gain.
    """
    import os
    import time

    env_cfg = parse_env_cfg(TASK, num_envs=1)
    env_cfg.scene.num_envs = 1
    # no exploration noise, no resets yanking the pose away
    env_cfg.events.reset_pregrasp = None
    # Newton, not the PhysX default: the spatula's raw triangle colliders are
    # rejected on a dynamic body by PhysX, and the right-hand contact sensors
    # fail to initialise there. This script has no `presets=` CLI hook.
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import PhysicsCfg

    env_cfg.sim.physics = PhysicsCfg().newton_mjwarp

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
            FINGERTIP_BODY_NAMES,
            HANDLE_SEGMENT_P0_B,
            HANDLE_SEGMENT_P1_B,
            LAB_TABLE_HEIGHT,
            PREGRASP_JOINT_POS,
        )
        from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import _handle_segment_geometry

        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        dev = env.device
        eid = torch.tensor([0], device=dev)

        fc = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        fc.resolve(env.scene)
        oc = SceneEntityCfg("spatula")
        oc.resolve(env.scene)
        tip_names = [robot.body_names[i] for i in fc.body_ids]

        # ---- starting pose -------------------------------------------------
        q = robot.data.default_joint_pos.torch[:1].clone()
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args_cli.pose == "map":
            gm = torch.load(os.path.join(here, "grasp_map.pt"), weights_only=True)
            q = gm["stages"][0]["joint_pos"].to(dev).unsqueeze(0).clone()
            print("[pose_lab] started from grasp_map.pt stage 'pre_grasp_caged'")
        elif args_cli.pose == "dict":
            lim = robot.data.joint_pos_limits.torch[0]
            for n, v in PREGRASP_JOINT_POS.items():
                if n in robot.joint_names:
                    i = robot.joint_names.index(n)
                    q[0, i] = min(max(v, lim[i, 0].item()), lim[i, 1].item())
            print("[pose_lab] started from PREGRASP_JOINT_POS")
        else:
            print("[pose_lab] started from the env default pose")

        # write the target ALONGSIDE the state: writing state alone leaves the PD
        # target at the previous pose, so the controller visibly drags the arm
        # into place over the next frames instead of it simply being there.
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=eid)
        robot.set_joint_position_target(q, env_ids=eid)

        # ---- live pose file -------------------------------------------------
        # Dragging joints in the viewport does not work while the solver is
        # stepping: Newton owns the articulation state and re-asserts it every
        # step, so GUI edits are overwritten instantly. Instead, edit a JSON
        # file and the pose updates in the window as soon as you save.
        import json

        pose_file = args_cli.pose_file
        editable = RIGHT_ARM + RIGHT_HAND

        def dump(path):
            cur = robot.data.joint_pos.torch
            with open(path, "w") as fh:
                json.dump(
                    {n: round(cur[0, robot.joint_names.index(n)].item(), 4) for n in editable},
                    fh,
                    indent=2,
                )

        # ---- slider window ---------------------------------------------------
        # GUI joint gizmos and the Property panel write USD drive targets, which
        # the Newton backend does not read at runtime (it owns articulation
        # state once stepping). These sliders call set_joint_position_target
        # directly -- the same API the sim uses -- so they actually move the
        # robot on this backend.
        _start_q = robot.data.joint_pos.torch[:1].clone()
        # ONE owned target tensor. Never re-read joint_pos_target inside a
        # callback: it lags the buffer, so fourteen slider callbacks each
        # cloning it would each restore stale values and undo a reset.
        _tgt = _start_q.clone()
        _syncing = {"on": False}

        try:
            import omni.ui as ui

            _win = ui.Window("Spatula Pose Lab", width=460, height=600)
            _sliders = {}
            _status_box = {}

            def _say(msg):
                print(f"[pose_lab] {msg}", flush=True)
                lab = _status_box.get("l")
                if lab is not None:
                    lab.text = msg

            def _make_cb(joint_index):
                def _cb(model):
                    if _syncing["on"]:
                        return  # programmatic resync, not a user drag
                    _tgt[0, joint_index] = model.get_value_as_float()
                    robot.set_joint_position_target(_tgt, env_ids=eid)
                    robot.write_data_to_sim()

                return _cb

            def _reset_arm():
                """Button handler: restore the start pose and resync the sliders."""
                try:
                    _tgt[:] = _start_q
                    reset_arm(env, eid, dev, _start_q)
                    _syncing["on"] = True
                    try:
                        for _jn, _sl in _sliders.items():
                            _sl.model.set_value(_start_q[0, robot.joint_names.index(_jn)].item())
                    finally:
                        _syncing["on"] = False
                    _say("arm reset to start pose")
                except Exception as err:
                    import traceback

                    _say(f"RESET ARM FAILED: {type(err).__name__}: {err}")
                    traceback.print_exc()

            def _reset_spatula():
                """Button handler. Never let an exception die silently in the UI."""
                try:
                    n, b, aft = reset_spatula(env, eid, dev)
                    _say(
                        f"spatula '{n}': ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) -> ({aft[0]:.3f},{aft[1]:.3f},{aft[2]:.3f})"
                    )
                except Exception as err:
                    import traceback

                    _say(f"RESET SPATULA FAILED: {type(err).__name__}: {err}")
                    traceback.print_exc()

            with _win.frame:
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=3, height=0):
                        with ui.HStack(height=32, spacing=6):
                            ui.Button("RESET ARM", clicked_fn=_reset_arm)
                            ui.Button("RESET SPATULA", clicked_fn=_reset_spatula)
                        _status = ui.Label("ready", height=22, word_wrap=True)
                        _status_box["l"] = _status
                        ui.Spacer(height=6)
                        ui.Label("RIGHT ARM", height=22)
                        for _n in RIGHT_ARM + ["__SEP__"] + RIGHT_HAND:
                            if _n == "__SEP__":
                                ui.Spacer(height=8)
                                ui.Label("RIGHT HAND", height=22)
                                continue
                            _i = robot.joint_names.index(_n)
                            _lo = robot.data.joint_pos_limits.torch[0, _i, 0].item()
                            _hi = robot.data.joint_pos_limits.torch[0, _i, 1].item()
                            with ui.HStack(height=24, spacing=6):
                                ui.Label(_n.replace("right_", "").replace("_joint", ""), width=150)
                                _sl = ui.FloatSlider(min=_lo, max=_hi, step=0.005)
                                _sl.model.set_value(robot.data.joint_pos.torch[0, _i].item())
                                _sl.model.add_value_changed_fn(_make_cb(_i))
                                _sliders[_n] = _sl
            print("[pose_lab] slider window ready: 'Spatula Pose Lab'", flush=True)
        except Exception as _err:
            print(f"[pose_lab] slider window unavailable ({_err}); use the pose file", flush=True)

        dump(pose_file)
        print("\n" + "=" * 78)
        print(f"EDIT THIS FILE, SAVE, AND THE POSE UPDATES LIVE:\n    {pose_file}")
        print("Goal: fingertip z near 0 (tips on the tabletop) and the thumb's local-y")
        print("opposite in sign to index/middle (the straddle). Ctrl+C to exit.")
        print("=" * 78, flush=True)

        last_mtime = os.path.getmtime(pose_file)
        last_print = 0.0
        try:
            while True:
                try:
                    m = os.path.getmtime(pose_file)
                except OSError:
                    m = last_mtime
                if m != last_mtime:
                    last_mtime = m
                    try:
                        with open(pose_file) as fh:
                            vals = json.load(fh)
                    except Exception as err:
                        print(f"[pose_lab] bad JSON, ignoring: {err}", flush=True)
                        vals = None
                    if vals:
                        q = robot.data.joint_pos.torch[:1].clone()
                        lim = robot.data.joint_pos_limits.torch[0]
                        for n, v in vals.items():
                            if n in robot.joint_names:
                                i = robot.joint_names.index(n)
                                clamped = min(max(float(v), lim[i, 0].item()), lim[i, 1].item())
                                if abs(clamped - float(v)) > 1e-6:
                                    print(f"[pose_lab] {n} clamped {float(v):+.4f} -> {clamped:+.4f}", flush=True)
                                q[0, i] = clamped
                        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=eid)
                        robot.set_joint_position_target(q, env_ids=eid)
                        last_print = 0.0  # force a fresh readout

                # DO NOT write PD targets here. The initial pose set them once;
                # after that the drive targets belong to the user. Rewriting them
                # every frame is what made the GUI joint manipulators and the
                # Property-panel drive targets appear completely dead — the loop
                # simply overwrote whatever was typed or dragged, 60x a second.
                # FLUSH the target buffer. set_joint_position_target() only
                # stages the value; env.step() would call write_data_to_sim()
                # for us, but this loop drives env.sim.step() directly, so
                # without this the sliders (and any target write) do nothing.
                robot.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env_cfg.sim.dt)

                now = time.time()
                if now - last_print >= args_cli.print_every:
                    last_print = now
                    d, y = _handle_segment_geometry(env, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, fc, oc)
                    z = robot.data.body_pos_w.torch[:, fc.body_ids, 2] - env.scene.env_origins[:, 2:3]
                    ti = tip_names.index("right_hand_thumb_2_link")
                    ii = tip_names.index("right_hand_index_1_link")
                    print("\n--- FINGERTIPS ---", flush=True)
                    for k, n in enumerate(tip_names):
                        print(
                            f"    {n:28s} z {100 * (z[0, k].item() - LAB_TABLE_HEIGHT):+6.2f} cm | "
                            f"handle-dist {100 * d[0, k].item():5.2f} cm | local-y {100 * y[0, k].item():+6.2f} cm",
                            flush=True,
                        )
                    print(f"    straddle: {bool((y[0, ti] * y[0, ii] < 0).item())}", flush=True)
        except KeyboardInterrupt:
            if args_cli.save:
                final = robot.data.joint_pos.torch[0].detach().cpu().clone()
                torch.save(
                    {
                        "stages": [
                            {
                                "name": "pre_grasp_caged",
                                "joint_pos": final,
                                "joint_vel": torch.zeros_like(final),
                                "object_pos": (env.scene["spatula"].data.root_pos_w.torch[0] - env.scene.env_origins[0])
                                .detach()
                                .cpu()
                                .clone(),
                                "object_quat": env.scene["spatula"].data.root_quat_w.torch[0].detach().cpu().clone(),
                            }
                        ]
                    },
                    args_cli.save,
                )
                print(f"\n[pose_lab] saved pose -> {args_cli.save}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
