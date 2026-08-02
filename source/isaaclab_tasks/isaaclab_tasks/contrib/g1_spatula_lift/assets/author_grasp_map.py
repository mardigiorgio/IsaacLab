# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author ``grasp_map.pt`` for the G1 spatula-lift task (g1_dish_rack pattern).

Two modes, both kit-less on the Newton preset (the grasp-map event is removed
so resets are nominal):

* ``--search`` — batch-teleport a grid of right-arm poses + finger curls over
  64 envs, settle a few zero-action steps, and rank candidates by the
  privileged straddle geometry (thumb and index/middle tips near the handle
  centerline on OPPOSITE local-y sides, handle near the palm, spatula
  undisturbed). Prints the top candidates as ready-to-paste ``--joints`` args.
* default — teleport 1 env into the given ``--joints`` pose, settle, snapshot
  ONE ``pre_grasp_caged`` stage, save, then validate: re-reset, teleport,
  survive 60 zero-action steps with < 5 mm spatula displacement and the
  straddle intact (relative joint-position actions hold a teleported pose —
  the property that makes stage teleports sound).

The stage snapshot holds the full joint state; the object pose is recorded
for provenance but never written back on reset (see
``mdp.reset_from_grasp_map``). Touch/straddle gates use the privileged
handle-segment geometry from ``mdp.functions`` (no contact sensors).

Run::

    ./isaaclab.sh -p .../assets/author_grasp_map.py --search
    ./isaaclab.sh -p .../assets/author_grasp_map.py --joints "p,r,y,e,wr,wp,wy" --curl 0.55 --thumb-curl 0.35
"""

import argparse
import os

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--search", action="store_true", help="grid-search caged poses over 64 envs instead of authoring")
parser.add_argument(
    "--joints",
    default=None,
    help="authoring: 7 comma-separated right-arm targets [rad] (sh_pitch,sh_roll,sh_yaw,elbow,wr_roll,wr_pitch,wr_yaw)",
)
parser.add_argument("--curl", type=float, default=0.55, help="index/middle flexion fraction of the joint limit")
parser.add_argument("--thumb-curl", type=float, default=0.35, help="thumb flexion fraction of the joint limit")
parser.add_argument(
    "--sweep-curls", action="store_true", help="diagnostic: sweep curl fractions at --joints and print geometry"
)
parser.add_argument(
    "--view",
    default=None,
    help="diagnostic: ';'-separated 7-value arm poses — teleport each, print geometry, save a frame",
)
parser.add_argument("--out", default=None, help="output path (default: ../grasp_map.pt next to the env cfg)")
parser.add_argument("--frames-dir", default=None, help="save rendered PNG frames of key moments here (debug)")
add_launcher_args(parser)
args_cli = parser.parse_args()

import itertools

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
    FINGERTIP_BODY_NAMES,
    HANDLE_GRASP_OFFSET_B,
    HANDLE_SEGMENT_P0_B,
    HANDLE_SEGMENT_P1_B,
)
from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import (
    _handle_grasp_point_w,
    _handle_segment_geometry,
    _sensor_peak_force,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_physics_preset

TASK = "IsaacContrib-Lift-Spatula-G1-v0"

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CAGE_RADIUS = 0.11  # [m] per-digit origin-to-segment gate. Calibrated for the
# CLAW grip: the fingertip LINK ORIGINS sit at the knuckles, and with the
# pads seated on the handle top/faces the origins still read ~0.055-0.06
# (search asymptote with nonzero spatula disp = pads grazing). Same
# origin-offset reasoning as the thumb's 0.08. The vertical-curl grip
# threaded origins to ~0.03 but cannot reach the 83 cm tabletop.
CAGE_THUMB_RADIUS = 0.08  # [m] thumb-to-segment gate (knuckle origin; pad reaches much closer)
CAGE_PALM_RADIUS = 0.12  # [m] palm-to-segment gate: the handle must be near the palm
SETTLE_STEPS = 10  # search: zero-action steps after the teleport before scoring
CAGED_STABLE_STEPS = 30
VALIDATE_STEPS = 60
MAX_OBJECT_DISPLACEMENT = 0.005  # [m] validation gate (5 mm)
SEARCH_ENVS = 64

# search grid: wrist roll fixed at the measured vertical-curl 1.87, wrist
# pitch at the raised-unfold -0.30; yaw candidates rotate the thumb-finger
# opposition axis across the handle's width axis. Round-2 grid centered on
# the round-1 near-miss "-0.4,-0.2,0,0.5,...,-0.6" (palm over the handle,
# tips landing far-side — needs a -y shift and a curl split).
SEARCH_GRID = {
    # PRONATED vertical curl (frame-verified knuckles-up, fingers hooking
    # DOWN over the handle): wrist roll -1.87, pitch +0.30 — the +1.87 roll
    # is supinated (fingers curl UP; video-verified dead end). Keep the tips
    # bracketing the handle just above contact so the zero-action arm sag
    # slides them through air (a tip in contact drags the spatula past the
    # 5 mm validation gate).
    # Round-4 grid: CLAW grip (wrist roll 0.30 — palm down, fingers forward-
    # down) at the 0.85 gantry mount. The vertical-curl grip cannot reach the
    # 83 cm tabletop (palm would need to go below the table surface); the
    # claw's thumb measured 2 cm from the handle line in scripted descents.
    "sh_pitch": (-0.54, -0.50),
    "sh_roll": (-0.15,),
    "sh_yaw": (-0.10, 0.0),
    "elbow": (1.15, 1.20),
    "w_yaw": (-0.40,),
    "w_pitch": (-0.40, -0.35, -0.30),
    "curls": ((0.6, 0.4), (0.7, 0.4), (0.55, 0.35)),
}
SEARCH_WRIST_ROLL = 0.30


def main():  # noqa: C901
    num_envs = SEARCH_ENVS if args_cli.search else 1
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=num_envs)
    # Newton preset must be applied BEFORE launch_simulation (kitless scan) and
    # before any other mutation (apply_physics_preset discards later edits)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    # author from nominal resets even if a previous map exists — and without
    # the reset randomization (stage snapshots must be deterministic)
    env_cfg.events.reset_grasp_map = None
    env_cfg.events.randomize_right_arm = None
    env_cfg.events.randomize_spatula_pose = None
    # the scripted teleport/settle/validate sequence outlives the 3 s training
    # episode — stretch it so no timeout reset fires mid-authoring
    env_cfg.episode_length_s = 60.0
    if args_cli.view:
        # close-up camera on the hand/handle for grip inspection frames
        env_cfg.viewer.eye = (0.85, -0.75, 1.05)
        env_cfg.viewer.lookat = (0.2, -0.1, 0.75)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.frames_dir else None)
        uenv = env.unwrapped

        robot = uenv.scene["robot"]
        device = uenv.device
        spatula = uenv.scene["spatula"]
        spatula_cfg = SceneEntityCfg("spatula")
        tips_cfg = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        tips_cfg.resolve(uenv.scene)
        palm_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
        palm_cfg.resolve(uenv.scene)
        # column order follows body_ids (ASSET order) — cfg.body_names keeps
        # the user-given order in this fork, so map through the articulation
        col_names = [uenv.scene["robot"].body_names[i] for i in tips_cfg.body_ids]
        ti = col_names.index("right_hand_thumb_2_link")
        ii = col_names.index("right_hand_index_1_link")
        mi = col_names.index("right_hand_middle_1_link")

        action_dim = uenv.action_space.shape[1]
        arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
        finger_ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("right_hand_")]
        thumb_ids = [i for i in finger_ids if "thumb" in robot.joint_names[i]]

        env.reset()
        spawn_pos = spatula.data.root_pos_w.torch.clone()

        limits = robot.data.joint_pos_limits.torch[0]
        # flexion target = the larger-|limit| bound scaled by the curl fraction
        # (sign-correct for the TriHand thumb, same convention as the rewards)
        flex_bound = torch.where(limits.abs().argmax(dim=-1).bool(), limits[:, 1], limits[:, 0])

        def build_joint_pos(arm7, finger_frac, thumb_frac, rows):
            jp = robot.data.default_joint_pos.torch[:rows].clone()
            for jid, val in zip(arm_ids, arm7):
                jp[:, jid] = val
            for jid in finger_ids:
                frac = thumb_frac if jid in thumb_ids else finger_frac
                jp[:, jid] = flex_bound[jid] * frac
            return jp

        def step_zero(n):
            zero = torch.zeros((uenv.num_envs, action_dim), device=device)
            with torch.inference_mode():
                for _ in range(n):
                    env.step(zero)

        def cage_metrics():
            """Vectorized straddle metrics over all envs."""
            tip_d, tip_y = _handle_segment_geometry(
                uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips_cfg, spatula_cfg
            )
            palm_d, _ = _handle_segment_geometry(uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, palm_cfg, spatula_cfg)
            palm_d = palm_d[:, 0]
            # near-grasp gate (CLAW grip): the handle sits inside the open
            # aperture with every digit origin within reach and the palm over
            # it. Origin-SIGN opposition is unsatisfiable in this grip (the
            # tip frames sit at the knuckles, which ride the far side of the
            # palm even with the thumb pad on the near face), and curling
            # alone cannot create contact (the tip arc bottoms out 1-2 cm
            # above the handle — measured). The stage's job is to start the
            # policy ONE ACTION from contact income; the descend-and-squeeze
            # is the move it must learn.
            near = tip_d < CAGE_RADIUS
            caged = near.all(dim=1) & (palm_d < CAGE_PALM_RADIUS)
            gp = _handle_grasp_point_w(uenv, HANDLE_GRASP_OFFSET_B, spatula_cfg)
            palm_pos = robot.data.body_pos_w.torch[:, palm_cfg.body_ids, :][:, 0]
            palm_grasp = torch.linalg.vector_norm(palm_pos - gp, dim=-1)
            opp_d = torch.minimum(
                torch.where(tip_y[:, ti] * tip_y[:, ii] < 0, tip_d[:, ii], torch.full_like(palm_d, 9.9)),
                torch.where(tip_y[:, ti] * tip_y[:, mi] < 0, tip_d[:, mi], torch.full_like(palm_d, 9.9)),
            )
            disp = torch.linalg.vector_norm(spatula.data.root_pos_w.torch - spawn_pos, dim=-1)
            return caged, tip_d, tip_y, palm_d, palm_grasp, opp_d, disp

        def print_env0_state(label):
            caged, tip_d, tip_y, palm_d, palm_grasp, _, disp = cage_metrics()
            origin = uenv.scene.env_origins[0]
            for name in FINGERTIP_BODY_NAMES:
                c = col_names.index(name)
                pos = (robot.data.body_pos_w.torch[0, tips_cfg.body_ids[c]] - origin).tolist()
                print(
                    f"[grasp_map] {label} {name}: seg-dist {tip_d[0, c].item():.4f} m,"
                    f" local-y {tip_y[0, c].item():+.4f} m, env pos {[round(v, 4) for v in pos]}"
                )
            palm_pos = (robot.data.body_pos_w.torch[0, palm_cfg.body_ids[0]] - origin).tolist()
            from isaaclab.utils.math import quat_apply

            root_pos = spatula.data.root_pos_w.torch[0]
            root_quat = spatula.data.root_quat_w.torch[0].unsqueeze(0)
            for tag, pt in (("seg p0", HANDLE_SEGMENT_P0_B), ("seg p1", HANDLE_SEGMENT_P1_B)):
                w = (root_pos + quat_apply(root_quat, torch.tensor([pt], device=device))[0] - origin).tolist()
                print(f"[grasp_map] {label} {tag}: env pos {[round(v, 4) for v in w]}")
            print(
                f"[grasp_map] {label} palm: seg-dist {palm_d[0].item():.4f} m,"
                f" palm->grasp {palm_grasp[0].item():.4f} m,"
                f" env pos {[round(v, 4) for v in palm_pos]},"
                f" spatula disp {disp[0].item():.4f} m, caged={bool(caged[0].item())}"
            )

        def save_frame(tag):
            if not args_cli.frames_dir:
                return
            frame = env.render()
            if frame is None:
                return
            os.makedirs(args_cli.frames_dir, exist_ok=True)
            from PIL import Image

            Image.fromarray(frame).save(os.path.join(args_cli.frames_dir, f"{tag}.png"))
            print(f"[grasp_map] frame saved: {tag}.png")

        if args_cli.search:
            g = SEARCH_GRID
            cands = [
                (sp, sr, sy, el, SEARCH_WRIST_ROLL, wp, wy, cf, tf)
                for sp, sr, sy, el, wy, wp, (cf, tf) in itertools.product(
                    g["sh_pitch"], g["sh_roll"], g["sh_yaw"], g["elbow"], g["w_yaw"], g["w_pitch"], g["curls"]
                )
            ]
            print(f"[grasp_map] searching {len(cands)} candidates over {SEARCH_ENVS} envs")
            results = []
            for start in range(0, len(cands), SEARCH_ENVS):
                batch = cands[start : start + SEARCH_ENVS]
                env.reset()
                jp = build_joint_pos((0,) * 7, 0.0, 0.0, uenv.num_envs)
                for row, cand in enumerate(batch):
                    for jid, val in zip(arm_ids, cand[:7]):
                        jp[row, jid] = val
                    for jid in finger_ids:
                        frac = cand[8] if jid in thumb_ids else cand[7]
                        jp[row, jid] = flex_bound[jid] * frac
                with torch.inference_mode():
                    robot.write_joint_state_to_sim(
                        jp, torch.zeros_like(jp), env_ids=torch.arange(uenv.num_envs, device=device)
                    )
                step_zero(SETTLE_STEPS)
                caged, tip_d, tip_y, palm_d, palm_grasp, opp_d, disp = cage_metrics()
                for row, cand in enumerate(batch):
                    ok = disp[row].item() < MAX_OBJECT_DISPLACEMENT
                    f_col = ii if tip_d[row, ii] < tip_d[row, mi] else mi
                    opposed = (tip_y[row, ti] * tip_y[row, f_col] < 0).item()
                    score = (
                        tip_d[row, f_col].item()
                        + tip_d[row, ti].item()
                        + (0.0 if opposed else 0.5)
                        + max(0.0, palm_grasp[row].item() - 0.10)
                        + (0.0 if ok else 10.0)
                    )
                    results.append(
                        (
                            bool(caged[row].item()) and ok,
                            score,
                            cand,
                            tip_d[row, ti].item(),
                            tip_y[row, ti].item(),
                            tip_d[row, f_col].item(),
                            tip_y[row, f_col].item(),
                            palm_grasp[row].item(),
                            disp[row].item(),
                        )
                    )
                done = min(start + SEARCH_ENVS, len(cands))
                n_caged = sum(1 for r in results if r[0])
                print(f"[grasp_map] {done}/{len(cands)} searched, {n_caged} caged so far")
            results.sort(key=lambda r: (not r[0], r[1]))
            print("[grasp_map] top candidates (caged, score, thumb d/y, best-finger d/y, palm->grasp, disp):")
            for ok, score, cand, td, ty, fd, fy, pg, dp in results[:15]:
                arm = ",".join(f"{v:g}" for v in cand[:7])
                print(
                    f"[grasp_map]   caged={ok} score={score:.3f} index {td:.3f}/{ty:+.3f}"
                    f" middle {fd:.3f}/{fy:+.3f} palm->grasp={pg:.3f} disp={dp:.4f}"
                    f'  --joints "{arm}" --curl {cand[7]:g} --thumb-curl {cand[8]:g}'
                )
            env.close()
            return

        if args_cli.view:
            arm_names = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
            lims = robot.data.joint_pos_limits.torch[0]
            print(
                "[grasp_map] right-arm limits:"
                f" { {n: [round(v, 3) for v in lims[j].tolist()] for n, j in zip(RIGHT_ARM_JOINTS, arm_names)} }"
            )
            for k, cand in enumerate(args_cli.view.split(";")):
                arm = [float(v) for v in cand.split(",")]
                env.reset()
                jp = build_joint_pos(arm, args_cli.curl, args_cli.thumb_curl, 1)
                with torch.inference_mode():
                    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=torch.tensor([0], device=device))
                step_zero(SETTLE_STEPS)
                print(f"[grasp_map] === view {k}: {arm}")
                print_env0_state(f"[view{k}]")
                from isaaclab.utils.math import matrix_from_quat

                palm_quat = robot.data.body_quat_w.torch[0, palm_cfg.body_ids[0]].unsqueeze(0)
                R = matrix_from_quat(palm_quat)[0]
                for ax, name in enumerate(("local-x", "local-y", "local-z")):
                    print(f"[grasp_map] [view{k}] palm {name} in world: {[round(v, 3) for v in R[:, ax].tolist()]}")
                save_frame(f"view_{k}")
            env.close()
            return

        # --- authoring: teleport into the given pose, settle, snapshot, validate ---
        if args_cli.joints is None:
            raise SystemExit("authoring needs --joints (run --search first and paste a printed candidate)")
        arm7 = [float(v) for v in args_cli.joints.split(",")]
        if len(arm7) != 7:
            raise SystemExit("--joints needs exactly 7 comma-separated values")

        if args_cli.sweep_curls:
            hand_names = [robot.joint_names[i] for i in finger_ids]
            print(f"[grasp_map] hand joints: {hand_names}")
            print(f"[grasp_map] hand limits: {[[round(v, 3) for v in limits[i].tolist()] for i in finger_ids]}")
            t0 = robot.joint_names.index("right_hand_thumb_0_joint")
            t1 = robot.joint_names.index("right_hand_thumb_1_joint")
            t2 = robot.joint_names.index("right_hand_thumb_2_joint")
            for th0 in (-1.0, -0.5, 0.0, 0.5, 1.0):
                for th1 in (-0.9, -0.3, 0.3, 0.7):
                    env.reset()
                    jp = build_joint_pos(arm7, 0.55, 0.0, 1)
                    jp[0, t0], jp[0, t1], jp[0, t2] = th0, th1, -0.5
                    with torch.inference_mode():
                        robot.write_joint_state_to_sim(
                            jp, torch.zeros_like(jp), env_ids=torch.tensor([0], device=device)
                        )
                    step_zero(SETTLE_STEPS)
                    print(f"[grasp_map] === thumb {th0}/{th1}/-0.5:")
                    print_env0_state(f"[t {th0}/{th1}]")
            env.close()
            return

        def teleport():
            jp = build_joint_pos(arm7, args_cli.curl, args_cli.thumb_curl, 1)
            with torch.inference_mode():
                robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=torch.tensor([0], device=device))

        save_frame("00_ready")
        teleport()
        step_zero(CAGED_STABLE_STEPS)
        print_env0_state("[caged]")
        for sname in ("thumb_handle_contact", "index_handle_contact", "middle_handle_contact"):
            print(f"[grasp_map] [caged] {sname}: {_sensor_peak_force(uenv, sname)[0].item():.4f} N")
        caged, *_ = cage_metrics()
        disp0 = torch.linalg.vector_norm(spatula.data.root_pos_w.torch[0] - spawn_pos[0]).item()
        if not bool(caged[0].item()) or disp0 >= MAX_OBJECT_DISPLACEMENT:
            save_frame("01_cage_failed")
            raise RuntimeError("pose does not hold a straddle over the settle — pick another --search candidate")

        origin = uenv.scene.env_origins[0]
        caged_stage = {
            "name": "pre_grasp_caged",
            "joint_pos": robot.data.joint_pos.torch[0].detach().cpu().clone(),
            "joint_vel": torch.zeros_like(robot.data.joint_vel.torch[0]).cpu(),
            "object_pos": (spatula.data.root_pos_w.torch[0] - origin).detach().cpu().clone(),
            "object_quat": spatula.data.root_quat_w.torch[0].detach().cpu().clone(),
        }
        arm_pose = {
            name: round(robot.data.joint_pos.torch[0, robot.joint_names.index(name)].item(), 3)
            for name in RIGHT_ARM_JOINTS
        }
        print(f"[grasp_map] caged right-arm joints (for DEFAULT_ARM_JOINT_POS baking): {arm_pose}")
        save_frame("01_caged")

        out_path = args_cli.out or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grasp_map.pt"
        )
        torch.save({"stages": [caged_stage]}, out_path)
        print(f"[grasp_map] wrote {out_path} with stages: ['pre_grasp_caged']")

        # --- validation: fresh reset, teleport into the stage, hold 60 zero-action steps ---
        env.reset()
        with torch.inference_mode():
            robot.write_joint_state_to_sim(
                caged_stage["joint_pos"].to(device).unsqueeze(0),
                caged_stage["joint_vel"].to(device).unsqueeze(0),
                env_ids=torch.tensor([0], device=device),
            )
        obj_before = spatula.data.root_pos_w.torch[0].clone()
        step_zero(VALIDATE_STEPS)
        disp = torch.linalg.vector_norm(spatula.data.root_pos_w.torch[0] - obj_before).item()
        caged, *_ = cage_metrics()
        still_caged = bool(caged[0].item())
        print_env0_state("[validate]")
        ok = disp < MAX_OBJECT_DISPLACEMENT and still_caged
        print(
            f"[grasp_map] validate pre_grasp_caged: displacement {disp:.4f} m,"
            f" caged={still_caged} -> {'OK' if ok else 'FAILED'}"
        )
        save_frame("02_validated" if ok else "02_validate_failed")
        if not ok:
            raise RuntimeError("stage pre_grasp_caged failed validation — re-tune and re-author")

        env.close()


if __name__ == "__main__":
    main()
