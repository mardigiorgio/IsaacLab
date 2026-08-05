# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic straddle search for the G1 spatula claw grasp.

Answers ONE question: does an arm + hand pose exist that puts the thumb pad on
one side of the handle and the index/middle pads on the other, with all of them
down at tabletop height? That is the cage the real hardware makes (photographed
on a pen: forearm parallel to the table, palm down, digits splayed wide, object
in the gap, then rake inward).

Why kinematic: the question is about GEOMETRY, so this teleports each candidate
pose and measures it. No scripted servo, hence none of the instability that
comes from driving a ``RelativeJointPositionActionCfg`` term at its action
limit (an action pinned at +-1 walks the joint target 0.2 rad EVERY control
step, which accelerates the arm until it hits something). Dynamic close-and-lift
testing belongs downstream, on the handful of poses that already straddle.

Every thumb joint is swept INDEPENDENTLY over its own limit range.
``author_grasp_map`` drives all three from one scalar (``flex_bound[jid] *
frac``), a 1-D ray through 3-D thumb space anchored at zero, which cannot
express an opposition pose — so its zero-straddle results are an artifact of
the parameterization, not evidence that no claw grasp exists.

Run::

    ./isaaclab.sh -p .../assets/straddle_search.py --headless
    ./isaaclab.sh -p .../assets/straddle_search.py --headless --out /tmp/claw.pt
"""

import argparse

from isaaclab.app import add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096, help="candidates per pass; does not change results")
parser.add_argument("--top", type=int, default=25, help="how many ranked candidates to print")
parser.add_argument("--out", default=None, help="write the winning pose as a grasp-map stage here")
parser.add_argument("--settle-steps", type=int, default=6, help="zero-action steps after the teleport")
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

SEARCH_GRID = {
    # Refined around the only region that both cages and traps with the thumb
    # opposing BOTH fingers: roll -1.2, pitch 0.6, yaw -0.3, sh_pitch -0.65,
    # elbow 1.1. The coarse 48k-candidate sweep produced exactly 2 hits there,
    # so this pass is about finding the MIDDLE of that region, not its edge.
    "sh_pitch": (-0.72, -0.65, -0.58),
    "elbow": (1.00, 1.10, 1.20),
    # extended past -1.05: the previous pass put 8 of 13 winners on that value,
    # which was its grid BOUNDARY, so the optimum was outside the swept range
    "w_roll": (-1.20, -1.05, -0.90, -0.75),
    "w_pitch": (0.50, 0.60, 0.70),
    "w_yaw": (-0.40, -0.30, -0.20),
    # likewise extended: thumb_0 0.65 was the old boundary
    "thumb_0": (0.50, 0.65, 0.80),
    "thumb_close": (0.4, 0.6, 0.8),
    "finger_close": (0.55, 0.70, 0.85),
}
"""8748 candidates, re-centred so no axis optimum sits on a boundary."""

SH_ROLL = -0.15
SH_YAW = -0.10
THUMB_OPEN_FRAC = 0.0
"""Thumb flexion at the open/approach pose, as a fraction of each joint's range."""

TRAP_RADIUS = 0.045
"""Tip-to-centerline distance [m] required of BOTH sides in the CLOSED pose.
Tighter than STRADDLE_RADIUS: caging at arm's length is not a grasp."""

TIP_BAND = 0.045
"""A tip within this height [m] of the tabletop counts as 'down at the table'."""

STRADDLE_RADIUS = 0.06
"""Tip-to-handle-centerline distance [m] that counts as caging it. Generous:
these are LINK ORIGINS at the knuckles, not the pads (see PALM_GRASP_RADIUS)."""

MAX_SPATULA_DISPLACEMENT = 0.010
"""Approach tolerance [m]: a teleport that shoves the spatula more than this
did not cage it — on approach the object must still be where it started."""

MAX_CLOSE_DISPLACEMENT = 0.050
"""Closing tolerance [m]. Deliberately generous: RAKING THE HANDLE IN IS THE
MECHANISM, so the object is supposed to move while the digits close. This bound
only rejects poses that fling it. The straddle geometry is computed in the
spatula's own body frame, so it already scores against wherever it ended up."""


def build_candidates():
    """Enumerate the grid as tuples in ``SEARCH_GRID`` key order."""
    g = SEARCH_GRID
    return list(
        itertools.product(
            g["sh_pitch"],
            g["elbow"],
            g["w_roll"],
            g["w_pitch"],
            g["w_yaw"],
            g["thumb_0"],
            g["thumb_close"],
            g["finger_close"],
        )
    )


def main():
    cands = build_candidates()
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg = apply_physics_preset(env_cfg, TASK, "newton_mjwarp")
    # Every candidate is teleported onto the same scene, so the scene must not
    # vary: a pregrasp reset drops the claw around the spatula and can nudge it
    # before the teleport, which the displacement gates then score.
    # `reset_pregrasp` is the only randomizing term EventCfg still has --
    # `reset_grasp_map` / `randomize_*` were removed from the task, and assigning
    # them here only created new attributes nobody read.
    env_cfg.events.reset_pregrasp = None
    env_cfg.terminations.blade_contact = None
    env_cfg.terminations.spatula_dropped = None
    env_cfg.terminations.robot_exploded = None
    env_cfg.rewards.blade_penalty = None
    env_cfg.episode_length_s = 120.0

    print(f"[straddle] {len(cands)} candidates, {args_cli.num_envs} envs/pass")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        spatula = uenv.scene["spatula"]
        device = uenv.device
        env.reset()

        joint_names = list(robot.joint_names)
        arm_ids = [joint_names.index(n) for n in ARM_JOINTS]
        limits = robot.data.joint_pos_limits.torch[0]

        def jid(name):
            return joint_names.index(name)

        t0, t1, t2 = jid("right_hand_thumb_0_joint"), jid("right_hand_thumb_1_joint"), jid("right_hand_thumb_2_joint")
        finger_ids = [jid(n) for n in joint_names if n.startswith("right_hand_") and ("index" in n or "middle" in n)]

        def interp(j, frac):
            lo, hi = limits[j, 0].item(), limits[j, 1].item()
            return lo + frac * (hi - lo)

        print("[straddle] thumb joint ranges (swept independently):")
        for name, j in (("thumb_0", t0), ("thumb_1", t1), ("thumb_2", t2)):
            print(f"[straddle]   {name}: [{limits[j, 0].item():+.3f}, {limits[j, 1].item():+.3f}]")

        tips_cfg = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        tips_cfg.resolve(uenv.scene)
        col = [robot.body_names[i] for i in tips_cfg.body_ids]
        ti = col.index("right_hand_thumb_2_link")
        ii = col.index("right_hand_index_1_link")
        mi = col.index("right_hand_middle_1_link")
        spat_cfg = SceneEntityCfg("spatula")

        action_dim = uenv.action_space.shape[1]
        zero = torch.zeros((uenv.num_envs, action_dim), device=device)
        origin_z = uenv.scene.env_origins[:, 2]
        spawn_env = (spatula.data.root_pos_w.torch - uenv.scene.env_origins)[0].clone()

        rows = []
        for start in range(0, len(cands), args_cli.num_envs):
            batch = cands[start : start + args_cli.num_envs]
            n = len(batch)
            env.reset()

            def build(closed: bool):
                """Joint targets for the batch, open (approach) or closed (grasp)."""
                jp = robot.data.default_joint_pos.torch.clone()
                for row, (sp, el, wr, wp, wy, th0, tc, fcl) in enumerate(batch):
                    for j, v in zip(arm_ids, (sp, SH_ROLL, SH_YAW, el, wr, wp, wy)):
                        jp[row, j] = v
                    # thumb_0 is opposition: held across both poses
                    jp[row, t0] = interp(t0, th0)
                    tf = tc if closed else THUMB_OPEN_FRAC
                    jp[row, t1] = interp(t1, tf)
                    jp[row, t2] = interp(t2, tf)
                    for j in finger_ids:
                        jp[row, j] = interp(j, fcl if closed else 0.0)
                return jp

            def measure(jp):
                """Teleport to jp, settle, and return (dist, y, tip_z, spat_disp)."""
                with torch.inference_mode():
                    robot.write_joint_state_to_sim(
                        jp, torch.zeros_like(jp), env_ids=torch.arange(uenv.num_envs, device=device)
                    )
                    for _ in range(args_cli.settle_steps):
                        env.step(zero)
                    d, yy = _handle_segment_geometry(uenv, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, tips_cfg, spat_cfg)
                    tz = robot.data.body_pos_w.torch[:, tips_cfg.body_ids, 2] - origin_z.unsqueeze(1)
                    sd = torch.linalg.vector_norm(
                        spatula.data.root_pos_w.torch - uenv.scene.env_origins - spawn_env, dim=-1
                    )
                return d, yy, tz, sd

            jp_open = build(closed=False)
            d_o, y_o, z_o, sd_o = measure(jp_open)
            env.reset()
            jp_cl = build(closed=True)
            d_c, y_c, z_c, sd_c = measure(jp_cl)

            def straddles(d, y, radius):
                """Thumb on ONE side, index AND middle BOTH on the other.

                The hardware grasp uses the two fingers as a single pad group
                with the thumb opposing them (user, replicating it on a pen).
                An earlier version accepted thumb-opposite-EITHER-finger, which
                admits the handle threading BETWEEN index and middle — a
                different grasp that the real hand does not make.
                """
                opp = (y[:, ti] * y[:, ii] < 0) & (y[:, ti] * y[:, mi] < 0)
                return opp & (d[:, ti] < radius) & (d[:, ii] < radius) & (d[:, mi] < radius)

            # OPEN: digits astride the handle at tabletop height, object untouched
            cage = (
                straddles(d_o, y_o, STRADDLE_RADIUS)
                & (z_o > LAB_TABLE_HEIGHT - 0.010).all(dim=1)
                & (z_o < LAB_TABLE_HEIGHT + TIP_BAND).all(dim=1)
                & (sd_o < MAX_SPATULA_DISPLACEMENT)
            )
            # CLOSED: still astride, and now CLOSE — the handle is trapped between
            # the pads rather than swept past. This is the criterion the first
            # version of this search was missing.
            # closing must TRAP the handle, not shove it: a pose that displaces
            # the spatula while closing has pushed it away, not grasped it. The
            # first version checked displacement on the OPEN pose only.
            trap = straddles(d_c, y_c, TRAP_RADIUS) & (d_c[:, ti] < TRAP_RADIUS) & (sd_c < MAX_CLOSE_DISPLACEMENT)
            finite = (
                torch.isfinite(d_o).all(dim=1)
                & torch.isfinite(d_c).all(dim=1)
                & torch.isfinite(z_o).all(dim=1)
                & torch.isfinite(sd_o)
                & torch.isfinite(sd_c)
            )
            ok = cage & trap & finite

            # margin = the least-decisive digit's offset from the centerline.
            # A pose whose middle finger sits 2 mm off the line technically
            # straddles but has no tolerance; rank that below a clean one.
            margin = torch.minimum(y_c[:, ti].abs(), torch.minimum(y_c[:, ii].abs(), y_c[:, mi].abs()))
            score = d_c[:, ti] + d_c[:, ii] + d_c[:, mi] - 3.0 * margin
            for row, cand in enumerate(batch):
                rows.append(
                    {
                        "cand": cand,
                        "ok": bool(ok[row].item()),
                        "cage": bool(cage[row].item()),
                        "trap": bool(trap[row].item()),
                        "straddle": bool(cage[row].item()),
                        "score": score[row].item(),
                        "thumb_d": d_c[row, ti].item(),
                        "thumb_y": y_c[row, ti].item(),
                        "idx_d": d_c[row, ii].item(),
                        "idx_y": y_c[row, ii].item(),
                        "mid_d": d_c[row, mi].item(),
                        "mid_y": y_c[row, mi].item(),
                        "tip_z_lo": z_o[row].min().item(),
                        "tip_z_hi": z_o[row].max().item(),
                        "spat_disp": sd_o[row].item(),
                        "finite": bool(finite[row].item()),
                        "margin": margin[row].item(),
                        "rake": sd_c[row].item(),
                        "joint_pos": jp_open[row].detach().cpu().clone(),
                        "joint_pos_closed": jp_cl[row].detach().cpu().clone(),
                    }
                )
            done = min(start + n, len(cands))
            print(
                f"[straddle] {done}/{len(cands)} searched,"
                f" {sum(1 for r in rows if r['cage'])} cage / {sum(1 for r in rows if r['ok'])} cage+trap"
            )

        report(rows, spatula, uenv)
        env.close()


def report(rows, spatula, uenv):
    """Rank, print, and snapshot the winner."""
    rows.sort(key=lambda r: (not r["ok"], not r["straddle"], r["score"]))
    n_cage = sum(1 for r in rows if r["cage"])
    n_trap = sum(1 for r in rows if r["trap"])
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n[straddle] {n_cage} cage when OPEN; {n_trap} trap when CLOSED; {n_ok} do BOTH")
    print("[straddle] (metrics below are the CLOSED pose — the one that has to hold the handle)")
    print(f"[straddle] top {args_cli.top} (cage, straddle, thumb d/y, index d/y, middle d/y, tip z lo..hi):")
    for r in rows[: args_cli.top]:
        sp, el, wr, wp, wy, th0, tc, fcl = r["cand"]
        print(
            f"[straddle]   cage={r['ok']:1} str={r['straddle']:1}"
            f" thumb {r['thumb_d']:.3f}/{r['thumb_y']:+.3f}"
            f" idx {r['idx_d']:.3f}/{r['idx_y']:+.3f}"
            f" mid {r['mid_d']:.3f}/{r['mid_y']:+.3f}"
            f" z {r['tip_z_lo']:.3f}..{r['tip_z_hi']:.3f} margin {r['margin']:.4f}"
            f" rake {r['rake']:.4f}"
            f"  sp={sp:g} el={el:g} roll={wr:g} pitch={wp:g} yaw={wy:g}"
            f" th0={th0:g} tclose={tc:g} fclose={fcl:g}"
        )

    # diagnostics that tell you how to re-center the grid when nothing cages.
    # Computed over FINITE rows only: a diverged env reports NaN, which would
    # otherwise turn every percentile into NaN and hide the real distribution.
    good = [r for r in rows if r["finite"]]
    n_bad = len(rows) - len(good)
    print(f"[straddle] {n_bad}/{len(rows)} candidates diverged (non-finite geometry); excluded from stats")
    if good:
        zs = torch.tensor([r["tip_z_lo"] for r in good])
        print(
            f"[straddle] lowest-tip z over finite candidates [m]: min {zs.min():.3f}"
            f" p50 {zs.quantile(0.5):.3f} max {zs.max():.3f} (tabletop {LAB_TABLE_HEIGHT})"
        )
        td = torch.tensor([r["thumb_d"] for r in good])
        print(f"[straddle] thumb-to-centerline [m]: min {td.min():.3f} p50 {td.quantile(0.5):.3f}")
        sd = torch.tensor([r["spat_disp"] for r in good])
        print(f"[straddle] spatula displacement [m]: p50 {sd.quantile(0.5):.4f} max {sd.max():.4f}")
        for label, key in (("cage+trap", "ok"), ("cage", "cage")):
            hist = {}
            for r in good:
                if r[key]:
                    hist[r["cand"][2]] = hist.get(r["cand"][2], 0) + 1
            print(f"[straddle] {label} per wrist_roll: {dict(sorted(hist.items()))}")
        winners = [r for r in good if r["ok"]]
        if winners:
            for idx, name in ((5, "thumb_0"), (6, "thumb_close"), (7, "finger_close")):
                hist = {}
                for r in winners:
                    hist[r["cand"][idx]] = hist.get(r["cand"][idx], 0) + 1
                print(f"[straddle] cage+trap per {name}: {dict(sorted(hist.items()))}")
    n_opp = sum(1 for r in rows if r["thumb_y"] * r["idx_y"] < 0 or r["thumb_y"] * r["mid_y"] < 0)
    print(f"[straddle] poses with thumb OPPOSITE a finger in local-y (any distance): {n_opp}/{len(rows)}")

    if not n_ok:
        print("[straddle] NO POSE BOTH CAGES AND TRAPS IN THIS GRID — re-center using the diagnostics above")
        raise SystemExit(1)

    if args_cli.out:
        origin = uenv.scene.env_origins[0]
        win = rows[0]
        torch.save(
            {
                "stages": [
                    {
                        "name": "claw_cage",
                        "joint_pos": win["joint_pos"],
                        "joint_vel": torch.zeros_like(win["joint_pos"]),
                        "joint_pos_closed": win["joint_pos_closed"],
                        "object_pos": (spatula.data.root_pos_w.torch[0] - origin).detach().cpu().clone(),
                        "object_quat": spatula.data.root_quat_w.torch[0].detach().cpu().clone(),
                    }
                ]
            },
            args_cli.out,
        )
        print(f"[straddle] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
