# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the fingers-down pre-grasp by collision-constrained IK.

Solves, against the vendor MuJoCo model: tool axis vertical (the descent
cannot clip the wall), pad-separation axis horizontal, pad midpoint over the
proven straddle offset, height chosen so the fingertips sit 3-12 mm below
the rim plane — then REJECTS any solution whose gripper vertices come
within 3 mm of the mug's wall material (cavity and exterior are free
space), searching small plan-view shifts for the maximum-clearance pose.

Paste the printed dict into GRASP_BANK_POSE and re-run
probe_bank_jacobian.py for the matching placement Jacobian.

USAGE (single line, CPU only)
  ~/Documents/code/icra2027/.venv/bin/python scripts/probes/probe_generate_pregrasp.py
"""

import mujoco
import numpy as np

XML = "/tmp/claude-1002/-home-mdigiorgio-Documents-code/fef98df8-95da-4aa4-a47c-133d6ad86ec5/scratchpad/trossen_mjc/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml"
SEED_POSE = np.array([0.042, 1.978, 1.586, -0.753, 0.000, -0.043])
MUG = np.array([-0.02, 0.0])
BOT, RIMZ, RIN, ROUT = 0.021, 0.1183, 0.035, 0.0388
CLEARANCE_FLOOR = 0.003
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
JOINTS = [f"follower_left_joint_{i}" for i in range(6)]
jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
qadr = np.array([m.jnt_qposadr[j] for j in jids])
vadr = np.array([m.jnt_dofadr[j] for j in jids])
lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
for n_, v in [("follower_left_left_carriage_joint", 0.021), ("follower_left_right_carriage_joint", 0.021)]:
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n_)]] = v
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "follower_left_ee_site")


def fk(q):
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    return d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()


def gpts(q):
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    out = {"tips": []}
    V = []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "follower_left_carriage" not in bn:
            continue
        R, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            w = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ R.T + t
        else:
            w = t + (CORNERS * m.geom_size[g]) @ R.T
        V.append(w)
        if "pad_lower" in gn:
            out[gn] = t.copy()
        if "tip" in gn:
            out["tips"].append(w[:, 2].min())
    out["verts"] = np.vstack(V)
    return out


def wall_clearance(verts):
    p = verts - np.array([MUG[0], MUG[1], 0.0])
    r = np.hypot(p[:, 0], p[:, 1])
    z = p[:, 2]
    band = (z > BOT + 0.006) & (z < RIMZ)
    c = np.where(r < RIN, RIN - r, np.where(r > ROUT, r - ROUT, -np.minimum(r - RIN, ROUT - r)))
    return np.where(band, c, np.inf).min()


p0, R0 = fk(SEED_POSE)
g0 = gpts(SEED_POSE)
padmid_w = (g0["follower_left_gripper_left_pad_lower"] + g0["follower_left_gripper_right_pad_lower"]) / 2
local_off = R0.T @ (padmid_w - p0)
Rt = np.column_stack([[0, 0, -1], [1, 0, 0], np.cross([0, 0, -1], [1, 0, 0])])
jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))


def solve(tgt):
    q = SEED_POSE.copy()
    ep = er = np.inf
    for _ in range(300):
        p, R = fk(q)
        st = tgt - Rt @ local_off
        e_p = st - p
        dR = Rt @ R.T
        e_r = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        ep, er = np.linalg.norm(e_p), np.linalg.norm(e_r)
        if ep < 1e-5 and er < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        J = np.vstack([jacp[:, vadr], jacr[:, vadr]])
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), np.concatenate([e_p, e_r])), lo, hi)
    return q, ep, er


base = padmid_w[:2] - MUG
best = None
for dx in np.arange(-0.006, 0.0065, 0.001):
    for dy in np.arange(-0.004, 0.0045, 0.002):
        q, ep, er = solve(np.array([MUG[0] + base[0] + dx, MUG[1] + base[1] + dy, 0.154]))
        if ep > 1e-3 or er > 5e-3:
            continue
        g = gpts(q)
        below = RIMZ - min(g["tips"])
        clr = wall_clearance(g["verts"])
        pads = [g["follower_left_gripper_left_pad_lower"], g["follower_left_gripper_right_pad_lower"]]
        rad = [float(np.hypot(p_[0] - MUG[0], p_[1] - MUG[1])) for p_ in pads]
        if not (min(rad) < RIN and max(rad) > ROUT and 0.003 <= below <= 0.012 and clr >= CLEARANCE_FLOOR):
            continue
        if best is None or clr > best[1]:
            best = (q, clr, rad, below)

assert best is not None, "no feasible fingers-down pose met all constraints"
q, clr, rad, below = best
print("GRASP_BANK_POSE = {")
for n_, v in zip(JOINTS, q):
    print(f'    "{n_}": {v:+.4f},')
print('    "follower_left_left_carriage_joint": 0.021,')
print('    "follower_left_right_carriage_joint": 0.021,')
print("}")
print(f"# fingers-down | tips {below*1000:.1f} mm below rim | wall clearance {clr*1000:.1f} mm"
      f" | straddle radials {[round(r*1000, 1) for r in rad]} mm")
