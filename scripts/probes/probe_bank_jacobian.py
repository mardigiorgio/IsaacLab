# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive BANK_POSE_XY_JACOBIAN: joint response that translates the banked
pre-grasp with a planar mug shift.

The grasp-bank pose is fixed, so the arm Jacobian there is a constant. The
damped pseudo-inverse of its position rows, restricted to (x, y), gives the
6-vector-per-axis joint update that translates the TCP by a mug placement
delta while (to first order) preserving orientation. Exact against the
vendor MuJoCo model; linearization error over a 1 cm shift is sub-mm.

Run whenever GRASP_BANK_POSE changes and paste the printed matrix into
trossen_mug_lift_env_cfg.BANK_POSE_XY_JACOBIAN.

USAGE (single line, CPU only)
  ~/Documents/code/icra2027/.venv/bin/python scripts/probes/probe_bank_jacobian.py
"""

import mujoco
import numpy as np

XML = "/tmp/claude-1002/-home-mdigiorgio-Documents-code/fef98df8-95da-4aa4-a47c-133d6ad86ec5/scratchpad/trossen_mjc/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml"
POSE = [0.042, 1.978, 1.586, -0.753, 0.000, -0.043]

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"follower_left_joint_{i}") for i in range(6)]
qadr = [m.jnt_qposadr[j] for j in jids]
vadr = [m.jnt_dofadr[j] for j in jids]
for a, v in zip(qadr, POSE):
    d.qpos[a] = v
mujoco.mj_forward(m, d)
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "follower_left_ee_site")
jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))
mujoco.mj_jacSite(m, d, jacp, jacr, site)
J = np.vstack([jacp[:, vadr], jacr[:, vadr]])  # 6x6: [pos; rot] rows
# damped least squares: translate in (x, y), hold z and orientation
target = np.zeros((6, 2))
target[0, 0] = 1.0
target[1, 1] = 1.0
M = np.linalg.solve(J.T @ J + 1e-8 * np.eye(6), J.T @ target)
print("BANK_POSE_XY_JACOBIAN = [")
for row in M:
    print(f"    [{row[0]:+.6f}, {row[1]:+.6f}],")
print("]")
# verify: FK at q + M @ [1cm, 0] and [0, 1cm]
for k, delta in enumerate([np.array([0.01, 0.0]), np.array([0.0, 0.01])]):
    q = np.array(POSE) + M @ delta
    for a, v in zip(qadr, q):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    p = d.site_xpos[site].copy()
    for a, v in zip(qadr, POSE):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    p0 = d.site_xpos[site].copy()
    moved = p - p0
    want = np.array([delta[0], delta[1], 0.0])
    print(f"axis {k}: moved {np.round(moved*1000,2)} mm, wanted {np.round(want*1000,2)} mm, err {np.linalg.norm(moved-want)*1000:.3f} mm")
