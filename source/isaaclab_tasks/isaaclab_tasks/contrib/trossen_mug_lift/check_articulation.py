# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation/actuation verification for IsaacContrib-Lift-Spatula-Trossen-v0.

Checks, under the (now default) Newton backend and optionally the adaptive solver:
  joints    all expected joints enumerate (both carriages included)
  arm       a held position action moves the commanded arm joint toward its target
  ee        the ee_frame sensor tracks the motion
  gripper   open/close drives the LEFT carriage to ~0.044 / ~0.0
  mimic     the RIGHT carriage mirrors the left (physxMimicJoint under Newton -- the
            known hazard: a PhysX-specific schema Newton may ignore)
Writes verdict JSON to $CHECK_OUT.
"""

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"
ADAPTIVE = os.environ.get("CHECK_ADAPTIVE") == "1"
res: dict = {"adaptive": ADAPTIVE}

try:
    env_cfg = parse_env_cfg(TASK, num_envs=4)
    if ADAPTIVE:
        env_cfg.sim.physics.solver_cfg.adaptive = True
        env_cfg.sim.physics.use_cuda_graph = False
    if os.environ.get("CHECK_PHYSX") == "1":
        from isaaclab_tasks.contrib.trossen_spatula_lift.trossen_spatula_lift_env_cfg import (
            TrossenSpatulaLiftPhysicsCfg,
        )

        env_cfg.sim.physics = TrossenSpatulaLiftPhysicsCfg().physx
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    res["solver"] = (
        "physx"
        if os.environ.get("CHECK_PHYSX") == "1"
        else type(
            getattr(
                __import__("isaaclab_newton.physics.newton_manager", fromlist=["NewtonManager"]), "NewtonManager"
            )._solver
        ).__name__
    )
    names = list(robot.joint_names)
    res["joint_names"] = names
    res["num_joints"] = len(names)

    def jidx(n):
        return names.index(n)

    L_CARR = "follower_left_left_carriage_joint"
    R_CARR = "follower_left_right_carriage_joint"
    ARM_J1 = "follower_left_joint_1"
    res["has_right_carriage"] = R_CARR in names

    obs, _ = env.reset()
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)

    def jp(name):
        return float(robot.data.joint_pos.torch[:, jidx(name)].mean())

    def ee():
        return u.scene["ee_frame"].data.target_pos_w.torch[..., 0, :].mean(dim=0)

    # --- arm tracks a held position action -------------------------------------
    q1_0, ee_0 = jp(ARM_J1), ee().clone()
    act[:] = 0.0
    act[:, 1] = 0.8  # joint_1 target offset = 0.8 * scale 0.5 = 0.4 rad
    act[:, 6] = 1.0  # gripper open
    for _ in range(40):
        obs, rew, term, trunc, info = env.step(act)
    q1_1, ee_1 = jp(ARM_J1), ee().clone()
    res["arm_joint1_delta_rad"] = round(q1_1 - q1_0, 4)
    res["ee_moved_m"] = round(float(torch.norm(ee_1 - ee_0)), 4)
    res["arm_ok"] = abs(q1_1 - q1_0) > 0.15
    res["ee_ok"] = float(torch.norm(ee_1 - ee_0)) > 0.02

    # --- gripper open ----------------------------------------------------------
    lo, ro = jp(L_CARR), (jp(R_CARR) if res["has_right_carriage"] else None)
    res["gripper_open_left"] = round(lo, 4)
    res["gripper_open_right"] = round(ro, 4) if ro is not None else None
    res["gripper_open_ok"] = lo > 0.035

    # --- gripper close ---------------------------------------------------------
    act[:, 6] = -1.0
    for _ in range(30):
        obs, rew, term, trunc, info = env.step(act)
    lc, rc = jp(L_CARR), (jp(R_CARR) if res["has_right_carriage"] else None)
    res["gripper_closed_left"] = round(lc, 4)
    res["gripper_closed_right"] = round(rc, 4) if rc is not None else None
    res["gripper_close_ok"] = lc < 0.01
    # Mimic: right carriage should mirror (gearing -1 about its offset range) -- the
    # observable contract is simply that it MOVES with the left by a comparable span.
    if res["has_right_carriage"]:
        res["mimic_span_right"] = round(abs((ro if ro is not None else 0) - (rc if rc is not None else 0)), 4)
        res["mimic_ok"] = res["mimic_span_right"] > 0.02
    else:
        res["mimic_ok"] = False

    res["rew_finite"] = bool(torch.isfinite(rew).all())
    res["ok"] = all(res[k] for k in ("arm_ok", "ee_ok", "gripper_open_ok", "gripper_close_ok", "rew_finite"))
    env.close()
except Exception as e:
    res.update({"ok": False, "err": repr(e), "tb": traceback.format_exc()[-1200:]})

with open(os.environ["CHECK_OUT"], "w") as f:
    json.dump(res, f, indent=1)
os._exit(0 if res.get("ok") else 1)
