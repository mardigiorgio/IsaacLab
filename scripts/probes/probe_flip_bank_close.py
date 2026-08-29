# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flip bank existence proof through the ENV pathway (reset event + action terms).

Every env resets from FLIP_GRASP_BANK_POSE (bank_fraction 1), holds with zero
arm action (the PD sag the policy will actually see), closes, then raises the
shoulder. Pass = handle forces on BOTH pads with 0 N on the body after the close,
and the mug root rising with the raise. Ran 2026-08-28: close 19.8/19.8/0.0 N,
raise 107 N handle-only, mug +12 cm.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_flip_bank_close.py
"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, numpy as np, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_flip.mdp import FINGER_LEN
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=16
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=100_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=2_000_000
cfg.events.reset_arm_grasp_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped; env.reset()
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
l6=robot.find_bodies("follower_left_link_6")[0][0]
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return 0.0 if f is None else float(torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).mean())
def report(tag):
    tcp=t(env.scene["ee_frame"].data.target_pos_w)[:,0,:]-org; w=t(robot.data.body_pos_w)[:,l6]-org; tool=(tcp-w); tool=tool/tool.norm(dim=-1,keepdim=True); tip=tcp+FINGER_LEN*tool
    car=t(robot.data.joint_pos)[:,robot.find_joints("follower_left_left_carriage_joint")[0][0]]
    print(f"[bank-close {tag}] TCP mean {tcp.mean(0).cpu().numpy().round(3)}  fingertip mean {tip.mean(0).cpu().numpy().round(3)} (bar: y 0.059-0.065, z 0.029-0.092)  carriage {car.mean()*1000:.1f} mm  forces L/R/body {fm('pad_left_handle'):.1f}/{fm('pad_right_handle'):.1f}/{fm('pad_body_contact'):.1f} N  mug z {float((t(obj.data.root_pos_w)[:,2]-org[:,2]).mean()):.3f}", flush=True)
A=env.action_manager.total_action_dim; act=torch.zeros(N,A,device=dev)
report("t=0 (after reset)")
for k in range(15):
    env.step(act)
report("t=15 hold, zero action")
act[:,-2:]=-1.0
for k in range(20):
    env.step(act)
report("t=35 after CLOSE")
act[:,-2:]=-1.0; j1=1  # raise shoulder a bit to test hold
for k in range(30):
    act[:,j1]=-0.4*min(k/30,1.0)/0.5
    env.step(act)
report("t=65 after raise")
env.close()
