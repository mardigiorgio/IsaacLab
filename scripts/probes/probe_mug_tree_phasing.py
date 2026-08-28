# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mug-tree PHASING census by physics timestep (2026-08-28).

Sweeps the fixed step (subdivisions of the 1/90 s boundary) and measures how far
the mug's TRI collision wall penetrates the tree's convex cylinders in two
regimes: DRIVEN (pushed at the trunk at carry speed) and IMPACT (dropped onto
the goal branch from 5 cm). A resting hang never phases at any dt.

Measured on the 4070 Ti (icf, convex tree, gap 0.5 mm, margin 0):
    dt 11.1 ms  driven +2.60 mm (a 4 mm wall)   impact +4.46 mm (branch r 4.55)
    dt  5.6 ms  driven +0.64 mm                 impact +4.14 mm
    dt  2.2 ms  driven  0.00 mm                 impact +3.39 mm
    dt  1.1 ms  driven  0.00 mm                 impact +3.42 mm
Phasing is an IN-MOTION, dt >= ~5 ms phenomenon: at 11 ms the mug travels
5.6 mm per step at 0.5 m/s, more than the wall is thick, so the trunk lands
inside the wall before the 0.5 mm gap band can raise a contact. The "pushed
THROUGH" column of the driven case is not meaningful (the velocity write
overrides the contact response); read the penetration column.

    SUBS=1,2,5,10 SPEED=0.5 PROBE_SOLVER=icf uv run --no-sync --project ~/Documents/research/IsaacLabRubato \
        isaaclab -p scripts/probes/probe_mug_tree_phasing.py
"""

import argparse, os
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, numpy as np, gymnasium as gym, isaaclab_tasks, meshio  # noqa
from scipy.spatial.transform import Rotation as R
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_tree import trossen_mug_tree_env_cfg as M
from isaaclab_tasks.contrib.trossen_mug_tree.assets import TRI_COLLISION_CYLINDERS
TASK="IsaacContrib-MugHang-Trossen-v0"; SOLVER=os.environ.get("PROBE_SOLVER","icf")
SUBS=[int(x) for x in os.environ.get("SUBS","1,2,5,10").split(",")]; SPEED=float(os.environ.get("SPEED","0.5"))
C="source/isaaclab_tasks/isaaclab_tasks/contrib"
vt=meshio.read(f"{C}/trossen_mug_lift/assets/lbm_src/mug_inomata_white_low_16faces.vtk").points.copy(); vt[:,2]-=vt[:,2].min()
Rt=R.from_euler("z",M.TREE_YAW_DEG,degrees=True); T=np.array(M.TREE_POS)
c,rpy,rT,L=TRI_COLLISION_CYLINDERS["trunk_col"]; ta=T+Rt.apply(np.array(c)-[0,0,L/2]); tb=T+Rt.apply(np.array(c)+[0,0,L/2])
def seg_d(P,a,b): ab=b-a; s=np.clip(((P-a)@ab)/(ab@ab),0,1); return np.linalg.norm(P-(a+s[:,None]*ab),axis=1)
for sub in SUBS:
    cfg=parse_env_cfg(TASK,num_envs=1); apply_solver_choice(cfg,SOLVER)
    cfg.sim.dt=(1/90)/sub; cfg.decimation=3*sub; cfg.sim.render_interval=cfg.decimation
    cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
    env=gym.make(TASK,cfg=cfg).unwrapped; env.reset()
    t=lambda x: x.torch if hasattr(x,"torch") else x
    obj=env.scene["object"]; org=t(env.scene.env_origins)[0].cpu().numpy(); dev=env.device
    # ---- DRIVEN: mug upright at trunk height, 8 cm from the trunk axis, pushed straight at it at SPEED m/s
    trunk_xy=T[:2]; start=np.array([trunk_xy[0]-0.08, trunk_xy[1], 0.15]); direction=np.array([1.0,0,0])
    q=np.array([0,0,0.7071,0.7071])
    obj.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(start+org),*q]],device=dev,dtype=torch.float32))
    vel=torch.zeros_like(t(obj.data.root_vel_w)); vel[0,:3]=torch.tensor(direction*SPEED,device=dev,dtype=torch.float32)
    steps_per_tick=sub*3; n_ticks=int(0.6/(cfg.sim.dt*steps_per_tick)); first_cross=-1; pen_max=0.0; k=0
    for tick in range(n_ticks):
        obj.write_root_velocity_to_sim_index(root_velocity=vel); env.scene.write_data_to_sim()   # gripper-like push, reasserted each control tick
        for _ in range(steps_per_tick):
            env.sim.step(render=False); env.scene.update(env.physics_dt); k+=1
            p=t(obj.data.root_pos_w)[0].cpu().numpy()-org; qq=t(obj.data.root_quat_w)[0].cpu().numpy(); P=R.from_quat(qq).apply(vt)+p
            d=seg_d(P,ta,tb); pen=rT-d.min(); pen_max=max(pen_max,pen)
            # true phasing: mug body-axis CENTER passes the trunk surface (wall has gone through)
            if first_cross<0 and np.linalg.norm(p[:2]-trunk_xy) < rT+0.0421-0.004: first_cross=k
    print(f"[drive:{SOLVER}] dt={cfg.sim.dt*1000:.2f}ms sub{sub} push {SPEED} m/s: max wall penetration into trunk {pen_max*1000:+.2f} mm; wall pushed THROUGH trunk at t={first_cross*cfg.sim.dt if first_cross>=0 else -1:.3f}s ({first_cross} steps)", flush=True)
    # ---- DROP: from 5 cm above the goal pose, impact penetration on the branch
    env.reset(); q_goal=np.array(M.GOAL_QUAT_XYZW); p_goal=np.array(M.GOAL_POSE_ENV[0])+[0,0,0.05]
    obj.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(p_goal+org),*q_goal]],device=dev,dtype=torch.float32)); obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(t(obj.data.root_vel_w))); env.scene.write_data_to_sim()
    bp,brpy=M.TRI_FRAMES[M.GOAL_BRANCH]; c2,rpy2,rB,L2=TRI_COLLISION_CYLINDERS[M.GOAL_BRANCH.replace("_base","_col")]; ax=R.from_euler("xyz",rpy2).apply([0,0,1]); ba=T+Rt.apply(np.array(c2)-ax*L2/2); bb=T+Rt.apply(np.array(c2)+ax*L2/2)
    pmax=0.0; tmax=0
    for k in range(int(1.0/cfg.sim.dt)):
        env.sim.step(render=False); env.scene.update(env.physics_dt)
        p=t(obj.data.root_pos_w)[0].cpu().numpy()-org; qq=t(obj.data.root_quat_w)[0].cpu().numpy(); P=R.from_quat(qq).apply(vt)+p
        pen=rB-seg_d(P,ba,bb).min()
        if pen>pmax: pmax,tmax=pen,k
    print(f"[drop:{SOLVER}]  dt={cfg.sim.dt*1000:.2f}ms sub{sub}: peak branch penetration on impact {pmax*1000:+.2f} mm at t={tmax*cfg.sim.dt:.3f}s (branch radius 4.55 mm; >4.55 = branch axis inside the handle wall)", flush=True)
    env.close()
app.close()
