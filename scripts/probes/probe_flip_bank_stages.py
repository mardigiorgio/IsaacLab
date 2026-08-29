"""Lifted / rotated banks through the env pathway: hold survival, body force and FSM stage reached under zero action, per squeeze offset."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, numpy as np, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
A=env.action_manager.total_action_dim
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
for bank in ("reset_arm_lift_bank","reset_arm_rotate_bank"):
    for goff in (-0.01,-0.05,-0.15):
        for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank"): env.event_manager.get_term_cfg(e).params["bank_fraction"]=0.0
        env.event_manager.get_term_cfg(bank).params["bank_fraction"]=1.0; env.event_manager.get_term_cfg(bank).params["gripper_offset"]=goff
        env.reset(); act=torch.zeros(N,A,device=dev); maxstage=torch.zeros(N,dtype=torch.long,device=dev); bodymax=torch.zeros(N,device=dev); heldsteps=torch.zeros(N,device=dev)
        for k in range(45):
            env.step(act); L,R,B=fm("pad_left_handle"),fm("pad_right_handle"),fm("pad_body_contact"); h=(L>0.01)&(R>0.01)
            heldsteps+=h.float(); bodymax=torch.maximum(bodymax,B); maxstage=torch.maximum(maxstage,env._fsm["stage"])
        print(f"[stages] {bank:22s} goff {goff:+.2f}: held steps mean {heldsteps.mean():4.1f}/45  held@45 {h.float().mean()*100:3.0f}%  body F max median {bodymax.median():5.2f} N (>5N in {(bodymax>5).float().mean()*100:.0f}%)  FSM stage reached >=2 {(maxstage>=2).float().mean()*100:3.0f}%  >=3 {(maxstage>=3).float().mean()*100:3.0f}%  up_cos now median {upcos().median():+.2f}", flush=True)
env.close()
