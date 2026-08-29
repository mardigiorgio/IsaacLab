"""From the held bank: lift, then sweep wrist pitch (j4) x roll (j5) slowly; can the held mug reach upright (up_cos > 0.87)?"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, numpy as np, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
TASK="IsaacContrib-Flip-Mug-Trossen-v0"
J3=np.linspace(-1.4,1.4,5); J4=np.linspace(-1.5,1.5,6); J5=np.linspace(-3.0,3.0,6); grid=[(c,a,b) for c in J3 for a in J4 for b in J5]; N=len(grid)
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
cfg.events.reset_arm_grasp_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped; env.reset()
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
A=env.action_manager.total_action_dim; term=env.action_manager.get_term("arm_action"); sc=t(term._scale)[0] if torch.is_tensor(term._scale) else torch.full((6,),float(term._scale),device=dev)
q0=t(robot.data.joint_pos)[:, robot.find_joints("follower_left_joint_[0-5]")[0]].clone()  # bank pose (offsets)
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
act=torch.zeros(N,A,device=dev); held_all=torch.ones(N,dtype=torch.bool,device=dev); best_up=torch.full((N,),-2.0,device=dev)
def step(k_extra=1):
    global held_all,best_up
    env.step(act); L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); held_all&=h
    best_up=torch.maximum(best_up, torch.where(h, upcos(), torch.full_like(best_up,-2.0)))
act[:,-2:]=-1.0  # gripper fully closed: a lifting policy must command this (14 N squeeze alone drops the mug)
for k in range(15): step()
held_all[:]=True; best_up[:]=-2.0
lift=-0.30  # j1 back = up (proof raise direction)
for k in range(40):
    act[:,1]=lift*min(k/40,1.0)/sc[1]; step()
z_lift=(t(obj.data.root_pos_w)[:,2]-org[:,2]).clone(); held_after_lift=held_all.clone()
tgt=torch.tensor(grid,device=dev,dtype=torch.float32)
for k in range(90):
    a=min(k/90,1.0); act[:,3]=(a*tgt[:,0])/sc[3]; act[:,4]=(a*tgt[:,1])/sc[4]; act[:,5]=(a*tgt[:,2])/sc[5]; step()
for k in range(30): step()
up=upcos(); L,R,B=fm("pad_left_handle"),fm("pad_right_handle"),fm("pad_body_contact"); held_end=(L>0.01)&(R>0.01)
print(f"[rot] held after lift {held_after_lift.float().mean()*100:.0f}%  mug z after lift median {z_lift.median():.3f} (rest 0.119)", flush=True)
print(f"[rot] best up_cos WHILE HELD over the sweep: max {best_up.max():.3f}; envs reaching >0.87 held: {(best_up>0.87).sum().item()}/{N}; >0.5: {(best_up>0.5).sum().item()}; >0.0: {(best_up>0.0).sum().item()}", flush=True)
order=torch.argsort(best_up,descending=True)[:8]
for i in order.tolist(): print(f"[rot] j3={grid[i][0]:+.2f} j4={grid[i][1]:+.2f} j5={grid[i][2]:+.2f}: best held up_cos {best_up[i]:+.3f}, held at end {bool(held_end[i])}", flush=True)
print(f"[rot] cells with held up_cos > 0.3: {(best_up>0.3).sum().item()}, > 0.5: {(best_up>0.5).sum().item()}, > 0.7: {(best_up>0.7).sum().item()} of {N}", flush=True)
env.close()
