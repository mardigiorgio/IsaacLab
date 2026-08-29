"""From the LIFTED bank (env pathway): scripted wrist trajectories; does ROTATED (up_cos>thr while held, 3 consecutive frames) happen?"""
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
cfg.events.reset_arm_grasp_bank.params["bank_fraction"]=0.0; cfg.events.reset_arm_lift_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
A=env.action_manager.total_action_dim; term=env.action_manager.get_term("arm_action"); sc=t(term._scale)[0] if torch.is_tensor(term._scale) else torch.full((6,),float(term._scale),device=dev)
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
aid=robot.find_joints("follower_left_joint_[0-5]")[0]
def run(tag, target, ramp=60, hold=30):
    env.reset(); act=torch.zeros(N,A,device=dev); act[:,-2:]=-1.0
    got=torch.zeros(N,dtype=torch.bool,device=dev); capq=torch.zeros(N,6,device=dev); capp=torch.zeros(N,3,device=dev); capr=torch.zeros(N,4,device=dev); capk=torch.zeros(N,device=dev)
    maxstage=torch.zeros(N,dtype=torch.long,device=dev)
    for k in range(ramp+hold):
        a=min(k/ramp,1.0)
        for j,v in target.items(): act[:,j]=(a*v)/sc[j]
        env.step(act); u=upcos(); L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01)
        maxstage=torch.maximum(maxstage,env._fsm["stage"])
        new=(u>0.7)&h&~got
        if new.any():
            capq[new]=t(robot.data.joint_pos)[new][:,aid]; capp[new]=(t(obj.data.root_pos_w)-org)[new]; capr[new]=t(obj.data.root_quat_w)[new]; capk[new]=k; got|=new
    n=int(got.sum()); print(f"[srot] {tag}: FSM ROTATED {(maxstage>=3).float().mean()*100:.0f}%; envs with a held up>0.7 frame {n}/{N} (first at step median {capk[got].median() if n else -1:.0f})", flush=True)
    if n:
        print(f"[srot]   ROTATED-held median: arm {[round(float(v),4) for v in capq[got].median(0).values]}", flush=True)
        print(f"[srot]   mug pos {[round(float(v),4) for v in capp[got].median(0).values]} quat {[round(float(v),4) for v in capr[got].median(0).values]}  (pos spread z {capp[got,2].min():.3f}..{capp[got,2].max():.3f})", flush=True)
run("j3 1.4 + j5 -1.8", {3:1.4,5:-1.8})
run("j3 1.4 + j5 -1.8 ramp 45", {3:1.4,5:-1.8}, ramp=45)
env.close()
