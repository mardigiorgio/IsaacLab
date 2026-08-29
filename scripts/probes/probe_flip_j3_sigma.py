"""From the LIFTED bank (hard squeeze): survival and rotate-progress gain under per-joint exploration noise, 30 and 60 steps."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.events.reset_arm_lift_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device; A=env.action_manager.total_action_dim
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
torch.manual_seed(0)
for tag,sig in (("all 0.05, j5 0.15 (current)",[.05,.05,.05,.05,.05,.15,.05,.05]),("j3 0.10",[.05,.05,.05,.10,.05,.15,.05,.05]),("j3 0.15",[.05,.05,.05,.15,.05,.15,.05,.05]),("j3 0.20",[.05,.05,.05,.20,.05,.15,.05,.05]),("j3 0.15 + j3 bias +0.3",[.05,.05,.05,.15,.05,.15,.05,.05])):
    env.reset(); s=torch.tensor(sig,device=dev); best=torch.full((N,),-2.0,device=dev); u0=upcos().clone(); gain=torch.zeros(N,device=dev)
    for k in range(60):
        a=s*torch.randn(N,A,device=dev)
        if "bias" in tag: a[:,3]+=0.3
        env.step(a); L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); u=upcos()
        nb=torch.where(h,torch.maximum(best,u),best); gain+=torch.where(h,(nb-best).clamp(min=0)*(best>-2).float(),torch.zeros_like(gain)); best=torch.where(best<=-2,torch.where(h,u,best),nb)
        if k==29: h30=h.float().mean(); g30=gain.median()
    print(f"[j3] {tag:32s} held@30 {h30*100:3.0f}%  held@60 {h.float().mean()*100:3.0f}%  ratchet-able up_cos gain median @30 {g30:+.3f} @60 {gain.median():+.3f}  best held up_cos median {best[best>-2].median():+.2f}", flush=True)
env.close()
