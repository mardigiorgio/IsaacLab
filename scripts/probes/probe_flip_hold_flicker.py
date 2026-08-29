"""Rotated bank, zero action (offsets at the rotated pose): do the hold predicates flicker frame to frame?"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_slide.mdp import _sensor_force_mag
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.events.reset_arm_rotate_bank.params["bank_fraction"]=1.0; cfg.events.reset_arm_rotate_bank.params["offset_pose"]=None; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; dev=env.device; A=env.action_manager.total_action_dim
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
env.reset(); act=torch.zeros(N,A,device=dev)
streak_h=torch.zeros(N,device=dev); best_h=torch.zeros(N,device=dev); streak_r=torch.zeros(N,device=dev); best_r=torch.zeros(N,device=dev); streak_u=torch.zeros(N,device=dev); best_u=torch.zeros(N,device=dev)
flick=torch.zeros(N,device=dev); frames=0
for k in range(90):
    env.step(act); L,R=fm("pad_left_handle"),fm("pad_right_handle"); held=(L>0.01)&(R>0.01); rel=_sensor_force_mag(env,"pad_object_contact")<0.01; u=upcos(); st=env._fsm["stage"]
    d_h=(st>=3)&(u>0.7)&held; d_r=(st>=3)&(u>0.7)&~rel; d_u=(u>0.7)
    streak_h=torch.where(d_h,streak_h+1,torch.zeros_like(streak_h)); best_h=torch.maximum(best_h,streak_h)
    streak_r=torch.where(d_r,streak_r+1,torch.zeros_like(streak_r)); best_r=torch.maximum(best_r,streak_r)
    streak_u=torch.where(d_u,streak_u+1,torch.zeros_like(streak_u)); best_u=torch.maximum(best_u,streak_u)
    if k>=10: flick+=(~held).float(); frames+=1
print(f"[flicker] over 90 zero-action frames from the rotated bank: both-pads 'held' false on {flick.mean()/frames*100:.1f}% of frames (after settle)", flush=True)
print(f"[flicker] longest consecutive streak: stage>=3 & up>0.7 & held  median {best_h.median():.0f} (>=30 in {(best_h>=30).float().mean()*100:.0f}%) | stage>=3 & up>0.7 & any-pad-contact  median {best_r.median():.0f} (>=30 in {(best_r>=30).float().mean()*100:.0f}%) | up>0.7 alone median {best_u.median():.0f}", flush=True)
print(f"[flicker] final up_cos med {upcos().median():+.2f}, FSM stage med {float(env._fsm['stage'].float().median()):.1f}", flush=True)
env.close()
