"""Approach rung: reset-time body force and mug displacement vs alpha (does the home->grasp joint path pass through the mug?)."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_flip.mdp import _tip_handle_distance
from isaaclab.managers import SceneEntityCfg
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.events.reset_arm_hover_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device; A=env.action_manager.total_action_dim
params=env.event_manager.get_term_cfg("reset_arm_hover_bank").params
wc=SceneEntityCfg("robot", body_names=["follower_left_link_6"]); wc.resolve(env.scene)
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
aid=robot.find_joints("follower_left_joint_[0-5]")[0]
for a in (1.0,0.5,0.0):
    params["alpha_min"]=a; params["noise"]=0.0
    # force alpha exactly a: sample alpha ~ U(a,1) -> use a narrow band by setting alpha_min=a and reading; instead run 3 resets and take medians
    env.reset(); spawn=t(obj.data.root_pos_w).clone(); q0=t(robot.data.joint_pos)[:,aid]; tip0=_tip_handle_distance(env,SceneEntityCfg("object"),SceneEntityCfg("ee_frame"),wc); z0=t(env.scene["ee_frame"].data.target_pos_w)[:,0,2]-org[:,2]
    print(f"[path]   at reset: j1 {q0[:,1].min():.2f}..{q0[:,1].max():.2f} j2 {q0[:,2].min():.2f}..{q0[:,2].max():.2f} j3 {q0[:,3].min():.2f}..{q0[:,3].max():.2f}  TCP z {z0.min():.3f}..{z0.max():.3f}  tip-handle {tip0.min()*100:.1f}..{tip0.max()*100:.1f} cm", flush=True)
    act=torch.zeros(N,A,device=dev); bmax=torch.zeros(N,device=dev)
    for k in range(10): env.step(act); bmax=torch.maximum(bmax,fm("pad_body_contact"))
    d=(t(obj.data.root_pos_w)[:,:2]-spawn[:,:2]).norm(dim=-1); tip=_tip_handle_distance(env,SceneEntityCfg("object"),SceneEntityCfg("ee_frame"),wc); z10=t(env.scene["ee_frame"].data.target_pos_w)[:,0,2]-org[:,2]; q10=t(robot.data.joint_pos)[:,aid]
    print(f"[path]   after 10 steps: TCP z {z10.min():.3f}..{z10.max():.3f}  tip-handle {tip.min()*100:.1f}..{tip.max()*100:.1f} cm  j2 {q10[:,2].min():.2f}..{q10[:,2].max():.2f}", flush=True)
    print(f"[path] alpha_min {a:.2f} (starts U[{a:.2f},1]): body F max over 10 steps median {bmax.median():5.1f} N (>5N in {(bmax>5).float().mean()*100:3.0f}%)  mug drift median {d.median()*1000:4.1f} mm  tip-handle median {tip.median()*100:4.1f} cm", flush=True)
env.close()
