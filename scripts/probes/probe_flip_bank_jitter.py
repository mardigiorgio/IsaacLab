"""Bank start under EXPLORATION: open-jaw vs seeded-pinch bank, zero-action hold vs random actions (sigma 0.9)."""
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
cfg.events.reset_arm_grasp_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
pose=env.event_manager.get_term_cfg("reset_arm_grasp_bank").params["pose"]
def forces(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
from isaaclab_tasks.contrib.trossen_mug_flip.mdp import FINGER_LEN
l6=robot.find_bodies("follower_left_link_6")[0][0]
def stats(tag, spawn):
    L,R,B=forces("pad_left_handle"),forces("pad_right_handle"),forces("pad_body_contact"); pinch=((L>0.01)&(R>0.01)).float().mean()
    tcp=t(env.scene["ee_frame"].data.target_pos_w)[:,0,:]; w=t(robot.data.body_pos_w)[:,l6]; tool=tcp-w; tool=tool/tool.norm(dim=-1,keepdim=True); tip=tcp+FINGER_LEN*tool
    inhand=((t(obj.data.root_pos_w)-tip).norm(dim=-1)<0.16).float().mean()   # mug root within 16 cm of the tips (root is 6 cm from the handle)
    d=(t(obj.data.root_pos_w)[:,:2]-spawn[:,:2]).norm(dim=-1); car=t(robot.data.joint_pos)[:,robot.find_joints("follower_left_left_carriage_joint")[0][0]]
    print(f"[jitter] {tag:40s} pinching {pinch*100:5.1f}%  in-hand {inhand*100:5.1f}%  handle F {((L+R)/2).mean():6.2f} N  body F {B.mean():5.2f} N  mug drift med {d.median()*1000:5.1f} mm  carriage med {car.median()*1000:4.1f} mm", flush=True)
A=env.action_manager.total_action_dim
torch.manual_seed(0)
arm_dim=env.action_manager.get_term("arm_action").action_dim
params=env.event_manager.get_term_cfg("reset_arm_grasp_bank").params
for goff in (-0.005, -0.01, -0.015):
    params["gripper_offset"]=goff
    for tag,bias_grip,sigma in (("no bias, sigma 0",0.0,0.0),("grip bias +0.05, sigma 0",0.05,0.0),("grip bias +0.1, sigma 0.1",0.1,0.1),("no bias, sigma 0.1",0.0,0.1)):
        env.reset(); spawn=t(obj.data.root_pos_w).clone()
        for k in range(30):
            act=sigma*torch.randn(N,A,device=dev); act[:,arm_dim:]+=bias_grip
            env.step(act)
        stats(f"HELD goff {goff:+.3f} | {tag} t=30", spawn)
env.close()
