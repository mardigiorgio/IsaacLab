"""Capture the LIFTED-HELD state (arm joints + mug pose) reached from the held bank with the gripper closed."""
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
env=gym.make(TASK,cfg=cfg).unwrapped; env.reset()
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
A=env.action_manager.total_action_dim; term=env.action_manager.get_term("arm_action"); sc=t(term._scale)[0] if torch.is_tensor(term._scale) else torch.full((6,),float(term._scale),device=dev)
aid=robot.find_joints("follower_left_joint_[0-5]")[0]; gid=robot.find_joints(["follower_left_left_carriage_joint","follower_left_right_carriage_joint"],preserve_order=True)[0]
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
act=torch.zeros(N,A,device=dev); act[:,-2:]=-1.0
for k in range(15): env.step(act)
for k in range(40):
    act[:,1]=-0.30*min(k/40,1.0)/sc[1]; env.step(act)
for k in range(30): env.step(act)   # settle lifted
snap=(t(robot.data.joint_pos).clone(), t(robot.data.joint_vel).clone(), t(obj.data.root_pose_w).clone(), t(obj.data.root_vel_w).clone())
def restore():
    robot.write_joint_state_to_sim(snap[0].clone(), snap[1].clone()); obj.write_root_pose_to_sim_index(root_pose=snap[2].clone()); obj.write_root_velocity_to_sim_index(root_velocity=snap[3].clone()); env.scene.write_data_to_sim()
base=act.clone()
for tag,ss,ws in (("zero action (restore sanity)",0.0,(0,0,0)),("all sigma 0.05",0.05,(0.05,0.05,0.05)),("shoulder 0.05 / wrist 0.1",0.05,(0.1,0.1,0.1)),("shoulder 0.05 / wrist 0.05/0.05/0.15",0.05,(0.05,0.05,0.15)),("all sigma 0.1 (repeat)",0.1,(0.1,0.1,0.1))):
    restore()
    for k in range(30):
        a=base.clone(); a[:,:3]+=ss*torch.randn(N,3,device=dev); a[:,3]+=ws[0]*torch.randn(N,device=dev); a[:,4]+=ws[1]*torch.randn(N,device=dev); a[:,5]+=ws[2]*torch.randn(N,device=dev); env.step(a)
    L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); z=t(obj.data.root_pos_w)[:,2]-org[:,2]
    print(f"[lifted] LIFTED + {tag}: held after 30 steps {h.float().mean()*100:.0f}%  mug still up (z>0.15) {(z>0.15).float().mean()*100:.0f}%", flush=True)
env.close()
