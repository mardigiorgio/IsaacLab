"""Low (rim-end) handle pinch: bank it, close at several squeeze levels, lift, settle -- does gravity swing the mug base-down while held?"""
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
for e in ("reset_arm_lift_bank","reset_arm_rotate_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.events.reset_arm_grasp_bank.params["bank_fraction"]=1.0; cfg.curriculum=None
LOW={"follower_left_joint_0":-0.0001,"follower_left_joint_1":1.7416,"follower_left_joint_2":0.7745,"follower_left_joint_3":0.0754,"follower_left_joint_4":0.0,"follower_left_joint_5":-0.0001,"follower_left_left_carriage_joint":0.0060,"follower_left_right_carriage_joint":0.0060}
env=gym.make(TASK,cfg=cfg).unwrapped
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device; A=env.action_manager.total_action_dim
term=env.action_manager.get_term("arm_action"); sc=t(term._scale)[0] if torch.is_tensor(term._scale) else torch.full((6,),float(term._scale),device=dev)
params=env.event_manager.get_term_cfg("reset_arm_grasp_bank").params
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
params["pose"]=LOW; params["gripper_offset"]=-0.05
for j3v in (1.0,1.2):
    for j4v in (-0.8,-0.4,0.4,0.8):
        env.reset(); act=torch.zeros(N,A,device=dev)
        for k in range(15): env.step(act)
        for k in range(40):
            act[:,1]=-0.30*min(k/40,1.0)/sc[1]; env.step(act)
        for k in range(60):
            a=min(k/45,1.0); act[:,3]=(a*j3v)/sc[3]; act[:,4]=(a*j4v)/sc[4]; act[:,5]=(a*-3.0)/sc[5]; env.step(act)
        for k in range(60): env.step(act)
        L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); u=upcos()
        print(f"[static] j3 {j3v:.1f} j4 {j4v:+.1f} j5 -3.0: STATIC up_cos after 2 s med {u.median():+.2f} (>0.7 in {(u>0.7).float().mean()*100:.0f}%, >0.5 in {(u>0.5).float().mean()*100:.0f}%)  held {h.float().mean()*100:.0f}%", flush=True)
env.close()
