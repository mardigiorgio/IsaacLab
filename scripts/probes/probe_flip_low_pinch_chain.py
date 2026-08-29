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
for ramp,j3v in ((15,1.0),(15,1.2),(25,1.0),(45,1.0)):
    env.reset(); act=torch.zeros(N,A,device=dev)
    for k in range(15): env.step(act)
    for k in range(25):
        act[:,1]=-0.30*min(k/25,1.0)/sc[1]; env.step(act)
    trace=[]; hold=torch.zeros(N,device=dev); best=torch.zeros(N,device=dev); succ=torch.zeros(N,dtype=torch.bool,device=dev)
    for k in range(ramp+75):
        a=min(k/ramp,1.0); act[:,3]=(a*j3v)/sc[3]; act[:,5]=(a*-3.0)/sc[5]; env.step(act); u=upcos()
        L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); st=env._fsm["stage"]
        d=(st>=3)&(u>0.5)&h; hold=torch.where(d,hold+1,torch.zeros_like(hold)); best=torch.maximum(best,hold); succ|=env._fsm["success"]
        if k%5==4: trace.append(f"{u.median():+.2f}")
    print(f"[swing] ramp {ramp:2d} j3 {j3v:.1f}: up_cos median every 5 frames: {' '.join(trace)}", flush=True)
    print(f"[swing]    longest hold streak median {best.median():.0f} frames (>=30 in {(best>=30).float().mean()*100:.0f}%), FSM success fired in {succ.float().mean()*100:.0f}%, held@end {h.float().mean()*100:.0f}%", flush=True)
env.close()
