"""Nominal descent only: offset via -> (when within tol of via) open-grasp pose, jaws open, zero policy.
Reports fingertip-to-pinch-point distance, body force and pinch feasibility (both pads on the handle after a close) at arrival."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--num_envs",type=int,default=128); parser.add_argument("--tol",type=float,default=0.08); parser.add_argument("--settle",type=int,default=25)
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_flip import trossen_mug_flip_env_cfg as C
from isaaclab_tasks.contrib.trossen_mug_flip.mdp import _tip_handle_distance
from isaaclab.managers import SceneEntityCfg
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=args.num_envs
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=N*800; cfg.sim.physics.collision_cfg.max_triangle_pairs=N*16000; cfg.curriculum=None
for e in [n for n in vars(cfg.events) if n.startswith("reset_arm_") and n.endswith("_bank")]: getattr(cfg.events,e).params["bank_fraction"]=0.0
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; dev=u.device; t=lambda x: x.torch if hasattr(x,"torch") else x
arm=u.action_manager.get_term("arm_action"); grip=u.action_manager.get_term("gripper_action"); robot=u.scene["robot"]; aid=robot.find_joints(list(arm._joint_names),preserve_order=True)[0]
P=lambda pose: torch.tensor([pose[n] for n in arm._joint_names],device=dev); VIA,OPEN=P(C.FLIP_VIA_POSE),P(C.FLIP_OPEN_GRASP_BANK_POSE)
wc=SceneEntityCfg("robot", body_names=["follower_left_link_6"]); wc.resolve(u.scene)
def fm(name):
    f=u.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
obs,_=env.reset(); phase=torch.zeros(N,dtype=torch.long,device=dev); arrived=torch.full((N,),-1,dtype=torch.long,device=dev); z=torch.zeros(N,u.action_manager.total_action_dim,device=dev)
with torch.inference_mode():
    for k in range(120):
        q=t(robot.data.joint_pos)[:,aid]
        phase=torch.where((phase==0)&((q-VIA).abs().max(1).values<args.tol),torch.ones_like(phase),phase)
        near=(phase==1)&((q-OPEN).abs().max(1).values<0.03)&(arrived<0); arrived=torch.where(near,torch.full_like(arrived,k),arrived)
        close=(arrived>=0)&(k-arrived>=args.settle)  # after settling at OPEN: squeeze
        arm._offset[:]=torch.where((phase==1).unsqueeze(1),OPEN,VIA); grip._offset[:]=torch.where(close.unsqueeze(1),torch.full_like(grip._offset,-0.05),torch.full_like(grip._offset,0.012))
        obs,*_=env.step(z)
        if k in (10,20,30,40,50,60,80,100,119):
            d=_tip_handle_distance(u,SceneEntityCfg("object"),SceneEntityCfg("ee_frame"),wc); B=fm("pad_body_contact"); held=(fm("pad_left_handle")>0.01)&(fm("pad_right_handle")>0.01)
            mz=(t(u.scene["object"].data.root_pos_w)[:,2]-t(u.scene.env_origins)[:,2])
            print(f"[nd] k={k:3d} phase1 {int((phase==1).sum())} arrived {int((arrived>=0).sum())} closing {int(close.sum())} | tip-handle med {d.median()*100:.1f} cm p90 {d.quantile(0.9)*100:.1f} cm | body F med {B.median():.1f} max {B.max():.0f} N | held {held.float().mean()*100:.0f}% | mug z med {mz.median():.3f} min {mz.min():.3f}", flush=True)
env.close()
