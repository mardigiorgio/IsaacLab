"""Scripted NOMINAL plan from true home with a ZERO policy: the arm action offset is
switched per env by a small phase machine (via -> open-grasp -> squeeze -> lifted ->
rotated), so zero action executes the plan. Reports success/pinch from home."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--num_envs",type=int,default=128); parser.add_argument("--steps",type=int,default=150)
parser.add_argument("--tol",type=float,default=0.05); parser.add_argument("--squeeze",type=float,default=-0.05)
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_tasks.contrib.trossen_mug_flip import trossen_mug_flip_env_cfg as C
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=args.num_envs
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=N*800; cfg.sim.physics.collision_cfg.max_triangle_pairs=N*16000; cfg.curriculum=None
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank","reset_arm_hover_bank","reset_arm_home_via_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; dev=u.device
t=lambda x: x.torch if hasattr(x,"torch") else x
arm=u.action_manager.get_term("arm_action"); grip=u.action_manager.get_term("gripper_action")
robot=u.scene["robot"]; aid=robot.find_joints(list(arm._joint_names),preserve_order=True)[0]
def P(pose): return torch.tensor([pose[n] for n in arm._joint_names],device=dev)
VIA,OPEN,LIFT,ROT=P(C.FLIP_VIA_POSE),P(C.FLIP_OPEN_GRASP_BANK_POSE),P(C.FLIP_LIFTED_BANK_POSE),P(C.FLIP_ROTATED_BANK_POSE)
obs,_=env.reset(); phase=torch.zeros(N,dtype=torch.long,device=dev); age=torch.zeros(N,dtype=torch.long,device=dev)
succ=torch.zeros(N,dtype=torch.bool,device=dev); pinched=torch.zeros(N,dtype=torch.bool,device=dev); ph_hist=[]
with torch.inference_mode():
    for k in range(args.steps):
        q=t(robot.data.joint_pos)[:,aid]
        err=lambda tgt: (q-tgt).abs().max(dim=1).values
        fsm=getattr(u,"_fsm",None); pe=fsm["pinched_ever"] if fsm is not None else torch.zeros(N,dtype=torch.bool,device=dev)
        # transitions
        adv0=(phase==0)&((err(VIA)<args.tol)|(age>30)); adv1=(phase==1)&((err(OPEN)<args.tol)|(age>40)); adv2=(phase==2)&pe; adv3=(phase==3)&((err(LIFT)<args.tol)|(age>30))
        adv=adv0|adv1|adv2|adv3; phase=torch.where(adv,phase+1,phase); age=torch.where(adv,torch.zeros_like(age),age+1)
        # offsets by phase
        tgt=torch.where((phase==0).unsqueeze(1),VIA,torch.where((phase==1).unsqueeze(1),OPEN,torch.where((phase<=3).unsqueeze(1),torch.where((phase==2).unsqueeze(1),OPEN,LIFT),ROT)))
        arm._offset[:]=tgt
        grip._offset[:]=torch.where((phase>=2).unsqueeze(1),torch.full_like(grip._offset,args.squeeze),torch.full_like(grip._offset,0.012))
        obs,*_=env.step(torch.zeros(N,u.action_manager.total_action_dim,device=dev))
        fsm=u._fsm; succ|=fsm["success"]; pinched|=fsm["pinched_ever"]
        if k%10==0 or k==args.steps-1:
            ph=[int((phase==i).sum()) for i in range(5)]; z=(t(u.scene["object"].data.root_pos_w)[:,2]-t(u.scene.env_origins)[:,2]).median()
            qo=t(u.scene["object"].data.root_quat_w); up=(1-2*(qo[:,0]**2+qo[:,1]**2)).median()
            print(f"[nom] t={k:3d} phases {ph}  pinched_ever {pinched.float().mean()*100:3.0f}%  success {succ.float().mean()*100:3.0f}%  mug z med {z:.3f} up_cos med {up:+.2f}  stage med {float(fsm['stage'].float().median()):.1f}", flush=True)
print(f"[nom] RESULT nominal plan from TRUE HOME (zero policy): success {succ.float().mean()*100:.1f}%  pinched {pinched.float().mean()*100:.1f}%  of {N}", flush=True)
env.close()
