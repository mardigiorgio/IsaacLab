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
import numpy as np
aid=robot.find_joints("follower_left_joint_[0-5]")[0]; gid=robot.find_joints(["follower_left_left_carriage_joint","follower_left_right_carriage_joint"],preserve_order=True)[0]
pad_ids=robot.find_bodies("follower_left_gripper_.*",preserve_order=True)[0]; l6=robot.find_bodies("follower_left_link_6")[0][0]
limits=t(robot.data.joint_pos_limits)[0]
Q_ROT=np.array([-0.0001,1.4758,0.7442,1.4397,-0.0001,-2.9499])
def fk_batch(qb):  # write joints (mug untouched), one step, read TCP / tool / jaw per env
    q=t(robot.data.joint_pos).clone(); q[:len(qb),aid]=torch.tensor(qb,dtype=torch.float32,device=dev); q[:, gid]=0.006
    robot.write_joint_state_to_sim(q, torch.zeros_like(q)); env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=env.physics_dt)
    tcp=(t(env.scene["ee_frame"].data.target_pos_w)[:,0,:]-org).cpu().numpy(); pads=t(robot.data.body_pos_w)[:,pad_ids]; sep=(pads[:,0]-pads[:,1]).cpu().numpy(); sep/=np.linalg.norm(sep,axis=1,keepdims=True)
    w=(t(robot.data.body_pos_w)[:,l6]-org).cpu().numpy(); tool=tcp-w; tool/=np.linalg.norm(tool,axis=1,keepdims=True); return tcp,sep,tool
EPS=1e-3
def solve(q0,target,jaw_z,tool_z,iters=200,lam=0.05):
    qq=q0.copy()
    for _ in range(iters):
        b=np.repeat(qq[None,:],13,axis=0)
        for j in range(6): b[1+2*j,j]+=EPS; b[2+2*j,j]-=EPS
        pos,seps,tools=fk_batch(b); e_pos=target-pos[0]; e_jz=np.array([jaw_z-abs(seps[0][2])*np.sign(jaw_z) if jaw_z!=0 else -seps[0][2]]); e_tz=np.array([tool_z-tools[0][2]])
        if np.linalg.norm(e_pos)<0.003 and abs(e_jz[0])<0.05 and abs(e_tz[0])<0.05: break
        J=np.zeros((5,6))
        for j in range(6):
            J[:3,j]=(pos[1+2*j]-pos[2+2*j])/(2*EPS); J[3,j]=(abs(seps[1+2*j][2])*np.sign(jaw_z)-abs(seps[2+2*j][2])*np.sign(jaw_z))/(2*EPS) if jaw_z!=0 else (seps[1+2*j][2]-seps[2+2*j][2])/(2*EPS); J[4,j]=(tools[1+2*j][2]-tools[2+2*j][2])/(2*EPS)
        e=np.concatenate([e_pos,0.15*e_jz,0.15*e_tz]); dq=J.T@np.linalg.solve(J@J.T+lam*lam*np.eye(5),e); qq=qq+np.clip(dq,-0.2,0.2)
        lo=limits[aid,0].cpu().numpy(); hi=limits[aid,1].cpu().numpy(); qq=np.clip(qq,lo+1e-3,hi-1e-3)
    return qq,float(np.linalg.norm(e_pos)),float(max(abs(e_jz[0]),abs(e_tz[0])))
def snap(tag, h):
    print(f"[chain]   {tag} state (median over held envs): arm {[round(float(v),4) for v in t(robot.data.joint_pos)[h][:,aid].median(0).values]} mug {[round(float(v),4) for v in (t(obj.data.root_pos_w)-org)[h].median(0).values]} quat {[round(float(v),4) for v in t(obj.data.root_quat_w)[h].median(0).values]}", flush=True)
def axes():
    tcp=(t(env.scene["ee_frame"].data.target_pos_w)[:,0,:]-org); w=(t(robot.data.body_pos_w)[:,l6]-org); tool=tcp-w; tool=tool/tool.norm(dim=-1,keepdim=True); return tcp,tool
for j3v in (1.4,-1.4):
    for j5v in (-3.0,3.0):
        for j4v in (-1.0,0.0,1.0):
            env.reset(); act=torch.zeros(N,A,device=dev)
            for k in range(15): env.step(act)
            for k in range(40):
                act[:,1]=-0.30*min(k/40,1.0)/sc[1]; env.step(act)
            best=torch.full((N,),-2.0,device=dev); btz=torch.zeros(N,device=dev); bz=torch.zeros(N,device=dev)
            for k in range(80):
                a=min(k/60,1.0); act[:,3]=(a*j3v)/sc[3]; act[:,5]=(a*j5v)/sc[5]; act[:,4]=(a*j4v)/sc[4]; env.step(act)
                L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01); u=upcos(); tcp,tool=axes()
                better=h&(u>best); best=torch.where(better,u,best); btz=torch.where(better,tool[:,2],btz); bz=torch.where(better,tcp[:,2],bz)
            L,R=fm("pad_left_handle"),fm("pad_right_handle"); h=(L>0.01)&(R>0.01)
            m=best>-2
            print(f"[dir] j3 {j3v:+.1f} j5 {j5v:+.1f} j4 {j4v:+.1f}: held@end {h.float().mean()*100:3.0f}%  best held up_cos med {best[m].median():+.2f}  tool_z at best med {btz[m].median():+.2f}  TCP z at best {bz[m].median():.2f}  frac up>0.7 {(best>0.7).float().mean()*100:3.0f}%", flush=True)
env.close()
