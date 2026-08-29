"""Record the doorknob path: from the LIFTED bank, ramp j3 -> 1.44 / j5 -> -2.95 over `ramp` steps with a squeeze,
and every `every` steps store the per-env (arm+gripper joints, mug pose in env frame) of the envs still holding the
handle. Writes trossen_mug_flip/rotpath.json: a list of {"pose": {joint: val}, "object_pose": [x,y,z,qx,qy,qz,qw], "up_cos": u}."""
import argparse, json, os
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ramp",type=int,default=60); parser.add_argument("--every",type=int,default=4); parser.add_argument("--out",default="")
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
import isaaclab_tasks.contrib.trossen_mug_flip as pkg
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=100_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=2_000_000; cfg.curriculum=None
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank","reset_arm_hover_bank","reset_arm_home_via_bank"): getattr(cfg.events,e).params["bank_fraction"]=(1.0 if e=="reset_arm_lift_bank" else 0.0)
env=gym.make(TASK,cfg=cfg).unwrapped; t=lambda x: x.torch if hasattr(x,"torch") else x
robot=env.scene["robot"]; obj=env.scene["object"]; org=t(env.scene.env_origins); dev=env.device
A=env.action_manager.total_action_dim; term=env.action_manager.get_term("arm_action"); sc=t(term._scale)[0] if torch.is_tensor(term._scale) else torch.full((6,),float(term._scale),device=dev)
names=[f"follower_left_joint_{i}" for i in range(6)]+["follower_left_left_carriage_joint","follower_left_right_carriage_joint"]
jid=robot.find_joints(names,preserve_order=True)[0]
def fm(name):
    f=env.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
env.reset(); act=torch.zeros(N,A,device=dev); act[:,-2:]=-1.0
target={3:1.44,5:-2.95}; samples=[]
for k in range(args.ramp+20):
    a=min(k/args.ramp,1.0)
    for j,v in target.items(): act[:,j]=(a*v)/sc[j]
    env.step(act)
    held=(fm("pad_left_handle")>0.01)&(fm("pad_right_handle")>0.01)
    q=t(robot.data.joint_pos)[:,jid]; p=t(obj.data.root_pos_w)-org; r=t(obj.data.root_quat_w); u=1-2*(r[:,0]**2+r[:,1]**2)
    if k%args.every==0 and int(held.sum())>=N//2:
        qm=q[held].median(0).values; pm=p[held].median(0).values; rm=r[held].median(0).values; rm=rm/rm.norm()
        samples.append({"step":k,"held":int(held.sum()),"up_cos":float(u[held].median()),"pose":{n:round(float(v),4) for n,v in zip(names,qm)},"object_pose":[round(float(v),4) for v in torch.cat([pm,rm])]})
        print(f"[rot] k={k:3d} held {int(held.sum())}/{N} up_cos med {float(u[held].median()):+.2f} j3 {float(qm[3]):+.2f} j5 {float(qm[5]):+.2f} mug z {float(pm[2]):.3f}", flush=True)
out=args.out or os.path.join(os.path.dirname(pkg.__file__),"rotpath.json")
json.dump(samples,open(out,"w"),indent=1); print(f"[rot] wrote {len(samples)} samples -> {out}", flush=True)
env.close()
