"""Roll a checkpoint's deterministic policy from LIFTED-bank starts: up_cos, j3/j5, hold and FSM stage over time."""
import argparse, sys
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ckpt",required=True); parser.add_argument("--bank",default="reset_arm_lift_bank"); parser.add_argument("--stochastic",action="store_true"); parser.add_argument("--alpha_min",type=float,default=1.0); parser.add_argument("--alpha_max",type=float,default=1.0)
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
from isaaclab_tasks.contrib.trossen_mug_flip.mdp import _tip_handle_distance
from isaaclab.managers import SceneEntityCfg
from rsl_rl.runners import OnPolicyRunner
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=64
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=200_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=4_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank","reset_arm_hover_bank"): getattr(cfg.events,e).params["bank_fraction"]=0.0  # bank none = TRUE home starts only
(getattr(cfg.events,args.bank).params.__setitem__("bank_fraction",1.0) if args.bank != "none" else None); (getattr(cfg.events,args.bank).params.__setitem__("alpha_min",args.alpha_min) if args.bank != "none" else None); (getattr(cfg.events,args.bank).params.__setitem__("alpha_max",args.alpha_max) if args.bank != "none" else None); cfg.curriculum=None
agent_cfg=load_cfg_from_registry(TASK,"rsl_rl_cfg_entry_point")
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; wenv=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner=OnPolicyRunner(wenv, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device); runner.load(args.ckpt); policy=runner.get_inference_policy(device=u.device)
if args.stochastic:
    actor=runner.alg.actor
    policy=lambda o: actor.act(o) if hasattr(actor,'act') else actor(o)
t=lambda x: x.torch if hasattr(x,"torch") else x
robot=u.scene["robot"]; obj=u.scene["object"]; org=t(u.scene.env_origins); dev=u.device
aid=robot.find_joints("follower_left_joint_[0-5]")[0]
def fm(name):
    f=u.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
def upcos():
    q=t(obj.data.root_quat_w); return 1-2*(q[:,0]**2+q[:,1]**2)
res=wenv.get_observations(); obs=res[0] if isinstance(res,tuple) else res
succ=torch.zeros(N,dtype=torch.bool,device=dev); everheld=torch.zeros(N,dtype=torch.bool,device=dev); rows=[]
with torch.inference_mode():
    for k in range(240 if args.bank == 'none' else 120):
        act=policy(obs); out=wenv.step(act); obs=out[0]; succ|=u._fsm['success']; everheld|=((fm('pad_left_handle')>0.01)&(fm('pad_right_handle')>0.01))
        if k%10==0 or k in (119,239):
            L,R,B=fm("pad_left_handle"),fm("pad_right_handle"),fm("pad_body_contact"); h=(L>0.01)&(R>0.01); q=t(robot.data.joint_pos)[:,aid]; a=act
            wc=SceneEntityCfg("robot", body_names=["follower_left_link_6"]); wc.resolve(u.scene); tip_d=_tip_handle_distance(u,SceneEntityCfg("object"),SceneEntityCfg("ee_frame"),wc)
            rows.append(f"[roll] t={k:3d} tip-handle med {tip_d.median()*100:4.1f} cm  held {h.float().mean()*100:3.0f}%  up_cos med {upcos().median():+.2f}  mug z med {(t(obj.data.root_pos_w)[:,2]-org[:,2]).median():.3f}  j3 {q[:,3].median():+.2f} j5 {q[:,5].median():+.2f}  act j3 {a[:,3].median():+.2f} j5 {a[:,5].median():+.2f} grip {a[:,6].median():+.2f}  body F med {B.median():4.1f} N  stage med {float(u._fsm['stage'].float().median()):.1f} max {int(u._fsm['stage'].max())}")
print("\n".join(rows), flush=True); print(f"[roll] SUCCESS (in-hand, FSM) within the rollout: {succ.float().mean()*100:.0f}% of {N} envs from bank={args.bank}; ever pinched (both pads on handle) {everheld.float().mean()*100:.0f}%  stochastic={args.stochastic}", flush=True)
env.close()
