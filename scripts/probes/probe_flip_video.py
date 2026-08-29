"""Record a video of the deterministic policy from TRUE HOME starts (full 8-s episode, no training truncation)."""
import argparse, os
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ckpt",required=True); parser.add_argument("--num_envs",type=int,default=4); parser.add_argument("--length",type=int,default=250); parser.add_argument("--out",default="")
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True; args.enable_cameras=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
from rsl_rl.runners import OnPolicyRunner
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=args.num_envs
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=N*2000; cfg.sim.physics.collision_cfg.max_triangle_pairs=N*40000; cfg.curriculum=None
if hasattr(cfg.terminations,"no_pinch"): cfg.terminations.no_pinch=None
for e in [n for n in vars(cfg.events) if n.startswith("reset_arm_") and n.endswith("_bank")]: getattr(cfg.events,e).params["bank_fraction"]=0.0
cfg.viewer.eye=(0.9,0.9,0.7); cfg.viewer.lookat=(0.0,0.0,0.15)
agent_cfg=load_cfg_from_registry(TASK,"rsl_rl_cfg_entry_point")
out=args.out or os.path.join(os.path.dirname(args.ckpt), "videos_home"); os.makedirs(out,exist_ok=True)
# the repo's own recorder (what --video --viz newton uses in training): visualizer source, newton viewer headless
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg
cfg.video_recorders=[VideoRecorderCfg(video_interval=0, video_length=args.length, output_dir=out, output_filename_prefix=os.path.basename(args.ckpt).replace(".pt","")+"_home")]
cfg.sim.default_visualizer_cfg=NewtonVisualizerCfg(headless=True)
env=gym.make(TASK,cfg=cfg)
u=env.unwrapped; wenv=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner=OnPolicyRunner(wenv, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device); runner.load(args.ckpt); policy=runner.get_inference_policy(device=u.device)
res=wenv.get_observations(); obs=res[0] if isinstance(res,tuple) else res
succ=torch.zeros(N,dtype=torch.bool,device=u.device)
with torch.inference_mode():
    for k in range(args.length+5):
        act=policy(obs); out_=wenv.step(act); obs=out_[0]; succ|=u._fsm["success"]
print(f"[video] success within the clip: {int(succ.sum())}/{N}; files: {sorted(os.listdir(out))}", flush=True)
env.close()
