"""Lift-bank start: observation + first action at episode 1 (after full reset) vs episode 2 (per-env reset)."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ckpt",required=True); parser.add_argument("--bank",default="reset_arm_lift_bank")
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
from rsl_rl.runners import OnPolicyRunner
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=16
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=100_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=2_000_000
for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank","reset_arm_hover_bank","reset_arm_home_via_bank"): getattr(cfg.events,e).params["bank_fraction"]=(1.0 if e==args.bank else 0.0)
cfg.curriculum=None
agent_cfg=load_cfg_from_registry(TASK,"rsl_rl_cfg_entry_point")
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; wenv=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner=OnPolicyRunner(wenv, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device); runner.load(args.ckpt); policy=runner.get_inference_policy(device=u.device)
names=[n for n in u.observation_manager.active_terms["policy"]]; dims=u.observation_manager.group_obs_term_dim["policy"]
def show(tag, o):
    o=(o["policy"] if hasattr(o,"keys") else o)[0].tolist(); i=0; parts=[]
    for n,d in zip(names,dims):
        k=d[0] if isinstance(d,(tuple,list)) else int(d); parts.append(f"{n}={[round(x,2) for x in o[i:i+k]]}"); i+=k
    print(f"[obs] {tag}: "+" | ".join(parts), flush=True)
res=wenv.get_observations(); obs=res[0] if isinstance(res,tuple) else res
show("ep1 k=0", obs); print(f"[obs] ep1 act0={[round(float(x),2) for x in policy(obs)[0]]}", flush=True)
first=None
with torch.inference_mode():
    for k in range(250):
        act=policy(obs); out=wenv.step(act); obs=out[0]
        if first is None and int(u.episode_length_buf[0])==0:
            first=k; show(f"ep2 k={k+1} (env0 reset at step {k})", obs); print(f"[obs] ep2 act0={[round(float(x),2) for x in policy(obs)[0]]}", flush=True)
            terms={n:int(u.termination_manager.get_term(n)[0]) for n in u.termination_manager.active_terms}; print(f"[obs] env0 ended by {[n for n,v in terms.items() if v]}", flush=True)
            break
env.close()
