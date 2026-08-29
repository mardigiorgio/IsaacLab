"""Failure breakdown from TRUE HOME (deterministic, first episode): per env, did it pinch, reach ROTATED, succeed;
if not, what happened (never pinched / pinched then lost the handle before rotating / rotated then dropped / mug fell)."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ckpt",required=True); parser.add_argument("--num_envs",type=int,default=256)
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
from rsl_rl.runners import OnPolicyRunner
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=args.num_envs
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=N*800; cfg.sim.physics.collision_cfg.max_triangle_pairs=N*16000; cfg.curriculum=None
if hasattr(cfg.terminations,"no_pinch"): cfg.terminations.no_pinch=None  # evaluation: full 8-s episode, no training-only truncation
if hasattr(cfg.terminations,"no_pinch"): cfg.terminations.no_pinch=None  # evaluation: full 8-s episode, no training-only truncation
for e in [n for n in vars(cfg.events) if n.startswith("reset_arm_") and n.endswith("_bank")]: getattr(cfg.events,e).params["bank_fraction"]=0.0
agent_cfg=load_cfg_from_registry(TASK,"rsl_rl_cfg_entry_point")
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; wenv=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner=OnPolicyRunner(wenv, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device); runner.load(args.ckpt); policy=runner.get_inference_policy(device=u.device)
dev=u.device; t=lambda x: x.torch if hasattr(x,"torch") else x
def fm(name):
    f=u.scene.sensors[name].data.force_matrix_w
    return torch.zeros(N,device=dev) if f is None else torch.linalg.vector_norm(t(f).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
res=wenv.get_observations(); obs=res[0] if isinstance(res,tuple) else res
Z=lambda: torch.zeros(N,dtype=torch.bool,device=dev)
succ,pinched,rotated,fell,ended=Z(),Z(),Z(),Z(),Z(); maxstage=torch.zeros(N,dtype=torch.long,device=dev); bodyF=torch.zeros(N,device=dev); t_pinch=torch.full((N,),-1,device=dev); t_succ=torch.full((N,),-1,device=dev); prev=u.episode_length_buf.clone()
with torch.inference_mode():
    for k in range(240):
        act=policy(obs); out=wenv.step(act); obs=out[0]; fsm=u._fsm; live=~ended
        held=(fm("pad_left_handle")>0.01)&(fm("pad_right_handle")>0.01)
        newp=live&held&~pinched; t_pinch[newp]=k; pinched|=live&held
        rotated|=live&(fsm["stage"]>=3); maxstage=torch.where(live,torch.maximum(maxstage,fsm["stage"]),maxstage)
        news=live&fsm["success"]&~succ; t_succ[news]=k; succ|=live&fsm["success"]
        z=t(u.scene["object"].data.root_pos_w)[:,2]-t(u.scene.env_origins)[:,2]; fell|=live&(z<0.08)&~held
        bodyF=torch.where(live,torch.maximum(bodyF,fm("pad_body_contact")),bodyF)
        ended|=(u.episode_length_buf<prev); prev=u.episode_length_buf.clone()
fail=~succ
cats={"never pinched":fail&~pinched,"pinched, never ROTATED (lost before/during flip)":fail&pinched&~rotated,"ROTATED but no 30-frame hold":fail&rotated,"mug fell (any category)":fail&fell,"body force > 5 N (any category)":fail&(bodyF>5)}
print(f"[fail] {args.ckpt.split('/')[-2]}/{args.ckpt.split('/')[-1]}: success {succ.float().mean()*100:.1f}% of {N}; pinch step med {t_pinch[pinched].float().median():.0f}; success step med {t_succ[succ].float().median():.0f}", flush=True)
for n,m in cats.items(): print(f"[fail]   {n:55s} {int(m.sum()):4d} = {100*int(m.sum())/max(int(fail.sum()),1):5.1f}% of failures", flush=True)
env.close()
