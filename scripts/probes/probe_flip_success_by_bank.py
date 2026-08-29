"""Training start mix, deterministic policy: first-episode success per bank owner (unbanked = true home)."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--ckpt",required=True); parser.add_argument("--hv_min",type=float,default=0.4); parser.add_argument("--hv_max",type=float,default=0.7); parser.add_argument("--num_envs",type=int,default=256); parser.add_argument("--only",default="mix"); parser.add_argument("--rc",type=int,default=400_000); parser.add_argument("--tp",type=int,default=8_000_000)
AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import os, torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs
from rsl_rl.runners import OnPolicyRunner
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=args.num_envs
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=args.rc; cfg.sim.physics.collision_cfg.max_triangle_pairs=args.tp
cfg.events.reset_arm_hover_bank.params["alpha_min"]=0.0; cfg.events.reset_arm_home_via_bank.params["alpha_min"]=args.hv_min; cfg.events.reset_arm_home_via_bank.params["alpha_max"]=args.hv_max; cfg.curriculum=None
if args.only != "mix":
    for e in ("reset_arm_grasp_bank","reset_arm_lift_bank","reset_arm_rotate_bank","reset_arm_hover_bank","reset_arm_home_via_bank"): getattr(cfg.events,e).params["bank_fraction"]=(1.0 if e==args.only else 0.0)
agent_cfg=load_cfg_from_registry(TASK,"rsl_rl_cfg_entry_point")
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; wenv=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner=OnPolicyRunner(wenv, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device); runner.load(args.ckpt); policy=runner.get_inference_policy(device=u.device)
names={"reset_arm_grasp_bank":"grasp","reset_arm_lift_bank":"lift","reset_arm_rotate_bank":"rotate","reset_arm_hover_bank":"hover","reset_arm_home_via_bank":"home_via"}
keymap={id(u.event_manager.get_term_cfg(n).params["pose"]):s for n,s in names.items()}
res=wenv.get_observations(); obs=res[0] if isinstance(res,tuple) else res
owner=["home"]*N
for k,v in u._bank_selected.items():
    for i in torch.nonzero(v).squeeze(1).tolist(): owner[i]=keymap[k]
dev=u.device; succ=torch.zeros(N,dtype=torch.bool,device=dev); ended=torch.zeros(N,dtype=torch.bool,device=dev); prev_len=u.episode_length_buf.clone()
with torch.inference_mode():
    for k in range(240):
        if k in (0,10,20,30):
            L=u.scene.sensors["pad_left_handle"].data.force_matrix_w; R_=u.scene.sensors["pad_right_handle"].data.force_matrix_w
            f=lambda F: torch.zeros(N,device=dev) if F is None else torch.linalg.vector_norm((F.torch if hasattr(F,"torch") else F).sum(dim=2),dim=-1).nan_to_num(0.0).max(dim=-1).values
            held=((f(L)>0.01)&(f(R_)>0.01)).float().mean()
            norm=getattr(runner.alg.actor,"obs_normalizer",None) or getattr(runner.alg.actor,"actor_obs_normalizer",None)
            cnt=None if norm is None else (getattr(norm,"count",None) if not torch.is_tensor(getattr(norm,"count",None)) else float(norm.count))
            print(f"[bybank] k={k} held={held*100:.0f}% actor.training={runner.alg.actor.training} norm_count={cnt}", flush=True)
        if k in (0,1,2,60,120,121,122,180) and False:
            off=u.action_manager.get_term("arm_action")._offset[0]; goff=u.action_manager.get_term("gripper_action")._offset[0]
            q=u.scene["robot"].data.joint_pos.torch[0,:8]
            print(f"[bybank] k={k} eplen0={int(u.episode_length_buf[0])} arm_offset0={[round(float(x),3) for x in off]} grip_off0={[round(float(x),3) for x in goff]} obs_offset0={[round(float(x),2) for x in (obs["policy"] if hasattr(obs,"keys") else obs)[0,-6:]]} q0={[round(float(x),3) for x in q]} act0={[round(float(x),2) for x in policy(obs)[0]]}", flush=True)
        act=policy(obs)
        if os.environ.get("TRACE0") and k<14:
            _q=u.scene["robot"].data.joint_pos.torch[0,:8]; _o=u.scene["object"].data.root_pos_w.torch[0]-(u.scene.env_origins.torch if hasattr(u.scene.env_origins,"torch") else u.scene.env_origins)[0]
            print(f"[trace0] k={k} q={[round(float(x),3) for x in _q]} obj={[round(float(x),3) for x in _o]} act={[round(float(x),2) for x in act[0]]}", flush=True)
        out=wenv.step(act); obs=out[0]
        succ|=(u._fsm['success'] & ~ended)
        just=(u.episode_length_buf<prev_len)
        if just.any() and (k<40 or k%40==0):
            terms={n:int(u.termination_manager.get_term(n)[just].sum()) for n in u.termination_manager.active_terms}
            print(f"[bybank] step {k}: {int(just.sum())} envs reset; terms {{k:v for k,v in terms.items() if v}}; succ_flag {int(u._fsm['success'][just].sum())}", flush=True)
        ended|=just; prev_len=u.episode_length_buf.clone()
import collections
cnt=collections.Counter(owner); ok=collections.Counter(o for o,s in zip(owner,succ.tolist()) if s)
ended_first=int((ended).sum()); print(f"[bybank] only={args.only} envs={N} envs whose first episode ended within 240 steps: {ended_first}")
for o in ("home","home_via","hover","grasp","lift","rotate"): print(f"[bybank] {o:9s} n={cnt[o]:4d} first-episode success {100*ok[o]/max(cnt[o],1):5.1f}%")
env.close()
