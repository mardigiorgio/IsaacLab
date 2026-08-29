"""Training-config reset: which bank owns each env, and where is the mug for home_via/hover-selected envs?"""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args,_=parser.parse_known_args(); args.headless=True
app=AppLauncher(args).app
import torch, gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.physics_presets import apply_solver_choice
TASK="IsaacContrib-Flip-Mug-Trossen-v0"; N=512
cfg=parse_env_cfg(TASK,num_envs=N); apply_solver_choice(cfg,"icf")
cfg.sim.physics.collision_cfg.rigid_contact_max=400_000; cfg.sim.physics.collision_cfg.max_triangle_pairs=8_000_000
cfg.events.reset_arm_hover_bank.params["alpha_min"]=0.0; cfg.curriculum=None
print("[ovl] event order:", [n for n in cfg.events.__dict__ if not n.startswith('_')])
env=gym.make(TASK,cfg=cfg); u=env.unwrapped; t=lambda x: x.torch if hasattr(x,"torch") else x
obj=u.scene["object"]; org=t(u.scene.env_origins)
names={"reset_arm_grasp_bank":"grasp","reset_arm_lift_bank":"lift","reset_arm_rotate_bank":"rotate","reset_arm_hover_bank":"hover"}
keymap={}
for n,short in names.items():
    p=u.event_manager.get_term_cfg(n).params["pose"]; keymap[id(p)]=short
tot={s:0 for s in names.values()}; air={s:0 for s in names.values()}; overlap={}
for rep in range(4):
    env.reset()
    for _ in range(3): u.step(torch.zeros(N,u.action_manager.total_action_dim,device=u.device))
    z=t(obj.data.root_pos_w)[:,2]-org[:,2]
    sel={keymap.get(k,str(k)):v.clone() for k,v in u._bank_selected.items()}
    for s,v in sel.items():
        tot[s]+=int(v.sum()); air[s]+=int((v&(z>0.15)).sum())
    for s in ("hover",):
        for o in ("lift","rotate","grasp"):
            overlap[(s,o)]=overlap.get((s,o),0)+int((sel[s]&sel[o]).sum())
print("[ovl] flags:", {k:list(keymap.values()).count(k) for k in keymap.values()})
for s in tot: print(f"[ovl] {s:9s} selected {tot[s]:5d}/{4*N}  mug airborne (z>0.15 after 3 steps) {air[s]:4d} = {100*air[s]/max(tot[s],1):.0f}%")
for (s,o),c in overlap.items(): print(f"[ovl] {s} & {o}: {c} = {100*c/max(tot[s],1):.0f}% of {s}")
env.close()
