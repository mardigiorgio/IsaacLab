"""Read back the bank-applied joint state at reset and diff vs the pose."""
from __future__ import annotations
import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
from isaaclab.app import add_launcher_args, launch_simulation
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]]

def main() -> int:
    import gymnasium as gym
    from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import GRASP_BANK_POSE
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice
    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 2
        apply_solver_choice(env_cfg, "icf")
        env_cfg.events.reset_arm_grasp_bank.params["bank_fraction"] = 1.0
        env_cfg.events.reset_arm_grasp_bank.params["alpha_min"] = 1.0
        env_cfg.events.reset_arm_grasp_bank.params["noise"] = 0.0
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        robot = env.unwrapped.scene["robot"]
        ids, names = robot.find_joints(list(GRASP_BANK_POSE.keys()), preserve_order=True)
        q = robot.data.joint_pos.torch[0, ids].cpu()
        print("joint                                   authored   realized   diff")
        for n, a, r in zip(names, [GRASP_BANK_POSE[k] for k in names], q.tolist()):
            print(f"{n:<38} {a:+9.4f} {r:+9.4f} {r-a:+9.4f}")
        env.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
