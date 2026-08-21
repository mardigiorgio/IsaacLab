# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test camera rendering on the mug rig under the Newton backend.

The teacher->student pipeline needs pixel observations from the rig's own
camera mounts. This probe answers only the gating question: does a
TiledCamera attached to the rig's cam_high optical frame produce non-trivial
images on this stack at all? It attaches one 128x128 RGB camera, steps the
env, and reports image tensor statistics. Varied non-zero pixels = pass;
an exception or an all-constant image names the failure mode.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_camera_smoke.py --num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=5)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[], enable_cameras=True)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> int:
    import gymnasium as gym
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        apply_solver_choice(env_cfg, "icf")
        env_cfg.scene.cam_high = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/cam_high_link/cam_high_color_frame/cam_high_color_optical_frame/smoke_cam",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            # The default renderer_cfg resolves to the PhysX RTX renderer, which
            # is not installed on the Newton stack; the Warp ray-tracer is.
            renderer_cfg=NewtonWarpRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
            ),
        )

        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        cam = env.unwrapped.scene.sensors["cam_high"]
        zero = torch.zeros(
            args_cli.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device
        )
        with torch.inference_mode():
            for step in range(args_cli.steps):
                env.step(zero)
                img = cam.data.output["rgb"]
                img = img.torch if hasattr(img, "torch") else img
                flat = img.float()
                # Per-env pixel spread: a rendered scene varies; a dead buffer is constant.
                spread = (flat.amax(dim=(1, 2, 3)) - flat.amin(dim=(1, 2, 3)))
                print(
                    f"[step {step}] shape {tuple(img.shape)} dtype {img.dtype}"
                    f"  mean {flat.mean():8.3f}  min {flat.min():6.1f}  max {flat.max():6.1f}"
                    f"  per-env spread min {spread.min():6.1f}"
                )
        ok = bool((spread > 0).all())
        print(f"[verdict] {'PASS: all envs render varied pixels' if ok else 'FAIL: constant image buffer'}")
        # Keep one frame on disk: pixel statistics cannot show WHERE the camera
        # points; the saved image is the pointing check.
        out = os.environ.get("SMOKE_IMG_OUT", "")
        if out:
            import numpy as np
            from PIL import Image

            Image.fromarray(np.asarray(img[0].cpu(), dtype="uint8")).save(out)
            print(f"[image] env0 frame -> {out}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
