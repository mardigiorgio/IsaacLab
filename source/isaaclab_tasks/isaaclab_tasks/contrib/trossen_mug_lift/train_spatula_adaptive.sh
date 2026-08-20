#!/usr/bin/env bash
# Adaptive-solver arm of the spatula experiment (full 0.01 boundary, solver picks dt).
# Usage: ./train_spatula_adaptive.sh [seed] [run-suffix]
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SEED="${1:-42}"
SUFFIX="${2:-c1}"
exec ./isaaclab.sh train --rl_library rsl_rl \
  --task IsaacContrib-Lift-Spatula-Trossen-v0 --seed "$SEED" --solver mujoco-adaptive \
  --video --video_length 200 --video_interval 4800 --viz newton \
  --logger wandb --log_project_name rubato-trossen \
  --run_name "spatula-adaptive-s${SEED}-${SUFFIX}" --run_group spatula-adaptive \
  physics=newton_mjwarp_adaptive
