#!/usr/bin/env bash
# Fixed-solver arm of the spatula experiment (2 substeps, mj dt 0.005).
# Usage: ./train_spatula_fixed.sh [seed] [run-suffix]
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SEED="${1:-42}"
SUFFIX="${2:-c1}"
exec ./isaaclab.sh train --rl_library rsl_rl \
  --task IsaacContrib-Lift-Spatula-Trossen-v0 --seed "$SEED" --solver mujoco \
  --video --video_length 200 --video_interval 4800 --viz newton \
  --logger wandb --log_project_name rubato-trossen \
  --run_name "spatula-mujoco-sub2-s${SEED}-${SUFFIX}" --run_group spatula-mujoco \
  physics=newton_mjwarp
