#!/usr/bin/env bash
# G1 legs-only STANDING balance pipeline (fixed solver, full DR).
#
# The policy actuates only the 12 leg joints; the waist/arms are driven by a
# randomized upper-body motion emulator (pick-and-place stand-in) with 0-1.5 kg
# per-hand payload DR. Recovery steps allowed but penalized. Obs contract (59):
# gyro(3) gravity(3) leg qpos(12) leg qvel(12) upper qpos(17) prev action(12).
#
# Stages: teacher 3000 it -> distillation 2000 -> bridge -> finetune 4000
# -> export to exports/g1stand-fixed/ (policy.pt + policy.onnx).
# Stage-resumable: rerun the same command to continue.
#
# Launch (from ~/Documents/code):
#   nohup bash IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29_dofs/scripts/train_g1_standing.sh > runlogs/g1_standing.out 2>&1 &
set -uo pipefail
RUN_TAG=${RUN_TAG:-r2}
PROJECT=${PROJECT:-g1-standing-policy}
RUN_ROOT="logs/rsl_rl/g1_29_dofs_stand"
DISTILL_ITERS=${DISTILL_ITERS:-2000}
FINETUNE_ITERS=${FINETUNE_ITERS:-4000}
OTHER_SCRIPTS="train_g1_dr.sh|train_g1_adaptive.sh|train_g1_v1_pipeline.sh"
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib_g1_pipeline.sh"

overall=0
echo; echo "==== pipeline: standing balance, fixed solver (tag $RUN_TAG) ===="
pipeline_one g1stand fixed mujoco \
  Isaac-Stand-G1-DR-v1 Stand-G1-Distillation-DR-v1 Stand-G1-Student-Finetune-DR-v1 \
  3000 0 newton_mjwarp || { overall=1; echo "[HALT] pipeline stopped at a failed stage"; }

echo; echo "==== g1-standing pipeline done (rc=$overall) ===="
column -t -s $'\t' "$summary"
ls -l exports/g1stand-fixed/ 2>/dev/null
exit $overall
