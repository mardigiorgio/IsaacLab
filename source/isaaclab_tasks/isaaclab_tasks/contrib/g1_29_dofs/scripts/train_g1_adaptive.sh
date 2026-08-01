#!/usr/bin/env bash
# G1 sim-to-real pipeline: zero-DR, ADAPTIVE solver (-Soft-v1 tasks: stock
# contact stiffness, required for --solver mujoco-adaptive).
#
# Stages: teacher 1500 it -> distillation 2000 -> bridge -> finetune 6000
# -> export to exports/g1v1-adaptive/ (policy.pt + policy.onnx).
# Stage-resumable: rerun the same command to continue (RUN_TAG r1 resumes the
# existing partial teacher checkpoint).
#
# Launch (from ~/Documents/code):
#   nohup bash IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29_dofs/scripts/train_g1_adaptive.sh > runlogs/g1_adaptive.out 2>&1 &
set -uo pipefail
RUN_TAG=${RUN_TAG:-r1}
OTHER_SCRIPTS="train_g1_dr.sh|train_g1_v1_pipeline.sh"
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib_g1_pipeline.sh"

overall=0
echo; echo "==== pipeline: zero-DR, adaptive solver (tag $RUN_TAG) ===="
pipeline_one g1v1 adaptive mujoco-adaptive \
  Isaac-Velocity-Flat-G1-Soft-v1 Velocity-G1-Distillation-Soft-v1 Velocity-G1-Student-Finetune-Soft-v1 \
  1500 1 newton_mjwarp || { overall=1; echo "[HALT] pipeline stopped at a failed stage"; }

echo; echo "==== g1-adaptive pipeline done (rc=$overall) ===="
column -t -s $'\t' "$summary"
ls -l exports/g1v1-adaptive/ 2>/dev/null
exit $overall
