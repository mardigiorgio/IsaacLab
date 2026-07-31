#!/usr/bin/env bash
# G1 sim-to-real pipeline: total-DR, FIXED solver (v2 hardware-informed DR set).
#
# DR levers: actuation latency (0-1 control steps) + first-order lag, IMU
# attitude/gyro bias, ankle-weighted action-rate penalty, ankle-specific
# actuator DR (±50% gains + joint friction/armature), slewed velocity commands
# with stop-from-backward practice (3-8 s resampling), vx ±1.2, plus COM/mass/
# friction/push/reset randomization. Obs/action contract unchanged.
#
# Stages: teacher 4000 it -> distillation 2000 -> bridge -> finetune 6000
# -> export to exports/g1v1dr-fixed/ (policy.pt + policy.onnx).
# Stage-resumable: rerun the same command to continue. RUN_TAG defaults to r2
# (v1 DR runs live under r1 and are complete).
#
# Launch (from ~/Documents/code):
#   nohup bash IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29_dofs/scripts/train_g1_dr.sh > runlogs/g1_dr.out 2>&1 &
set -uo pipefail
RUN_TAG=${RUN_TAG:-r3}
PROJECT=${PROJECT:-g1-walking-policy}
OTHER_SCRIPTS="train_g1_adaptive.sh|train_g1_v1_pipeline.sh"
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib_g1_pipeline.sh"

overall=0
echo; echo "==== pipeline: total-DR, fixed solver (tag $RUN_TAG) ===="
pipeline_one g1v1dr fixed mujoco \
  Isaac-Velocity-Flat-G1-DR-v1 Velocity-G1-Distillation-DR-v1 Velocity-G1-Student-Finetune-DR-v1 \
  4000 0 newton_mjwarp || { overall=1; echo "[HALT] pipeline stopped at a failed stage"; }

echo; echo "==== g1-dr pipeline done (rc=$overall) ===="
column -t -s $'\t' "$summary"
ls -l exports/g1v1dr-fixed/ 2>/dev/null
exit $overall
