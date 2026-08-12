#!/usr/bin/env bash
# THE TEST: one training iteration in the contact-heavy region. Minimize it.
#
# Resumes the frozen benchmark checkpoint (iteration 150, policy engaging the
# spatula: the contact-heavy regime) at 2048 envs, runs 8 iterations, prints
# each collection time and a single headline number. Fresh process each run, so
# this measures solver cost, NOT the long-process cliff (see LONG mode).
#
# Usage (single line):
#   ./source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_spatula_lift/bench_contact_iter.sh
# Knobs pass through the environment, e.g.:
#   NEWTON_ADAPTIVE_TAIL_COMPACT=1 ./source/.../bench_contact_iter.sh
# Extra hydra overrides append as arguments, e.g.:
#   ./.../bench_contact_iter.sh env.sim.physics.solver_cfg.ls_parallel=true
# LONG mode (cliff hunting: 300 iterations, watch ~iteration 87):
#   BENCH_LONG=1 ./.../bench_contact_iter.sh
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$HOME/Documents/code/IsaacLabRubato/.venv}"
ITERS=$(( ${BENCH_LONG:-0} == 1 ? 300 : 158 ))
LOG=$(mktemp /tmp/bench_contact_XXXX.log)
timeout 7200 ./isaaclab.sh train --rl_library rsl_rl \
  --task IsaacContrib-Lift-Spatula-Trossen-v0 --seed 42 --solver mujoco-adaptive \
  --num_envs 2048 --max_iterations $ITERS --resume \
  --load_run benchmark-contact-heavy --checkpoint model_150.pt \
  --logger wandb --log_project_name rubato-trossen --run_name bench-contact "$@" \
  physics=newton_mjwarp_adaptive > "$LOG" 2>&1
echo "--- per-iteration collection times:"
grep "Collection time" "$LOG" | awk '{print NR": "$3}'
grep "Collection time" "$LOG" | awk 'NR>3 {gsub("s","",$3); s+=$3; n+=1; if($3<m||m==0)m=$3} END {if(n>0) printf "\nCONTACT-HEAVY ITERATION: best %.2fs | mean(after warmup) %.2fs | n=%d\n", m, s/n, n; else print "NO ITERATIONS COMPLETED - see " l}' l="$LOG"
echo "full log: $LOG"
