#!/usr/bin/env bash
# g1-walking-policy: G1 flat-terrain walking with Newton (physics=newton_mjwarp),
# trained twice back-to-back: FIXED-step solver first (--solver mujoco), then
# ADAPTIVE (--solver mujoco-adaptive).
#
# Task: Isaac-Velocity-Flat-G1-v0 — the Newton-supported G1 flat velocity task.
# Contacts come from Newton's CollisionPipeline (use_mujoco_contacts=False in the
# task's newton_mjwarp preset); the foot colliders keep their raw collision meshes.
#
# W&B: project g1-walking-policy, runs g1-flat-{fixed,adaptive}-s<seed>-<tag>,
# videos via rsl_rl's built-in WandbSummaryWriter (--video). Recorder sized to
# ~1.6% rendering duty cycle (--video_length 100 --video_interval 6400).
#
# Re-running skips runs that already finished with exit 0 (status/ dir); an
# interrupted run auto-resumes from its newest model_*.pt, capped so total
# training still equals the stock iteration budget, and rejoins the same W&B run
# when the local wandb files identify it. RESUME=1 continues an interrupted stage from its newest checkpoint.
# ALWAYS launch detached so an SSH drop cannot kill it:
#   nohup bash IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29_dofs/scripts/train_g1_walking.sh > runlogs/g1_walking.out 2>&1 &   (from ~/Documents/code)
#
# Stock task config throughout (num_envs, iterations, PPO params untouched).
# Knobs (env vars): SEED, RUN_TAG, PROJECT, FRESH=1, WANDB_MODE=offline.
set -uo pipefail

CODE_DIR=${CODE_DIR:-$HOME/Documents/code}
RUBATO_DIR=${RUBATO_DIR:-$CODE_DIR/isaac-rubato}
ISAACLAB_DIR=${ISAACLAB_DIR:-$CODE_DIR/IsaacLab}

PROJECT=${PROJECT:-g1-walking-policy}
RUN_TAG=${RUN_TAG:-r1}
SEED=${SEED:-42}
TASK="Isaac-Velocity-Flat-G1-v0"
EXP_DIR=${EXP_DIR:-$RUBATO_DIR/experiments/g1-walking-policy}

# label -> solver flag value; trained in this order: fixed first, then adaptive.
LABELS=(fixed adaptive)
declare -A SOLVER_OF=([fixed]=mujoco [adaptive]=mujoco-adaptive)

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "isaaclab.sh not found at $ISAACLAB_DIR"
[[ -f "$RUBATO_DIR/.venv/bin/activate" ]] || die "rubato venv not found at $RUBATO_DIR/.venv"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[WARN] nvidia-smi not found"

# Deliberately no repo auto-update. NEWTON_ADAPTIVE_* env vars gate adaptive-solver
# behavior; report any that are set.
if env | grep -q '^NEWTON_ADAPTIVE_'; then
  echo "[WARN] NEWTON_ADAPTIVE_* knobs are set and WILL affect the adaptive run:"
  env | grep '^NEWTON_ADAPTIVE_' | sed 's/^/    /'
else
  echo "[INFO] no NEWTON_ADAPTIVE_* env knobs set (stock adaptive behavior)"
fi

# shellcheck disable=SC1091
source "$RUBATO_DIR/.venv/bin/activate" || die "venv activation failed"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[WARN] no W&B credentials found; runs will stall at login."
  echo "[WARN] either 'wandb login' first or rerun with WANDB_MODE=offline."
fi

mkdir -p "$EXP_DIR"/{status,joblogs}
cd "$EXP_DIR" || die "cannot cd to $EXP_DIR"
summary="$EXP_DIR/summary.tsv"
[[ -f "$summary" ]] || printf "run\ttask\tsolver\tseed\trc\tminutes\n" > "$summary"

n_pass=0 n_fail=0 n_skip=0

run_one() {
  local label=$1 solver=${SOLVER_OF[$1]}
  local run_name="g1-flat-${label}-s${SEED}-${RUN_TAG}"
  local status_f="status/${run_name}.exit"
  local prev_dir last_ckpt ckpt_it total_it wdir t0 rc mins
  local resume_args=() wandb_env=() cmd=()

  if [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
    echo "[SKIP] $run_name (already done)"; n_skip=$((n_skip+1)); return 0
  fi

  # Auto-resume an interrupted run: newest checkpoint, iteration budget capped to
  # the stock remainder, same W&B run when the local wandb files identify it.
  if [[ "${FRESH:-0}" != 1 ]]; then
    prev_dir=$(ls -dt logs/rsl_rl/*/*_"$run_name" 2>/dev/null | head -n1)
    last_ckpt=""
    [[ -n "$prev_dir" ]] && last_ckpt=$(ls -v "$prev_dir"/model_*.pt 2>/dev/null | tail -n1)
    if [[ -n "$last_ckpt" ]]; then
      ckpt_it=$(basename "$last_ckpt"); ckpt_it=${ckpt_it//[^0-9]/}
      total_it=$(grep -rho 'max_iterations: *[0-9]*' "$prev_dir/params" 2>/dev/null \
                 | head -n1 | tr -dc 0-9)
      if [[ -n "$total_it" && "$ckpt_it" -ge "$total_it" ]]; then
        echo "[DONE] $run_name (checkpoint $ckpt_it >= $total_it; marking complete)"
        echo 0 > "$status_f"; n_skip=$((n_skip+1)); return 0
      fi
      resume_args=( --resume --load_run "$(basename "$prev_dir")"
                    --checkpoint "$(basename "$last_ckpt")" )
      if [[ -n "$total_it" ]]; then
        resume_args+=( --max_iterations $(( total_it - ckpt_it )) )
      else
        echo "[WARN] $run_name: stock max_iterations not found under $prev_dir/params; resuming uncapped"
      fi
      for wdir in $(ls -dt "$prev_dir"/wandb/run-* wandb/run-* 2>/dev/null); do
        if grep -qsF -- "$run_name" "$wdir/files/wandb-metadata.json"; then
          wandb_env=( WANDB_RESUME=allow "WANDB_RUN_ID=${wdir##*-}" ); break
        fi
      done
      echo "[RES ] $run_name <- $(basename "$prev_dir")/$(basename "$last_ckpt") (iter $ckpt_it${total_it:+/$total_it})${wandb_env[1]:+, W&B run ${wandb_env[1]#WANDB_RUN_ID=}}"
    fi
  fi

  cmd=( "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl
        --task "$TASK" --solver "$solver" --seed "$SEED"
        --video --video_length 100 --video_interval 6400
        --logger wandb --log_project_name "$PROJECT"
        --run_name "$run_name" --run_group "g1-flat-${label}"
        "${resume_args[@]}"
        physics=newton_mjwarp )

  echo "[RUN ] $(date +%F_%T) $run_name (solver=$solver)"
  t0=$(date +%s)
  echo "==== $(date +%F_%T) launch $run_name ====" >> "joblogs/${run_name}.log"
  env "${wandb_env[@]}" "${cmd[@]}" >> "joblogs/${run_name}.log" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  echo "$rc" > "$status_f"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$TASK" "$solver" "$SEED" "$rc" "$mins" >> "$summary"
  if [[ $rc == 0 ]]; then
    echo "[PASS] $run_name (${mins}m)"; n_pass=$((n_pass+1))
  else
    echo "[FAIL] $run_name rc=$rc (${mins}m) -- tail of joblogs/${run_name}.log:"
    tail -n 15 "joblogs/${run_name}.log" | sed 's/^/    /'
    n_fail=$((n_fail+1))
  fi
}

for label in "${LABELS[@]}"; do
  run_one "$label"
done

echo
echo "==== g1-walking-policy done: $n_pass passed, $n_fail failed, $n_skip skipped ===="
column -t -s $'\t' "$summary"
