#!/usr/bin/env bash
# g1-walking-policy: full sim-to-real pipeline for the 29-DoF
# Isaac-Velocity-Flat-G1-v1 task.
#
# Per solver — fixed (--solver mujoco) first, then adaptive (mujoco-adaptive):
#   1. teacher   : Isaac-Velocity-Flat-G1-v1        (1500 it, 4096 envs)
#   2. distill   : Velocity-G1-Distillation-v1      (1500 it, 4096 envs; loads teacher ckpt)
#   3. bridge    : distilled student -> PPO init ckpt (rsl-rl cannot load a
#                  distillation ckpt directly into PPO; a 1-iter template run
#                  provides the checkpoint skeleton)
#   4. finetune  : Velocity-G1-Student-Finetune-v1  (4000 it, 4096 envs; resumes bridged ckpt)
#   5. export    : play -> exported/policy.pt + policy.onnx  (the sim-to-real files)
#                  copied to $EXP_DIR/exports/g1v1-<label>/
#
# W&B: project g1-walking-policy, runs g1v1-<stage>-<label>-s<seed>-<tag>, with
# video (~1.6% duty). Rerunning skips finished stages (status/) and resumes
# interrupted ones from the newest checkpoint, capped to the stock budget.
# Waits for any running train_g1_walking.sh baseline to release the GPU
# (SKIP_WAIT=1 bypasses). ALWAYS launch detached:
#   nohup bash IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29_dofs/scripts/train_g1_v1_pipeline.sh > runlogs/g1_v1_pipeline.out 2>&1 &   (from ~/Documents/code)
set -uo pipefail

CODE_DIR=${CODE_DIR:-$HOME/Documents/code}
RUBATO_DIR=${RUBATO_DIR:-$CODE_DIR/isaac-rubato}
ISAACLAB_DIR=${ISAACLAB_DIR:-$CODE_DIR/IsaacLab}

PROJECT=${PROJECT:-g1-walking-policy}
RUN_TAG=${RUN_TAG:-r1}
SEED=${SEED:-42}
NUM_ENVS=${NUM_ENVS:-4096}
EXP_DIR=${EXP_DIR:-$RUBATO_DIR/experiments/g1-walking-policy}

TEACHER_TASK="Isaac-Velocity-Flat-G1-v1"
DISTILL_TASK="Velocity-G1-Distillation-v1"
FINETUNE_TASK="Velocity-G1-Student-Finetune-v1"
TEACHER_ITERS=1500
DISTILL_ITERS=1500
FINETUNE_ITERS=4000
RUN_ROOT="logs/rsl_rl/g1_29_dofs_flat"

LABELS=(fixed adaptive)
declare -A SOLVER_OF=([fixed]=mujoco [adaptive]=mujoco-adaptive)

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "isaaclab.sh not found at $ISAACLAB_DIR"
[[ -f "$RUBATO_DIR/.venv/bin/activate" ]] || die "rubato venv not found at $RUBATO_DIR/.venv"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[WARN] nvidia-smi not found"

if env | grep -q '^NEWTON_ADAPTIVE_'; then
  echo "[WARN] NEWTON_ADAPTIVE_* knobs are set and WILL affect the adaptive runs:"
  env | grep '^NEWTON_ADAPTIVE_' | sed 's/^/    /'
else
  echo "[INFO] no NEWTON_ADAPTIVE_* env knobs set (stock adaptive behavior)"
fi

# shellcheck disable=SC1091
source "$RUBATO_DIR/.venv/bin/activate" || die "venv activation failed"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[WARN] no W&B credentials found; runs will stall at login."
fi

# Wait for a running train_g1_walking.sh (if any) to release the GPU.
if [[ "${SKIP_WAIT:-0}" != 1 ]] && pgrep -f "train_g1_walking.sh" >/dev/null; then
  echo "[WAIT] train_g1_walking.sh baseline is running; waiting for it to finish ($(date +%F_%T))"
  while pgrep -f "train_g1_walking.sh" >/dev/null; do sleep 60; done
  echo "[WAIT] baseline finished; starting pipeline ($(date +%F_%T))"
fi

mkdir -p "$EXP_DIR"/{status,joblogs,exports}
cd "$EXP_DIR" || die "cannot cd to $EXP_DIR"
summary="$EXP_DIR/summary_v1.tsv"
[[ -f "$summary" ]] || printf "run\ttask\tsolver\tseed\trc\tminutes\n" > "$summary"

newest_run_dir() { ls -dt "$RUN_ROOT"/*_"$1" 2>/dev/null | head -n1; }
newest_ckpt() { ls -v "$1"/model_*.pt 2>/dev/null | tail -n1; }
ckpt_iter() { local b; b=$(basename "$1"); echo "${b//[^0-9]/}"; }

# run_stage <run_name> <task> <total_iters> <load_mode> [load_dir] [load_ckpt]
#   load_mode: none | teacher (distill: load w/o --resume) | resume
# Auto-resume: if the stage has its own run dir with checkpoints, resume from the
# newest one with the remaining budget and rejoin the same W&B run if identifiable.
run_stage() {
  local run_name=$1 task=$2 total_iters=$3 load_mode=$4 load_dir=${5:-} load_ckpt=${6:-}
  local status_f="status/${run_name}.exit"
  local own_dir own_ckpt it remaining wdir t0 rc mins
  local extra_args=() wandb_env=()

  if [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
    echo "[SKIP] $run_name (already done)"; return 0
  fi

  own_dir=$(newest_run_dir "$run_name")
  own_ckpt=""
  [[ -n "$own_dir" ]] && own_ckpt=$(newest_ckpt "$own_dir")
  if [[ -n "$own_ckpt" && "${FRESH:-0}" != 1 ]]; then
    it=$(ckpt_iter "$own_ckpt")
    if [[ "$it" -ge $((total_iters - 1)) ]]; then
      echo "[DONE] $run_name (checkpoint $it >= $((total_iters - 1)); marking complete)"
      echo 0 > "$status_f"; return 0
    fi
    remaining=$((total_iters - it))
    extra_args=( --resume --load_run "$(basename "$own_dir")"
                 --checkpoint "$(basename "$own_ckpt")" --max_iterations "$remaining" )
    for wdir in $(ls -dt "$own_dir"/wandb/run-* wandb/run-* 2>/dev/null); do
      if grep -qsF -- "$run_name" "$wdir/files/wandb-metadata.json"; then
        wandb_env=( WANDB_RESUME=allow "WANDB_RUN_ID=${wdir##*-}" ); break
      fi
    done
    echo "[RES ] $run_name <- $(basename "$own_dir")/$(basename "$own_ckpt") (iter $it/$total_iters)"
  else
    case "$load_mode" in
      teacher) extra_args=( --load_run "$load_dir" --checkpoint "$load_ckpt" ) ;;
      resume)  extra_args=( --resume --load_run "$load_dir" --checkpoint "$load_ckpt"
                            --max_iterations "$total_iters" ) ;;
    esac
  fi

  echo "[RUN ] $(date +%F_%T) $run_name (task=$task solver=$solver)"
  t0=$(date +%s)
  echo "==== $(date +%F_%T) launch $run_name ====" >> "joblogs/${run_name}.log"
  env "${wandb_env[@]}" "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl \
      --task "$task" --solver "$solver" --seed "$SEED" --num_envs "$NUM_ENVS" \
      --video --video_length 100 --video_interval 6400 \
      --logger wandb --log_project_name "$PROJECT" \
      --run_name "$run_name" --run_group "g1v1-${label}" \
      "${extra_args[@]}" physics=newton_mjwarp >> "joblogs/${run_name}.log" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  echo "$rc" > "$status_f"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$task" "$solver" "$SEED" "$rc" "$mins" >> "$summary"
  if [[ $rc == 0 ]]; then
    echo "[PASS] $run_name (${mins}m)"
  else
    echo "[FAIL] $run_name rc=$rc (${mins}m) -- tail of joblogs/${run_name}.log:"
    tail -n 15 "joblogs/${run_name}.log" | sed 's/^/    /'
  fi
  return $rc
}

pipeline_one() {
  local label=$1
  solver=${SOLVER_OF[$label]}
  local sfx="${label}-s${SEED}-${RUN_TAG}"
  local teacher_name="g1v1-teacher-$sfx" distill_name="g1v1-distill-$sfx"
  local finetune_name="g1v1-finetune-$sfx" export_name="g1v1-export-$sfx"
  local tdir tckpt ddir dckpt fdir fckpt rc

  # 1. teacher
  run_stage "$teacher_name" "$TEACHER_TASK" "$TEACHER_ITERS" none || return 1
  tdir=$(newest_run_dir "$teacher_name"); tckpt=$(newest_ckpt "$tdir")
  [[ -n "$tckpt" ]] || { echo "[FAIL] $label: no teacher checkpoint found"; return 1; }

  # 2. distillation (loads teacher; rsl-rl's distillation load takes the actor
  #    weights as the frozen teacher)
  run_stage "$distill_name" "$DISTILL_TASK" "$DISTILL_ITERS" teacher \
    "$(basename "$tdir")" "$(basename "$tckpt")" || return 1
  ddir=$(newest_run_dir "$distill_name"); dckpt=$(newest_ckpt "$ddir")
  [[ -n "$dckpt" ]] || { echo "[FAIL] $label: no distillation checkpoint found"; return 1; }

  # 3. bridge: template finetune ckpt skeleton + transplant the distilled student
  if [[ ! -f "status/${finetune_name}.bridged" ]]; then
    fdir=$(newest_run_dir "$finetune_name")
    if [[ -z "$fdir" || -z "$(newest_ckpt "$fdir")" ]]; then
      echo "[RUN ] $(date +%F_%T) $finetune_name (template for bridge)"
      "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl \
        --task "$FINETUNE_TASK" --solver "$solver" --seed "$SEED" --num_envs 256 \
        --max_iterations 1 --run_name "$finetune_name" physics=newton_mjwarp \
        >> "joblogs/${finetune_name}.log" 2>&1 \
        || { echo "[FAIL] $label: finetune template run failed"; return 1; }
      fdir=$(newest_run_dir "$finetune_name")
    fi
    fckpt=$(newest_ckpt "$fdir")
    [[ -n "$fckpt" ]] || { echo "[FAIL] $label: no template checkpoint"; return 1; }
    python - "$fckpt" "$dckpt" <<'PYEOF' || { echo "[FAIL] $label: bridge failed"; return 1; }
import sys, torch
tpl_path, distill_path = sys.argv[1], sys.argv[2]
t = torch.load(tpl_path, map_location="cpu", weights_only=False)
d = torch.load(distill_path, map_location="cpu", weights_only=False)
ta, ds = t["actor_state_dict"], d["student_state_dict"]
mm = [k for k in ta if k not in ds or ta[k].shape != ds[k].shape]
ex = [k for k in ds if k not in ta]
assert not mm and not ex, f"architecture mismatch: {mm[:3]} extra={ex[:3]}"
t["actor_state_dict"] = ds
t["iter"] = 0
torch.save(t, tpl_path)
print(f"[BRIDGE] student -> actor: {len(ta)} tensors transplanted into {tpl_path}")
PYEOF
    touch "status/${finetune_name}.bridged"
  fi

  # 4. student finetune (resumes the bridged checkpoint)
  fdir=$(newest_run_dir "$finetune_name"); fckpt=$(newest_ckpt "$fdir")
  run_stage "$finetune_name" "$FINETUNE_TASK" "$FINETUNE_ITERS" resume \
    "$(basename "$fdir")" "$(basename "$fckpt")" || return 1
  fdir=$(newest_run_dir "$finetune_name"); fckpt=$(newest_ckpt "$fdir")

  # 5. export: play writes exported/policy.pt + policy.onnx next to the ckpt.
  #    play keeps simulating after the export, so cap it with timeout.
  if [[ ! -f "status/${export_name}.exit" || "$(cat "status/${export_name}.exit")" != 0 ]]; then
    echo "[RUN ] $(date +%F_%T) $export_name"
    # play accepts neither --solver (train-only flag) nor a bare checkpoint
    # filename (--checkpoint is a full file path there)
    timeout 600 "$ISAACLAB_DIR/isaaclab.sh" play --rl_library rsl_rl \
      --task "$FINETUNE_TASK" --num_envs 32 \
      --checkpoint "$PWD/$fckpt" \
      --video --video_length 200 physics=newton_mjwarp \
      >> "joblogs/${export_name}.log" 2>&1
    if [[ -f "$fdir/exported/policy.pt" ]]; then
      mkdir -p "exports/g1v1-${label}"
      cp "$fdir/exported/policy.pt" "exports/g1v1-${label}/policy.pt"
      # the .onnx.data external-weights file must travel with the .onnx
      cp "$fdir/exported/policy.onnx" "$fdir/exported/policy.onnx.data" "exports/g1v1-${label}/" 2>/dev/null
      cp -r "$fdir/videos" "exports/g1v1-${label}/videos" 2>/dev/null
      echo 0 > "status/${export_name}.exit"
      echo "[PASS] $export_name -> $EXP_DIR/exports/g1v1-${label}/policy.pt"
    else
      echo 1 > "status/${export_name}.exit"
      echo "[FAIL] $export_name: no exported/policy.pt -- tail of joblogs/${export_name}.log:"
      tail -n 15 "joblogs/${export_name}.log" | sed 's/^/    /'
      return 1
    fi
  else
    echo "[SKIP] $export_name (already done)"
  fi
}

overall=0
for label in "${LABELS[@]}"; do
  echo
  echo "==== pipeline: $label (solver=${SOLVER_OF[$label]}) ===="
  pipeline_one "$label" || { overall=1; echo "[HALT] $label pipeline stopped at a failed stage"; }
done

echo
echo "==== g1-v1 sim-to-real pipeline done (rc=$overall) ===="
column -t -s $'\t' "$summary"
ls -l exports/g1v1-*/ 2>/dev/null
exit $overall
