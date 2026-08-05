# lib_g1_pipeline.sh — shared machinery for the G1 sim-to-real pipelines.
# Sourced by scripts/train_g1_dr.sh and scripts/train_g1_adaptive.sh; not runnable alone.
# Callers set RUN_TAG and OTHER_SCRIPTS before sourcing, then call pipeline_one.
set -uo pipefail

CODE_DIR=${CODE_DIR:-$HOME/Documents/code}
RUBATO_DIR=${RUBATO_DIR:-$CODE_DIR/IsaacLabRubato}
ISAACLAB_DIR=${ISAACLAB_DIR:-$CODE_DIR/IsaacLab}

PROJECT=${PROJECT:-g1-walking-policy}
RUN_TAG=${RUN_TAG:-r1}
SEED=${SEED:-42}
NUM_ENVS=${NUM_ENVS:-4096}
EXP_DIR=${EXP_DIR:-$RUBATO_DIR/experiments/g1-walking-policy}

DISTILL_ITERS=${DISTILL_ITERS:-2000}
FINETUNE_ITERS=${FINETUNE_ITERS:-6000}
RUN_ROOT=${RUN_ROOT:-"logs/rsl_rl/g1_29_dofs_flat"}

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "isaaclab.sh not found at $ISAACLAB_DIR"
[[ -f "$RUBATO_DIR/.venv/bin/activate" ]] || die "rubato venv not found at $RUBATO_DIR/.venv"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[WARN] nvidia-smi not found"

# shellcheck disable=SC1091
source "$RUBATO_DIR/.venv/bin/activate" || die "venv activation failed"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[WARN] no W&B credentials found; runs will stall at login."
fi

OTHER_SCRIPTS=${OTHER_SCRIPTS:-"train_g1_v1_pipeline.sh"}
if [[ "${SKIP_WAIT:-0}" != 1 ]] && pgrep -f "$OTHER_SCRIPTS" >/dev/null; then
  echo "[WAIT] another pipeline is running; waiting ($(date +%F_%T))"
  while pgrep -f "$OTHER_SCRIPTS" >/dev/null; do sleep 60; done
fi

mkdir -p "$EXP_DIR"/{status,joblogs,exports}
cd "$EXP_DIR" || die "cannot cd to $EXP_DIR"
summary="$EXP_DIR/summary_dr_vs_adaptive.tsv"
[[ -f "$summary" ]] || printf "run\ttask\tsolver\tseed\trc\tminutes\n" > "$summary"

newest_run_dir() { ls -dt "$RUN_ROOT"/*_"$1" 2>/dev/null | head -n1; }
newest_ckpt() { ls -v "$1"/model_*.pt 2>/dev/null | tail -n1; }
ckpt_iter() { local b; b=$(basename "$1"); echo "${b//[^0-9]/}"; }

# Per-pipeline globals set by pipeline_one: solver, label, stage_env (array)
run_stage() {
  local run_name=$1 task=$2 total_iters=$3 load_mode=$4 load_dir=${5:-} load_ckpt=${6:-}
  local status_f="status/${run_name}.exit"
  local own_dir own_ckpt it remaining wdir t0 rc mins
  local extra_args=() wandb_env=("${stage_env[@]}")

  if [[ -f "$status_f" && "$(cat "$status_f")" == 0 ]]; then
    echo "[SKIP] $run_name (already done)"; return 0
  fi

  # Fresh-by-default: an interrupted stage RESTARTS from scratch unless
  # RESUME=1 is set. Blind checkpoint-resume kept continuing policies trained
  # on since-changed (or broken) env configs.
  own_dir=$(newest_run_dir "$run_name")
  own_ckpt=""
  [[ -n "$own_dir" ]] && own_ckpt=$(newest_ckpt "$own_dir")
  if [[ -n "$own_ckpt" && "${RESUME:-0}" != 1 ]]; then
    echo "[NOTE] $run_name has a previous unfinished run ($(basename "$own_dir")); starting FRESH (set RESUME=1 to continue it)"
    own_ckpt=""
  fi
  if [[ -n "$own_ckpt" ]]; then
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
        wandb_env+=( WANDB_RESUME=allow "WANDB_RUN_ID=${wdir##*-}" ); break
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
      --run_name "$run_name" --run_group "${prefix}-${label}" \
      "${extra_args[@]}" "physics=${phys_preset}" >> "joblogs/${run_name}.log" 2>&1
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

# pipeline_one <prefix> <label> <solver> <teacher_task> <distill_task> <finetune_task> <teacher_iters> <nan_guard> <phys_preset>
pipeline_one() {
  prefix=$1; label=$2; solver=$3
  local teacher_task=$4 distill_task=$5 finetune_task=$6 teacher_iters=$7 nan_guard=$8
  phys_preset=${9:-newton_mjwarp}
  local sfx="${label}-s${SEED}-${RUN_TAG}"
  local teacher_name="${prefix}-teacher-$sfx" distill_name="${prefix}-distill-$sfx"
  local finetune_name="${prefix}-finetune-$sfx" export_name="${prefix}-export-$sfx"
  local tdir tckpt ddir dckpt fdir fckpt

  stage_env=()
  [[ "$nan_guard" == 1 ]] && stage_env=( NEWTON_ADAPTIVE_NAN_GUARD=1 )

  # 1. teacher
  run_stage "$teacher_name" "$teacher_task" "$teacher_iters" none || return 1
  tdir=$(newest_run_dir "$teacher_name"); tckpt=$(newest_ckpt "$tdir")
  [[ -n "$tckpt" ]] || { echo "[FAIL] $label: no teacher checkpoint found"; return 1; }

  # 2. distillation (loads the teacher's actor as the frozen teacher)
  run_stage "$distill_name" "$distill_task" "$DISTILL_ITERS" teacher \
    "$(basename "$tdir")" "$(basename "$tckpt")" || return 1
  ddir=$(newest_run_dir "$distill_name"); dckpt=$(newest_ckpt "$ddir")
  [[ -n "$dckpt" ]] || { echo "[FAIL] $label: no distillation checkpoint found"; return 1; }

  # 3. bridge: template finetune ckpt skeleton + transplant the distilled student
  if [[ ! -f "status/${finetune_name}.bridged" ]]; then
    fdir=$(newest_run_dir "$finetune_name")
    if [[ -z "$fdir" || -z "$(newest_ckpt "$fdir")" ]]; then
      echo "[RUN ] $(date +%F_%T) $finetune_name (template for bridge)"
      env "${stage_env[@]}" "$ISAACLAB_DIR/isaaclab.sh" train --rl_library rsl_rl \
        --task "$finetune_task" --solver "$solver" --seed "$SEED" --num_envs 256 \
        --max_iterations 1 --run_name "$finetune_name" "physics=${phys_preset}" \
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
  run_stage "$finetune_name" "$finetune_task" "$FINETUNE_ITERS" resume \
    "$(basename "$fdir")" "$(basename "$fckpt")" || return 1
  fdir=$(newest_run_dir "$finetune_name"); fckpt=$(newest_ckpt "$fdir")

  # 5. export: play writes exported/policy.pt (+ .onnx/.onnx.data) next to the ckpt.
  #    play accepts neither --solver nor a bare ckpt filename; it keeps simulating
  #    after the export, so cap with timeout.
  if [[ ! -f "status/${export_name}.exit" || "$(cat "status/${export_name}.exit")" != 0 ]]; then
    echo "[RUN ] $(date +%F_%T) $export_name"
    env "${stage_env[@]}" timeout 600 "$ISAACLAB_DIR/isaaclab.sh" play --rl_library rsl_rl \
      --task "$finetune_task" --num_envs 32 \
      --checkpoint "$PWD/$fckpt" \
      --video --video_length 200 "physics=${phys_preset}" \
      >> "joblogs/${export_name}.log" 2>&1
    if [[ -f "$fdir/exported/policy.pt" ]]; then
      mkdir -p "exports/${prefix}-${label}"
      cp "$fdir/exported/policy.pt" "exports/${prefix}-${label}/policy.pt"
      cp "$fdir/exported/policy.onnx" "$fdir/exported/policy.onnx.data" "exports/${prefix}-${label}/" 2>/dev/null
      cp -r "$fdir/videos" "exports/${prefix}-${label}/videos" 2>/dev/null
      echo 0 > "status/${export_name}.exit"
      echo "[PASS] $export_name -> $EXP_DIR/exports/${prefix}-${label}/policy.pt"
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
