#!/bin/bash
# The CENIC campaign, encoded: per task, in order —
#   K1, K2, K3, adaptive, then K3wall (fixed K3 rerun at the SAME WALL
#   CLOCK the task's adaptive run actually consumed).
# Tasks in Marco's order: slide, lift, plate, flip.
#
# NOT LAUNCHED BY ANYTHING AUTOMATICALLY. Preflight, in order, before
# invoking:
#   1. scripts/probes/probe_campaign_coefficients.py exits 0 (invariants),
#   2. Marco's mug-lift scene pass is merged (mug convexified),
#   3. flip settle + reward-smoke gates have passed,
#   4. GOAL_SPEED in the slide cfg is Marco-confirmed,
#   5. no other trainer owns the GPU.
#
# Invariants encoded here, asserted by the probe: seed 42, 2048 envs, no
# DR, 30 Hz control at every rung, single W&B project, per-task contact
# caps (mug scenes 1024, plate 8192 — plate-in-rack peak demand measured
# 3624 contacts/env).
set -u
cd "$(dirname "$0")/../.."
SP=${CAMPAIGN_LOG_DIR:-/tmp/trossen_campaign}
mkdir -p "$SP"
export VIRTUAL_ENV="$HOME/Documents/code/icra2027/.venv"
export TROSSEN_RAILS=1
Q=$SP/campaign.log

# The K ladder is the set of time steps a practitioner would plausibly
# guess: n uniform physics steps per 1/30 s control boundary, n = 1..5.
# The paper's claim is that NO guessed rung trains the hard tasks and the
# adaptive arm does, at the same control boundary, untuned.
K1="env.sim.dt=0.03333333333333333 env.decimation=1 env.sim.render_interval=1"
K2="env.sim.dt=0.016666666666666666 env.decimation=2 env.sim.render_interval=2"
K3="env.sim.dt=0.011111111111111112 env.decimation=3 env.sim.render_interval=3"
K4="env.sim.dt=0.008333333333333333 env.decimation=4 env.sim.render_interval=4"
K5="env.sim.dt=0.006666666666666667 env.decimation=5 env.sim.render_interval=5"
# Every task runs the SAME rungs (Marco, 2026-08-30). Flip and tree are
# authored at 1/450 x 15, so every rung overrides the step explicitly; the
# adaptive arm's boundary is sim.dt x num_substeps, so flip/tree adaptive is
# pinned to the family's 1/90 x 3 boundary like everyone else's.
declare -A ADAPT_OVERRIDES
ADAPT_OVERRIDES[flip]="env.sim.dt=0.011111111111111112 env.decimation=3 env.sim.render_interval=3"
ADAPT_OVERRIDES[tree]="env.sim.dt=0.011111111111111112 env.decimation=3 env.sim.render_interval=3"

epoch() { date -d "2026-$1 $2" +%s; }
wall() { # wall <run-name> -> seconds from last START to DONE
  local s d
  s=$(grep "START $1 " "$Q" | tail -1 | sed -E 's/^\[cq ([0-9-]+) ([0-9:]+)\].*/\1 \2/')
  d=$(grep "DONE  $1 " "$Q" | tail -1 | sed -E 's/^\[cq ([0-9-]+) ([0-9:]+)\].*/\1 \2/')
  [ -n "$s" ] && [ -n "$d" ] || return 1
  echo $(( $(epoch $d) - $(epoch $s) ))
}

run() { # run <gym-task> <solver> <name> <log> <iters> [K overrides]
  local task=$1 solver=$2 name=$3 log=$4 iters=$5; shift 5
  local tsk=$(echo "$name" | grep -oE 'slide|lift|plate|flip|tree')
  local k=$(echo "$name" | grep -oE 'K[0-9]wall|K[0-9]' | head -1); k=${k:-adaptive}
  local sol=icf-fixed; case "$name" in *adaptive*) sol=icf-adaptive;; esac
  export WANDB_TAGS="$tsk,$sol,$k,campaign"
  case "$tsk" in plate) export ICF_MAX_RIGID_CONTACT=8192;; *) export ICF_MAX_RIGID_CONTACT=1024;; esac
  # Wait for the previous trainer's GPU teardown to drain (up to 5 min):
  # a fresh launch into a draining GPU dies at boot and burns retries.
  for g in $(seq 1 30); do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -eq 0 ] && break
    sleep 10
  done
  for attempt in 1 2 3 4 5; do
    echo "[cq $(date '+%m-%d %H:%M:%S')] START $name (try $attempt)" >> "$Q"
    ./isaaclab.sh -p scripts/reinforcement_learning/train.py --rl_library rsl_rl \
      --task "$task" --num_envs 2048 --seed 42 --max_iterations "$iters" \
      --solver "$solver" physics=newton --logger wandb \
      --log_project_name rubato-trossen \
      --run_name "$name" --video --video_length 200 --video_interval 300 \
      --viz newton "$@" > "$SP/$log" 2>&1 &
    local pid=$!
    local ok=0
    local idle=0
    for w in $(seq 1 120); do
      sleep 30
      kill -0 $pid 2>/dev/null || break
      grep -qE "Traceback|SyntaxError" "$SP/$log" && break
      grep -q "Learning iteration" "$SP/$log" && { ok=1; break; }
      cpu=$(ps -o %cpu= -p $(pgrep -P $pid -f train.py | head -1) 2>/dev/null | awk '{print int($1)}')
      if [ "${cpu:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
      [ $idle -ge 5 ] && break
    done
    if [ $ok -eq 1 ]; then
      wait $pid
      echo "[cq $(date '+%m-%d %H:%M:%S')] DONE  $name exit $?" >> "$Q"
      return 0
    fi
    grep -qE "Traceback|SyntaxError" "$SP/$log" && { echo "[cq] REAL ERROR in $log, skipping" >> "$Q"; return 1; }
    kill -9 $pid 2>/dev/null
    pkill -9 -f "run_name $name" 2>/dev/null
    sleep 5
    echo "[cq $(date '+%m-%d %H:%M:%S')] STALLED/CRASHED $name (try $attempt)" >> "$Q"
  done
  echo "[cq] GAVE UP on $name after 5 tries" >> "$Q"
  return 1
}

task_ladder() { # task_ladder <gym-task> <short-name> <iters>
  local gym=$1 t=$2 iters=$3
  run "$gym" icf          "icf-fixed-$t-K1-s42"    "cq_${t}_K1.log"       "$iters" $K1
  run "$gym" icf          "icf-fixed-$t-K2-s42"    "cq_${t}_K2.log"       "$iters" $K2
  run "$gym" icf          "icf-fixed-$t-K3-s42"    "cq_${t}_K3.log"       "$iters" $K3
  run "$gym" icf          "icf-fixed-$t-K4-s42"    "cq_${t}_K4.log"       "$iters" $K4
  run "$gym" icf          "icf-fixed-$t-K5-s42"    "cq_${t}_K5.log"       "$iters" $K5
  run "$gym" icf-adaptive "icf-adaptive-$t-s42"    "cq_${t}_adaptive.log" "$iters" ${ADAPT_OVERRIDES[$t]:-}
  # K5wall: the finest GUESSED step, given the same wall clock the adaptive
  # run actually consumed. Budget = (adaptive wall - K5 startup) / K5
  # per-iter, all measured from THIS campaign's own logs; startup = K5 wall
  # minus its summed iteration time.
  local aw k5w itersum count startup budget
  aw=$(wall "icf-adaptive-$t-s42") || { echo "[cq] SKIP $t K5wall: no adaptive wall" >> "$Q"; return; }
  k5w=$(wall "icf-fixed-$t-K5-s42") || { echo "[cq] SKIP $t K5wall: no K5 wall" >> "$Q"; return; }
  read itersum count <<< $(grep -E "Iteration time:" "$SP/cq_${t}_K5.log" | awk '{gsub(/s/,"",$3); s+=$3; n+=1} END{print s, n}')
  [ "${count:-0}" -gt 0 ] || { echo "[cq] SKIP $t K5wall: no iteration times" >> "$Q"; return; }
  startup=$(awk -v w=$k5w -v s=$itersum 'BEGIN{d=w-s; print (d>0)?int(d):0}')
  budget=$(awk -v aw=$aw -v st=$startup -v s=$itersum -v n=$count 'BEGIN{print int((aw-st)*n/s)}')
  echo "[cq] $t K5wall: adaptive_wall=${aw}s startup=${startup}s -> budget=$budget iters" >> "$Q"
  run "$gym" icf "icf-fixed-$t-K5wall-s42" "cq_${t}_K5wall.log" "$budget"
}

task_ladder IsaacContrib-Slide-Mug-Trossen-v0     slide 700
task_ladder IsaacContrib-Lift-Mug-Trossen-v0      lift  1000
task_ladder IsaacContrib-PlatePick-Trossen-v0     plate 1000
task_ladder IsaacContrib-Flip-Mug-Trossen-v0      flip  1000
task_ladder IsaacContrib-MugHang-Trossen-v0       tree  1000
echo "[cq $(date '+%m-%d %H:%M:%S')] CAMPAIGN COMPLETE" >> "$Q"
