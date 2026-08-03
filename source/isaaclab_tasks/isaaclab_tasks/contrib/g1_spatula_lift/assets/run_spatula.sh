#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# One entrypoint for the G1 spatula pick-up task. Run from anywhere:
#
#   ./source/isaaclab_tasks/isaaclab_tasks/contrib/g1_spatula_lift/assets/run_spatula.sh
#   ./…/run_spatula.sh --num_envs 2048       # any extra args pass through to training
#   ./…/run_spatula.sh --check               # verify the claw cage at reset, then exit
#   ./…/run_spatula.sh --play <checkpoint>   # watch a trained policy
#
# Builds the spatula USD first if it is missing. The USD is NOT in git
# (.gitignore forbids USD in this repo and .gitattributes routes it through
# LFS), so a fresh machine downloads the public LBM release once (~823 MB,
# cached afterwards) and converts it.
set -euo pipefail

ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# six levels: assets -> g1_spatula_lift -> contrib -> isaaclab_tasks (inner)
# -> isaaclab_tasks (outer) -> source -> repo root
REPO_ROOT="$(cd "${ASSETS_DIR}/../../../../../.." && pwd)"
USD_PATH="${ASSETS_DIR}/usd/thimma_wood_natural_flat_spatula.usd"
TASK="IsaacContrib-Lift-Spatula-G1-v0"

MODE="train"
PLAY_CKPT=""
PASS_ARGS=()
WANT_VIDEO=0
HAS_INTERVAL=0
HAS_LENGTH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE="check"; shift ;;
        --play)  MODE="play"; PLAY_CKPT="${2:-}"; shift 2 ;;
        --video) WANT_VIDEO=1; PASS_ARGS+=("$1"); shift ;;
        --video_interval|--video_interval=*) HAS_INTERVAL=1; PASS_ARGS+=("$1"); shift ;;
        --video_length|--video_length=*)     HAS_LENGTH=1;   PASS_ARGS+=("$1"); shift ;;
        *)       PASS_ARGS+=("$1"); shift ;;
    esac
done

# --video alone records ONE clip at step 0 and the next at step 2000, which at
# 24 steps/env is iteration 83 — long enough that it reads as "no video".
# Default to every 480 steps (20 iterations) unless the caller says otherwise.
if [[ "${WANT_VIDEO}" == "1" ]]; then
    [[ "${HAS_INTERVAL}" == "1" ]] || PASS_ARGS+=(--video_interval 480)
    [[ "${HAS_LENGTH}"   == "1" ]] || PASS_ARGS+=(--video_length 200)
fi

cd "${REPO_ROOT}"

if [[ ! -f "${USD_PATH}" ]]; then
    echo "[run_spatula] spatula USD missing — building it."
    echo "[run_spatula] first run downloads the LBM models wheel (~823 MB, cached afterwards)."
    ./isaaclab.sh -p "${ASSETS_DIR}/fetch_lbm_spatula.py"
    ./isaaclab.sh -p "${ASSETS_DIR}/convert_assets.py"
fi

case "${MODE}" in
    check)
        echo "[run_spatula] === claw cage at reset ==="
        echo "[run_spatula] expect: thumb local-y NEGATIVE, index and middle POSITIVE,"
        echo "[run_spatula]         all tips 1-4 cm above the 0.83 m tabletop."
        ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/contrib/test_g1_spatula_lift.py \
            -k newton_mjwarp -q
        ;;
    play)
        if [[ -z "${PLAY_CKPT}" ]]; then
            echo "[run_spatula] --play needs a checkpoint path" >&2
            exit 2
        fi
        ./isaaclab.sh play --rl_library rsl_rl --task "${TASK}" --num_envs 16 \
            --checkpoint "${PLAY_CKPT}" presets=newton_mjwarp \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}"
        ;;
    train)
        echo "[run_spatula] === training ${TASK} ==="
        echo "[run_spatula] watch for the staircase: reach -> lift -> track."
        echo "[run_spatula] decide at ~200 and ~400 iterations; no single termination should dominate."
        if [[ "${WANT_VIDEO}" == "1" ]]; then
            echo "[run_spatula] videos -> logs/rsl_rl/g1_spatula_lift/<timestamp>/videos/train/rl-video-step-N.mp4"
            echo "[run_spatula] (also uploaded to W&B project g1-spatula-lift)"
        fi
        # NOTE: `-p scripts/.../train_rsl_rl.py` looks right but is NOT — that
        # module has no __main__ guard, so it exits 0 having done nothing.
        # `isaaclab.sh train` is the supported entrypoint.
        ./isaaclab.sh train --rl_library rsl_rl --task "${TASK}" \
            --headless presets=newton_mjwarp \
            "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}"
        ;;
esac
