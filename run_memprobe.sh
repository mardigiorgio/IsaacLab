#!/bin/bash
# Peak-GPU-memory probe: runs the real scene under ICF at the CORRECTED contact
# cap so the env-count ceiling is computed from the configuration that will
# actually train, not the one that silently dropped contacts.
cd /home/mdigiorgio/Documents/code/IsaacLab || exit 1
export VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv
export ICF_CONTACT_STIFFNESS="${ICF_CONTACT_STIFFNESS:-5200}"
export ICF_MAX_RIGID_CONTACT="${ICF_MAX_RIGID_CONTACT:-512}"
ENVS="${ENVS:-1024}"
OUT="${OUT:-/tmp/claude-1002/-home-mdigiorgio-Documents-code/940d92d8-77fb-4109-81da-ef22a1363f13/scratchpad/mem_${ENVS}.log}"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$OUT" &
SAMPLER=$!
./isaaclab.sh train --rl_library rsl_rl --task IsaacContrib-Lift-Spatula-Trossen-v0 --num_envs "$ENVS" --seed 42 --max_iterations 6 --viz none physics=newton --solver icf
kill $SAMPLER 2>/dev/null
echo "=== peak MiB at ENVS=$ENVS ==="; sort -n "$OUT" | tail -1
