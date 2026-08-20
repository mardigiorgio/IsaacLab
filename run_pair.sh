#!/bin/bash
# Paired ICF-vs-MuJoCo PPO run: same scene, same reward, same seed, same
# preset, same dt/decimation. --solver is the ONLY variable.
#
# ICF_CONTACT_STIFFNESS is the one coefficient that cannot be made identical
# by construction: MuJoCo's contact compliance is mass-normalized
# (K_eff = k*imp^2*m_eff/(1-imp)) while ICF's is a flat k [N/m]. It is matched
# by measurement at the mug-on-table reference condition; every other pair
# then differs by its mass ratio, which is a stated limitation, not a bug.
cd /home/mdigiorgio/Documents/code/IsaacLab || exit 1
export VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv
# The Newton viewer falls back to EGL offscreen when DISPLAY is unset, so no
# window opens. :2 is the active seat0 graphical session on this machine.
export DISPLAY="${DISPLAY:-:2}"
SEED="${SEED:-42}"
ENVS="${ENVS:-1024}"
ITERS="${ITERS:-1000}"
GROUP="${GROUP:-pair-s${SEED}}"
export ICF_CONTACT_STIFFNESS="${ICF_CONTACT_STIFFNESS:-5200}"
export ICF_MAX_RIGID_CONTACT="${ICF_MAX_RIGID_CONTACT:-512}"
COMMON="--rl_library rsl_rl --task IsaacContrib-Lift-Spatula-Trossen-v0 --num_envs $ENVS --seed $SEED --max_iterations $ITERS --logger wandb --log_project_name rubato-trossen --run_group $GROUP --viz newton --video --video_length 200 --video_interval 600 physics=newton"
echo "=== ARM 1/2: MuJoCo fixed  (seed $SEED, $ENVS envs, group $GROUP) ==="
./isaaclab.sh train $COMMON --run_name "mujoco-s${SEED}" --solver mujoco
echo "=== ARM 2/2: ICF fixed  k=$ICF_CONTACT_STIFFNESS  (seed $SEED, group $GROUP) ==="
./isaaclab.sh train $COMMON --run_name "icf-k${ICF_CONTACT_STIFFNESS}-s${SEED}" --solver icf
