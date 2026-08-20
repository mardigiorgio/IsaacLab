#!/bin/bash
# ICF on the Trossen mug scene.
# ICF_CONTACT_STIFFNESS is a global linear contact stiffness [N/m]; MuJoCo's
# compliance is mass-normalized instead, so the two cannot be equal by
# construction and the value must be calibrated against a reference contact.
# The 5200 default is NOT calibrated. Override: ICF_CONTACT_STIFFNESS=500 ./run_icf.sh
cd /home/mdigiorgio/Documents/code/IsaacLab || exit 1
export VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv
# The Newton viewer falls back to EGL offscreen when DISPLAY is unset, so no
# window opens. :2 is the active seat0 graphical session on this machine.
export DISPLAY="${DISPLAY:-:2}"
export ICF_CONTACT_STIFFNESS="${ICF_CONTACT_STIFFNESS:-5200}"
# Per-world contact budget. ICF drops contacts past it instead of failing, so
# it must exceed the scene's peak; raise until the "over max_rigid_contact"
# warnings stop.
export ICF_MAX_RIGID_CONTACT="${ICF_MAX_RIGID_CONTACT:-512}"
NAME="${RUN_NAME:-icf-mug-k${ICF_CONTACT_STIFFNESS}}"
echo "ICF_CONTACT_STIFFNESS=$ICF_CONTACT_STIFFNESS  ICF_MAX_RIGID_CONTACT=$ICF_MAX_RIGID_CONTACT  run_name=$NAME"
exec ./isaaclab.sh train --rl_library rsl_rl --task IsaacContrib-Lift-Spatula-Trossen-v0 --num_envs 1024 --seed 42 --max_iterations 1000 --logger wandb --log_project_name rubato-trossen --run_name "$NAME" --run_group icf-check --viz newton --video --video_length 200 --video_interval 600 physics=newton --solver icf
