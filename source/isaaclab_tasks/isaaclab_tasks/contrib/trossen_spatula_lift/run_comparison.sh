#!/usr/bin/env bash
# THE COMPARISON: fixed-step vs adaptive on the spatula-lift scene, from scratch.
#
# Three serial arms (one GPU), 2048 envs, 2000 iterations each, native W&B
# logging into rubato-trossen with periodic video clips:
#   1. fixed1p25ms-hulled-2k    fixed mj dt 1.25 ms, MuJoCo native contacts
#                               (convex hulls), 8 substeps
#   2. fixed1p25ms-exactmesh-2k fixed mj dt 1.25 ms, Newton exact-mesh
#                               injected contacts, 8 substeps
#   3. adaptive-2k              adaptive solver, 1 substep: dt chosen inside
#                               the 0.01 boundary by the error controller
#
# From scratch, deliberately: resumed checkpoints carry the learned habits of
# the scene they were trained in. Fixed dt is 1.25 ms because the authored
# compliant contact converts to a 3 ms timeconst and refsafe floors every
# pair at 2*dt: coarser fixed steps cannot express the authored contact.
#
# Usage (single line, from anywhere):
#   ./source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_spatula_lift/run_comparison.sh
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$HOME/Documents/code/IsaacLabRubato/.venv}"

COMMON=(--rl_library rsl_rl --task IsaacContrib-Lift-Spatula-Trossen-v0 --seed 42
  --num_envs 2048 --max_iterations 2000
  --logger wandb --log_project_name rubato-trossen
  --video --video_length 250 --video_interval 2000 --viz newton)

echo "=== 1/3 fixed1p25ms-hulled-2k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco \
  --run_name fixed1p25ms-hulled-2k \
  env.sim.physics.solver_cfg.use_mujoco_contacts=True env.sim.physics.collision_cfg=null \
  env.sim.physics.num_substeps=8
echo "exit $? at $(date +%T)"

echo "=== 2/3 fixed1p25ms-exactmesh-2k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco \
  --run_name fixed1p25ms-exactmesh-2k \
  env.sim.physics.num_substeps=8
echo "exit $? at $(date +%T)"

echo "=== 3/3 adaptive-2k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco-adaptive \
  --run_name adaptive-2k physics=newton_mjwarp_adaptive
echo "exit $? at $(date +%T)"
echo "=== ALL DONE $(date +%T)"
