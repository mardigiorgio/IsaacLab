#!/usr/bin/env bash
# THE COMPARISON: fixed-step vs adaptive on the spatula-lift scene, from scratch.
#
# Scene and solver authoring are the upstream core/lift (Franka pick-up) stack,
# adopted verbatim: pyramidal cone, impratio 1, njmax 300 / nconmax 200,
# ls_iterations 15, update_data_interval 2, default materials and shapes,
# 120 Hz outer step / decimation 4 / 2 substeps (fixed tiers at mj dt ~4.2 ms).
#
# Two contact modes x two solvers, serial on one GPU, 2048 envs, 2000
# iterations, native W&B logging into rubato-trossen. Within each mode the
# ONLY varied factor is the solver (adaptive flag + substep count — the fixed
# arm subdivides the outer boundary uniformly, the adaptive solver chooses dt
# inside the same boundary):
#
#   MODE A — Newton collision pipeline, injected contacts (upstream default):
#     1. fixed-injected-4k
#     2. adaptive-injected-4k
#   MODE B — MuJoCo native contacts (internal convexification):
#     3. fixed-hulled-4k
#     4. adaptive-hulled-4k
#
# From scratch, deliberately: resumed checkpoints carry the learned habits of
# the scene they were trained in.
#
# Usage (single line, from anywhere):
#   ./source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_spatula_lift/run_comparison.sh
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$HOME/Documents/code/IsaacLabRubato/.venv}"

COMMON=(--rl_library rsl_rl --task IsaacContrib-Lift-Spatula-Trossen-v0 --seed 42
  --num_envs 4096 --max_iterations 2000
  --logger wandb --log_project_name rubato-trossen
  --video --video_length 250 --video_interval 2000 --viz newton)

echo "=== 1/4 fixed-hulled-4k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco \
  --run_name fixed-hulled-4k \
  env.sim.physics.solver_cfg.use_mujoco_contacts=True env.sim.physics.collision_cfg=null
echo "exit $? at $(date +%T)"

echo "=== 2/4 adaptive-hulled-4k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco-adaptive \
  --run_name adaptive-hulled-4k physics=newton_mjwarp_adaptive \
  env.sim.physics.solver_cfg.use_mujoco_contacts=True env.sim.physics.collision_cfg=null
echo "exit $? at $(date +%T)"
echo "=== 3/4 fixed-injected-4k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco \
  --run_name fixed-injected-4k
echo "exit $? at $(date +%T)"

echo "=== 4/4 adaptive-injected-4k $(date +%T)"
timeout 36000 ./isaaclab.sh train "${COMMON[@]}" --solver mujoco-adaptive \
  --run_name adaptive-injected-4k physics=newton_mjwarp_adaptive
echo "exit $? at $(date +%T)"

echo "=== ALL DONE $(date +%T)"
