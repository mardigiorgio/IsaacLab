# 5090 validation protocol — conditional-graph boundary loop (option 2)

Goal: validate and measure the eager-kill stack on the production box (RTX 5090,
8192 envs). Everything is flag-gated and defaults OFF; a failed capture downgrades
automatically to the proven per-iteration tier, so no run can be broken by it.

## What was built (2026-07-01, "option 2")

1. **`MjwStepAllocCache`** (`newton/_src/solvers/mujoco/mjw_alloc_cache.py`):
   scoped shim that monkeypatches `wp.empty/zeros/full/clone` during each mjw step
   and returns cached per-call-site buffers — the step records with ZERO allocation
   nodes after warmup (CUDA forbids alloc nodes inside conditional body graphs).
   mjw code runs unmodified. Enable standalone: `NEWTON_MJW_ALLOC_CACHE=1`.
2. **Conditional-march tier** (`solver_mujoco_adaptive.py`): with
   `NEWTON_MJ_ADAPTIVE_CONDITIONAL=1` the whole boundary loop becomes ONE
   `wp.capture_while` conditional while-node (max_substeps enforced on-device):
   zero host syncs per boundary. Warmup = 2 boundaries on the per-iteration tier.
   Any capture failure prints a "downgrading permanently" warning and reverts to
   the per-iteration tier mid-run.
3. **Manager-level capture** (IsaacLab): with the same flag, the manager captures
   the full `decimation x (actuators + solver)` loop like the fixed-step path
   (plus a one-step eager warmup before the startup capture).

## Run matrix (from IsaacLab root; identical warmup/steps for comparability)

```bash
B="./isaaclab.sh -p scripts/benchmarks/bench_adaptive_allegro.py \
   --headless --num_envs 8192 --warmup 64 --steps 256"
export NEWTON_ADAPTIVE_LOG_EVERY=0

# R1: baseline (current default tiers)
$B --label 5090_baseline

# R2: alloc cache alone (must be ~= R1; sanity for the shim)
NEWTON_MJW_ALLOC_CACHE=1 $B --label 5090_alloccache

# R3: full conditional stack (solver while-node + manager graph)
NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 $B --label 5090_conditional

# R3b: conditional solver tier only (isolates manager-graph contribution)
NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 $B --label 5090_conditional_solver_only --no_manager_graph

# R4: fully eager (sizes the total overhead pie on Blackwell)
NEWTON_MJ_ADAPTIVE_GRAPH=0 $B --label 5090_eager

# R5: shared forward prefix (skips the duplicated kinematics/collision/mass/bias
#     pass for the first half eval -- cuts 1 of 3 prefix passes per iteration)
NEWTON_MJ_ADAPTIVE_SHARED_FWD=1 $B --label 5090_sharedfwd

# R6: full stack (shared prefix + conditional while-node + manager graph)
NEWTON_MJ_ADAPTIVE_SHARED_FWD=1 NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 $B --label 5090_full_stack
```

## Acceptance criteria (each run)

- Exit code 0; a `[bench] {...}` JSON line is printed and appended to
  `bench_results.jsonl` next to the script (untracked).
- `status_summary.error_max <= 1e-3` (tol) and
  `sim_time_min == sim_time_max == 0.008333` (all worlds land the boundary).
- `adaptive_iterations` within a few % across R1–R3b (controller math unchanged;
  differences are run-to-run trajectory divergence).
- R3/R3b logs: NO "downgrading permanently" warning (grep the log). R3 log should
  contain "Newton CUDA graph captured (standard Warp mode)".
- `solver_niter` field: if `max` sits at 100 (the cfg cap), worlds are not
  converging — raise it or investigate; if `max` << 100, the budget costs nothing
  (mjw early-exits) and needs no tuning.

## Reading the results

- R1 vs R4: the total launch+sync overhead pie on Blackwell.
- R1 vs R3b: what the conditional while-node buys (sync removal).
- R3b vs R3: what manager-level capture adds (actuator/decimation glue).
- R2 vs R1 must be ~0 (shim overhead sanity); if R2 is measurably slower, report.
- R1 vs R5: the shared forward prefix — expected to be the LARGEST single win
  (it removes real GPU work: ~1/3 of the kinematics/collision/mass passes per
  iteration). Iterations should match R1 tightly: the physics is identical, and
  both Richardson estimates now judge the same contact set. If iterations shift
  by more than a few % or error_max approaches tol, stop and report — that would
  mean a prefix quantity was NOT valid for reuse.
- R6 is the production candidate if R3 and R5 both validate.
- If R3 wins: flip the flag on for production runs (env var only, no code change).
  If it loses or downgrades: run stays correct on the default tier; capture the
  warning text — it names the exact operation CUDA rejected.

## Also worth one pass on the 5090 box

```bash
# unit tests (incl. shim contracts) + kernel-level controller checks
cd ~/Documents/code/newton-adaptive
uv run --extra dev --with pytest -m pytest newton/tests/test_mjw_alloc_cache.py \
  newton/tests/test_adaptive_*.py -q
uv run python scripts/verify_kernel_fixes.py
```
(`test_floor_nan_guard::test_floor_diverged_is_held_not_committed` is a known
pre-existing failure — it asserts the removed divergence latch.)

## Known limitation

Scenes that exceed ~250k candidate collision pairs select mjw's SAP broadphase,
which calls `wp.utils.array_scan` / `segmented_sort_pairs` — potential C++-internal
allocations invisible to the shim. The Allegro/Shadow in-hand scenes use NXN and
never hit that path; a future large-scene run that triggers a downgrade warning
should look there first.
