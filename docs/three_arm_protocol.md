# Three-arm protocol: fixed MuJoCo vs fixed ICF vs adaptive ICF

Task: `IsaacContrib-Lift-Spatula-Trossen-v0` (Trossen Stationary AI rig, LBM Inomata
mug, lift by the body). Launcher: `run_three_arm.sh`. Calibration probe:
`scripts/probes/probe_contact_compliance.py`.

| arm | `--solver` | solver |
|---|---|---|
| 1 | `mujoco` | `SolverMuJoCo`, fixed step |
| 2 | `icf` | `icf_warp.SolverICF`, fixed step |
| 3 | `icf-adaptive` | `icf_warp.SolverICFAdaptive`, CENIC per-world step doubling |

## 0. What this design can and cannot isolate

**Arm 2 vs arm 3 is the only single-variable contrast.** Same solver object graph,
same contact law, same `IcfParams` instance (built once above the fixed/adaptive
split in `NewtonMJWarpManager._create_solver`), same manager collision pipeline,
same contact set, same collide cadence. Only the stepping scheme differs.

**Arm 1 vs arm 2 is not a solver-only contrast.** It is a
solver + contact-model + friction-combination + joint-dissipation contrast, and
several of those differences cannot be removed by any choice of numbers
(section 3). Any writeup that presents arm 1 vs arm 2 as "the solver" is wrong.

**Provenance of everything below.** Statements marked *(measured)* were produced by
running the probe on this machine; statements marked *(read)* were read from the
working-tree source; statements marked *(unverified)* come from the survey that
preceded this document and were **not** re-checked. Nothing here is folklore with
a citation attached — where the check was not run, it says so.

## 1. Identical by construction

* Scene graph, USD assets, spawn poses, `env_spacing`, `num_envs`,
  `episode_length_s`, `decimation = 3`, `sim.dt = 1/90`, `num_substeps = 1`. *(read)*
* **Outer control boundary and collide cadence.** `NewtonManager._simulate_full`
  calls `CollisionPipeline.collide()` once per decimation iteration, then one
  solver boundary of `sim.dt`. All three arms: three boundaries of 11.111 ms per
  30 Hz control tick, one `collide()` each. *(read)*
* **The contact set.** `use_mujoco_contacts=False`, so all three arms consume the
  same `newton.Contacts` from the same `CollisionPipeline`: same broad phase,
  narrow phase, reduction, `gap = 0.01` generation threshold, `margin = 0`. *(read)*
* Joint armature, mug mass/COM/inertia, PD targets, action scale and clip,
  gripper open/close commands, observations, events, seeds, episode structure. *(read)*
* No per-contact hydroelastic stiffness on any arm, so ICF applies one global
  `contact_stiffness` to every contact and MuJoCo's per-contact override never
  fires. *(read)*

**Caveat that is easy to miss.** "Same contact set" means *the same buffer* — not
the same number of contacts at run time. Both arms collide the same way, but the
two arms' *states* diverge, so the generated set diverges with them. Measured on
the settle probe: object↔slab penetrating contacts were 4 on both arms, but total
pipeline contacts per world were 24 (MuJoCo) versus 56 (ICF) at the same nominal
rest pose. *(measured)* The reference pair matched; the scene as a whole did not.

## 2. Matched by calibration

### 2.1 Normal contact stiffness

MuJoCo's normal contact is acceleration-space and mass-normalized. From
`mujoco_warp/_src/constraint.py` *(read)*:

```
k    = 1 / (dmax^2 * timeconst^2 * dampratio^2)
D    = imp / ((1 - imp) * invweight)          invweight = invweight0[b1] + invweight0[b2]
aref = -k * imp * pos
=>  K_eff(delta) = imp(delta)^2 * k / ((1 - imp(delta)) * invweight)
```

ICF's is `fn = k * delta` with one global `k` in N/m *(read,
`icf_warp/patches.py`)*. There is no k that makes these equal; there is a k that
makes them equal at one operating point.

**Reference contact:** the mug at rest on `table_guard`, arm at its default pose,
zero action. **Observable:** the mug's root-frame drop `slab_top_z - root_z`,
averaged over the trailing 150 env steps, with a settled certificate
(max per-step `|dz|` over the trailing 60 steps `< 1e-6 m`).

**Measured, 2026-08-18, RTX 5090, 16 envs, 300 env steps** *(measured)*:

| quantity | fixed MuJoCo | fixed ICF (k = 289.2 N/m) |
|---|---|---|
| root drop | 2.62823e-4 m | 2.63418e-4 m |
| mean penetration over object↔slab contacts | 1.39346e-4 m | 1.43980e-4 m |
| penetrating object↔slab contacts / world | 4 | 4 |
| settle certificate (max per-step \|dz\|) | 3.73e-9 m | 2.61e-8 m |

Acceptance: **AC1 PASS** (root-drop mismatch 0.23%, `<= 5%` required; mean
penetration 3.3%), **AC2 PASS** (same contact count), **AC4 REPORTED-FAIL**
(section 3.1), **AC3 NOT RUN** (section 5).

Reproduce:

```
export VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv
./isaaclab.sh -p scripts/probes/probe_contact_compliance.py --arm mujoco --num_envs 16 --steps 300 --tail 150 --settle_win 60 --viz none --out /tmp/cal_mujoco.json
ICF_CONTACT_STIFFNESS=289.2 ICF_MAX_RIGID_CONTACT=512 ./isaaclab.sh -p scripts/probes/probe_contact_compliance.py --arm icf --num_envs 16 --steps 300 --tail 150 --settle_win 60 --viz none --out /tmp/cal_icf.json
./isaaclab.sh -p scripts/probes/probe_contact_compliance.py --verdict /tmp/cal_mujoco.json /tmp/cal_icf.json
```

**How k was found, and why it took more than one division.** The intended method is
one division: ICF's rest depth is exactly `W / (n*k)`, so
`k* = k0 * delta_ICF(k0) / delta_MJ`. That gave 318.5 N/m from `k0 = 5200`, and at
318.5 the deeper penetration pulled a **fifth** contact into the object↔slab set,
breaking the `n`-unchanged assumption the division rests on *(measured)*. The
converged value came instead from a two-point solve on the root drop,
`d(k) = d0 + C/k`, using two runs that reported the same `n`: `d0 = 1.111e-4 m`
(the fixed offset between the mug's body origin and its collision mesh's lowest
witness), `C = 0.0439 N`, hence `k = 289.2 N/m` *(measured)*. The probe prints both
methods. **This is a two-point solve, not a sweep and not a fit to a curve, but it
is also not the assumption-free division the method promised** — say so rather than
quoting the one-shot number.

Independent cross-check: on the ICF arm the probe's derived `W/(n*pen)` reproduced
the configured stiffness to five figures (5200.28 against 5200.0) *(measured)*,
which validates the census count, the penetration accumulator and the weight
together. MuJoCo's secant stiffness at the same depth was 318.6 N/m — the ICF k
that matches the *depth* is 289.2 rather than 318.6 because MuJoCo's contacts do
not share the load equally over four vertical normals.

### 2.2 Normal contact dissipation

MuJoCo's `dampratio = 1.0` is critical damping at `timeconst`; ICF's
`contact_hc_dissipation` is a Hunt & Crossley rate per unit approach velocity
[s/m]. There is no algebraic map. It is left at the ICF default (10.0) and is
**not calibrated**. The observable that would constrain it (drop-and-settle) has
not been run.

### 2.3 Per-world contact budget

ICF's `IcfParams.max_rigid_contact` defaults to 128 *(read)* and both arms drop
contacts silently on overflow. The launcher sets 512. Measured peak at rest was
~56 pipeline contacts per world *(measured)*; **the grasp-phase peak has not been
measured** and is the number that matters. Measure it before any training claim.

### 2.4 Friction — matched by nothing, and this is the largest unflagged confound

MuJoCo combines a pair's friction by element-wise **max**
(`newton/_src/solvers/mujoco/kernels.py`) *(read)*; ICF combines by **harmonic
mean**, `2*mu0*mu1/(mu0+mu1)` (`icf_warp/contact_material.py`) *(read)*. Authored
coefficients: rig 1.0, mug 0.2, table 0.6 *(read)*.

| pair | MuJoCo | ICF | ratio |
|---|---|---|---|
| pad ↔ mug | 1.0 | 0.333 | 3.0x |
| mug ↔ table | 0.6 | 0.300 | 2.0x |

A 3x friction difference on the pair that decides whether the mug is held makes
arm 1 vs arm 2 uninterpretable as a stepping or contact-compliance comparison. The
two rules agree for every pair **iff mu is uniform scene-wide**. Either set one mu
on rig, mug and table for the controlled runs, or report the 3x gap in every
arm-1 claim. **Neither has been done in the current task config.** Do not "fix" it
by editing `_combine_mu` — that changes ICF's contact model away from the
SAP/Drake convention.

## 3. Structurally unmatchable

### 3.1 Effective-mass scaling (AC4)

`K_eff^MJ` is proportional to `m_eff`, the reduced mass from `body_invweight0`;
ICF's k is one constant for every pair. Measured spread over the pairs this scene
can load *(measured)*:

* object pairs only (mug against worldbody and against each active-arm collider):
  **R = 1.048**. The mug is the lightest body in every pair it makes, so `m_eff` is
  essentially the mug's mass throughout and a single ICF k covers all of them.
* including the active arm's own colliders against the slab: **R = 4.79e6**,
  running from mug↔`gripper_right` (K_eff 1371 N/m) to `link_1`↔world
  (K_eff 6.57e9 N/m).

So: **the fixed-MuJoCo arm is a stiffness-matched control for mug contacts and is
not one for arm-on-table contacts.** ICF presses the slab with the same ~289 N/m
spring it uses for the mug, where MuJoCo uses a stiffness larger by up to seven
orders of magnitude. Every arm-1 claim must be scoped to the mug pairs, or must
show that arm-on-table contact never occurs in the episodes being compared.

### 3.2 Shape of the contact law

MuJoCo's impedance ramps `dmin -> dmax` over `width` with `power = 2`, so `K_eff`
stiffens with depth; ICF is a flat linear spring. Measured law per geom class
*(measured)*: mug geoms carry `solimp` `dmin 0.9, dmax 0.999, width 0.002`, while
the slab, gripper and link geoms carry `0.9, 0.95, 0.001`; MuJoCo blends them by
`solmix` for the pair. **The survey's claim that solimp is nowhere authored and
falls back to Newton's default is wrong** — the mug authors its own. Matching at
one `(m_eff, delta)` therefore leaves every other depth off, and the mismatch is
not even symmetric between the two geoms of the reference pair.

### 3.3 REFSAFE: MuJoCo's contact law is a function of dt

`timeconst = max(timeconst, 2*dt)` unless `DisableBit.REFSAFE`
(`mujoco_warp/_src/constraint.py`) *(read)*. Measured on the running model:
authored `timeconst = 0.02 s`, `dt = 1/90 = 0.011111 s`, effective
`timeconst = 0.022222 s` — **the clamp binds on every contact** *(measured)*.

Consequences:

1. The arm runs a contact stiffness `(0.02/0.022222)^2 = 0.81x` the authored one.
2. A dt-refinement ladder on this arm silently stiffens the contact model until
   `dt <= 10 ms` and then freezes it. Such a ladder therefore refines *model and
   stepping together*, which is exactly the confound section 4's P7 exists to
   remove.

**Recommended fix, not applied:** raise the authored `solref[0]` to 0.025 s
scene-wide (mug `MujocoCollisionCfg`, rig `MujocoCollisionCfg`, and
`table_guard`'s `NewtonMaterialCfg(contact_stiffness=1600, contact_damping=80)`,
which is the `(ke, kd)` pair `convert_solref` maps to exactly `(0.025, 1.0)`), and
add an assertion that every authored `solref[0] >= 2 * sim.dt / num_substeps`. The
current ICF calibration is against the clamped law, so **changing this invalidates
`k = 289.2` and the probe must be re-run.**

### 3.4 Torsional and rolling friction

The rig's geoms carry `condim=6` and the task states the grasp depends on it
*(read)*. `grep -rn "torsional\|rolling" icf_warp/` returns nothing *(read)*: ICF
models neither, and resists twist only through the geometric spread of contact
points in a patch. Equalizing means dropping `condim=6` from the MuJoCo arm, which
the task config says breaks its grasp.

### 3.5 Friction cone and impratio

MuJoCo: elliptic cone, `impratio = 10` — the tangential constraint is 10x stiffer
than the normal one *(read)*. ICF: no `impratio` at all *(read)*; friction is
regularized by `contact_sigma` and `contact_stiction_tolerance`. No mapping exists.

### 3.6 Joint dissipation

`ImplicitActuatorCfg(friction=0.1)` on every actuator group reaches MuJoCo's
`dof_frictionloss`; `grep -rn "joint_friction\|joint_damping" icf_warp/` finds only
a docstring, no kernel *(read)*. **The MuJoCo arm gets 0.1 N·m / 0.1 N of dry
friction per joint that neither ICF arm gets.** This is removable (set
`friction=0.0` on all three arms) and has **not** been removed. The 3g probe run
makes the consequence visible: under tripled gravity the ICF arm's scene collapsed
(mug root drop `-0.027 m`, 555 contacts per world, not settled) while the MuJoCo
arm stayed within 0.5 mm of its slab *(measured)*.

### 3.7 Joint limit model, velocity limit, actuation discretization, integrator

Different by construction (MuJoCo `solreflimit`/`solimplimit` from `invweight0` vs
ICF's near-rigid limit with `limit_beta`; ICF enforces a joint velocity limit that
MuJoCo has no counterpart for; `implicitfast` position actuator vs ICF's implicit
per-dof PD constraint). *(unverified — read from the survey, not re-checked here.)*
The gripper close command drives both carriages onto their hard lower limit, so the
limit model **binds** in normal operation.

### 3.8 `update_data_interval = 2`

The MuJoCo arm re-reads Newton joint state into `MjData` every other step *(read in
the task cfg; the solver-side consequence is unverified)*, giving it up to one step
of reset latency that the ICF arms do not have. Removable: set 1.

## 4. Matched-accuracy protocol for any wall-time claim

**P1. There is no single reference for all three arms.** Arms 1 and 2 do not share
a continuum limit (3.1, 3.2, 2.4, 3.4, 3.6). **No matched-accuracy wall-time claim
may cross the arm-1 / arm-2 boundary.**

**P2. Reference for arms 2 and 3:** fixed ICF at `dt_ref`, same `k`, `d`, `mu` and
the same once-per-boundary collide cadence. `dt_ref` is found by self-convergence —
halve until the observable moves by less than one tenth of the arm-to-arm
difference being claimed — and the convergence table is published.

**P3. Driving input:** an open-loop action tape, identical across arms and dts. A
policy in the loop makes each arm follow a different trajectory.

**P4. Error metric,** reported as a distribution over worlds (median / p95 / max),
never a mean: `max` over boundaries of mug position inf-norm, geodesic rotation
error, arm joint-position inf-norm. Report a task-level outcome (final mug height,
held / not held) alongside — the trajectory norm can be small while the outcome
flips.

**P5. Wall time:** steady-state ms per env step at the production world count, same
GPU, **both arms CUDA-graph captured** (`_supports_cuda_graph_capture` now returns
True for both ICF arms), warm, first 200 steps discarded, and
`NEWTON_ADAPTIVE_LOG_EVERY=0` — the telemetry reads `solver.dt.numpy()`, a full
device sync. Report a schedule-invariant work axis beside it: cumulative accepted
inner substeps for arm 3, boundaries for arm 2.

**P6. Claim form:** "adaptive ICF reaches error e at wall time `T_a`; fixed ICF
needs `dt*` to reach the same e, and costs `T_f` there, `T_f/T_a = X`." `dt*` is
found by the refinement ladder. Comparing arm 3's wall time against arm 2 at
`dt = 1/90`, where arm 2's error is larger, is forbidden.

**P7. The load-bearing control for the arm-1 story.** If the claim is "fixed-step
MuJoCo produces artifact X", you must show X is a *stepping* artifact and not a
contact-model artifact, by refining fixed MuJoCo's own dt **with the contact law
held fixed** (3.3 first, otherwise REFSAFE changes the model as you refine) and
showing X disappears. Without that self-refinement study, "fixed-step fails" is
indistinguishable from "MuJoCo's contact model fails".

**P8. Training comparisons:** same seeds, same env-step budget (not same wall
time), same network and PPO hyperparameters, and a solver-invariant reward. The
task's largest reward term, `pad_contact` (weight 40), thresholds a **contact
force** at 0.1 N and therefore reads a solver-dependent quantity that also depends
on ICF's per-world contact budget. Fix or replace it before comparing curves. Report
N >= 3 seeds per arm; one seed cannot support "trains vs does not train".

## 5. Known gaps in this document

1. **AC3 (transferability) was not run.** The check that MuJoCo's mass
   normalization actually predicts a second pair's stiffness — the pad↔mug
   penetration ratio predicted in advance from `body_invweight0` — needs a scripted
   gripper close, which the probe does not implement. Until it runs, the
   calibration is untested outside the mug↔slab pair.
2. **The 3g load-non-linearity check produced no usable number.** Both arms failed
   the settled certificate at 3g (MuJoCo `dz_max = 2.5e-5 m`, ICF `-0.027 m` root
   drop with the scene collapsed), so the probe reports INVALID rather than a ratio
   *(measured)*. The MuJoCo ratio was trending to 2.65 rather than 3.00, consistent
   with its depth-stiffening impedance, but on unsettled data.
3. **`contact_hc_dissipation` is uncalibrated** (2.2).
4. **Grasp-phase peak contact count is unmeasured** (2.3).
5. **Friction combination is unequalized** (2.4) and **joint Coulomb friction is
   unequalized** (3.6). Both are removable and neither has been removed.
6. **REFSAFE is binding** (3.3) and has not been fixed; the current k is calibrated
   against the clamped law.
7. **The ICF knobs are still shell environment variables**
   (`ICF_CONTACT_STIFFNESS`, `ICF_HC_DISSIPATION`, `ICF_MAX_RIGID_CONTACT`,
   `NEWTON_ADAPTIVE_TOL`, `ICF_ADAPTIVE_DT_MIN`, `ICF_ADAPTIVE_RTOL`,
   `NEWTON_ADAPTIVE_MAX_SUBSTEPS`), not task-config fields. They are read **once,
   above the fixed/adaptive split**, so both ICF arms provably get the same
   `IcfParams` object within a process — but a different shell can still give arm 2
   and arm 3 different physics across processes. `run_three_arm.sh` exists to make
   that impossible in practice; moving them onto `MJWarpSolverCfg` would make it
   impossible in principle.
8. **MDP parity between the ICF arms is approximate, not exact.** Arm 3 latches
   `diverged` from its own controller; arm 2 now reports the same
   `contact_solve.converged_env` certificate through the same `_diverged_pending`
   mask, so both have a `physics_diverged` pathway where before only arm 3 did.
   The *thresholds still differ*: arm 3 rejects a failed solve and retries at a
   smaller dt, latching only when it is still failing at the dt floor, while arm 2
   has no retry and latches on the first failure. Arm 1 has no such pathway at all.
9. **Arm 3 has not been run.** `--solver icf-adaptive` is wired end to end and
   fails with an actionable `ImportError` naming `IcfAdaptiveParams` /
   `SolverICFAdaptive`, because that solver does not yet exist in the `icf_warp`
   checkout. Nothing in this document about arm 3's behaviour has been measured.
