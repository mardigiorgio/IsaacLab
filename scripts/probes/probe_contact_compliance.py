# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matched-compliance calibration probe for the fixed-MuJoCo and fixed-ICF arms.

MuJoCo's normal contact is acceleration-space and mass-normalized
(``K_eff = m_eff * imp^2 / (dmax^2 * timeconst^2 * dampratio^2 * (1 - imp))``),
ICF's is one global linear spring ``fn = k * delta`` in N/m. No choice of ``k``
makes the two laws equal; the most that can be done is to make them equal at ONE
named operating point and to state the residual spread everywhere else. This
script does exactly that, and nothing more.

REFERENCE CONTACT: the manipulated object resting on the tabletop slab, arm at
its default pose, zero action. It is the pair the object's whole weight loads and
the reduced mass is dominated by a body whose mass is authored.

WHAT IT MEASURES
  build  Read straight off the constructed MuJoCo model, no simulation:
         ``body_invweight0``, ``geom_solref``, ``geom_solimp``, ``opt.timestep``.
         Yields ``m_eff`` and ``K_eff`` per load-carrying body pair, the spread
         ``R = max K_eff / min K_eff`` over those pairs, and whether MuJoCo's
         REFSAFE clamp ``timeconst = max(timeconst, 2*dt)`` is binding (it
         rewrites the authored contact law as a function of dt).
  settle Steady-state ROOT DROP ``slab_top_z - object_root_z`` under a zero action
         -- the acceptance observable, because it is a pose defined the same way
         on both arms. It equals the contact penetration plus a fixed geometric
         offset between the body origin and the collision mesh's lowest witness,
         so the MEAN PENETRATION over the object-slab contacts is measured too:
         that is the quantity ``k`` sets, and the only one proportional to 1/k.
         Also: a settled-ness certificate (max per-step |dz| over the tail) and a
         per-world census of penetrating object-slab contacts.

NO HOST SYNC IN THE PHYSICS LOOP: every per-step quantity is folded into device
accumulators (torch elementwise ops for the pose observable, one Warp kernel for
the contact census) and read back exactly once, after the loop.

USAGE (from the IsaacLab root; single lines only)
  export VIRTUAL_ENV=$HOME/Documents/code/icra2027/.venv
  export P="scripts/probes/probe_contact_compliance.py"
  export A="--num_envs 16 --steps 300 --tail 150 --settle_win 60 --viz none"
  ./isaaclab.sh -p $P --arm mujoco $A --out /tmp/cal_mujoco.json
  ICF_CONTACT_STIFFNESS=289.2 ./isaaclab.sh -p $P --arm icf $A --out /tmp/cal_icf.json
  ./isaaclab.sh -p $P --verdict /tmp/cal_mujoco.json /tmp/cal_icf.json

The last invocation runs no simulation. It prints the acceptance verdict and two
rescales: the one-shot ``k* = k0 * pen_ICF(k0) / pen_MJ`` (exact only while the
contact count ``n`` is unchanged, which the verdict tests), and, when two ICF runs
at different ``k`` are supplied, the closed-form solve of ``d(k) = d0 + C/k`` on
the root drop, which also recovers the geometric offset ``d0``. Neither is a
sweep; the second is a two-point solve and is needed whenever the one-shot rescale
moves ``n``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# --------------------------------------------------------------------------- verdict


def _fmt(x, unit="", digits=6):
    return "n/a" if x is None else f"{x:.{digits}g}{unit}"


def print_verdict(paths: list[str]) -> int:
    """Print the acceptance verdict from result files. Runs no simulation."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    for path, r in zip(paths, runs):
        r["_path"] = path

    print("\n================ CALIBRATION VERDICT ================")
    print(
        f"{'file':<30} {'arm':<6} {'g':>5} {'k_ICF[N/m]':>11} {'pen[m]':>11} {'rootdrop[m]':>11} "
        f"{'settled[m]':>11} {'n_pen':>6}"
    )
    for r in runs:
        print(
            f"{os.path.basename(r['_path']):<30} {r['arm']:<6} {r['gravity_scale']:>5.2f} "
            f"{_fmt(r.get('icf_contact_stiffness'), digits=5):>11} "
            f"{_fmt(r.get('penetration_mean'), digits=5):>11} "
            f"{_fmt(r['delta_rest_mean'], digits=5):>11} "
            f"{_fmt(r['settle_dz_max'], digits=3):>11} "
            f"{r['contacts_penetrating_median']:>6}"
        )

    one_g = {r["arm"]: r for r in runs if abs(r["gravity_scale"] - 1.0) < 1e-9}
    mj = one_g.get("mujoco")
    icf_runs = sorted(
        (r for r in runs if r["arm"] == "icf" and abs(r["gravity_scale"] - 1.0) < 1e-9),
        key=lambda r: r.get("icf_contact_stiffness") or 0.0,
    )
    # When several ICF stiffnesses were run, the acceptance is judged on the one
    # closest to the MuJoCo reference; the others are the rescale's evidence.
    if mj is not None and icf_runs:
        icf = min(icf_runs, key=lambda r: abs(r["delta_rest_mean"] - mj["delta_rest_mean"]))
    else:
        icf = icf_runs[-1] if icf_runs else None
    one_g["icf"] = icf

    print("\n---- STEP 1: build-time K_eff spread (MuJoCo model constants, no simulation) ----")
    if mj is None or not mj.get("keff_pairs"):
        print("  NOT AVAILABLE: no 1g fixed-MuJoCo run in the given files.")
    else:
        for row in mj["keff_pairs"]:
            print(
                f"  {row['body_a']:<40} <-> {row['body_b']:<40} "
                f"m_eff={row['m_eff']:.5g} kg  K_eff(sat)={row['k_eff_saturated']:.5g} N/m"
            )
        print(
            f"  timeconst authored={_fmt(mj.get('solref_timeconst_authored'), ' s')}  "
            f"effective(after REFSAFE)={_fmt(mj.get('solref_timeconst_effective'), ' s')}  "
            f"dt={_fmt(mj.get('mj_timestep'), ' s')}"
        )
        if mj.get("refsafe_binding"):
            print("  !! REFSAFE IS BINDING: the running contact law is NOT the authored one, and it")
            print("     changes with dt. Any dt-refinement study on this arm refines model AND stepping.")

    print("\n---- STEP 3: one-shot rescale (on the MEAN PENETRATION observable) ----")
    k_star = None
    if mj is None or icf is None:
        print("  NOT AVAILABLE: need one 1g fixed-MuJoCo run and one 1g fixed-ICF run.")
    elif icf.get("icf_contact_stiffness") is None:
        print("  NOT AVAILABLE: the ICF run did not record its contact stiffness.")
    elif not (mj.get("penetration_mean") or 0.0) > 0.0 or not (icf.get("penetration_mean") or 0.0) > 0.0:
        print("  NOT AVAILABLE: one arm reported no force-bearing object<->slab contact, so there is")
        print("  no reference penetration to match. Check the shape-role matching in both runs.")
    else:
        k0 = icf["icf_contact_stiffness"]
        k_star = k0 * icf["penetration_mean"] / mj["penetration_mean"]
        print(
            f"  k0 = {k0:.6g} N/m   pen_ICF(k0) = {icf['penetration_mean']:.6g} m   "
            f"pen_MJ = {mj['penetration_mean']:.6g} m"
        )
        print(f"  k* = k0 * pen_ICF(k0) / pen_MJ = {k_star:.6g} N/m")
        ks_mj, ks_icf = mj.get("k_secant_measured"), icf.get("k_secant_measured")
        if ks_mj and ks_icf:
            print(f"  cross-check: MuJoCo secant k at this depth = {ks_mj:.6g} N/m; the ICF arm's own")
            print(f"  W/(n*pen) at k0 = {ks_icf:.6g} N/m (should equal k0={k0:.6g} if the load is shared")
            print("  equally over vertical normals -- a large gap means the census n is not the")
            print("  effective count and k* inherits that error).")
        print("  ICF's rest depth is exactly W/(n*k), so this rescale is a division, not a fit --")
        print("  but only while the contact set n is unchanged, which AC2 tests. Re-run the ICF arm")
        print("  at k* and confirm AC1.")

    # TWO-POINT SOLVE, on the root-drop observable. The one-shot rescale above
    # assumes the rest depth is exactly W/(n*k); the root drop is instead
    # d(k) = d0 + C/k, where d0 is the fixed offset between the body origin and
    # the collision mesh's lowest witness. Two ICF runs at different k determine
    # d0 and C without a sweep, and the target d_MJ then gives k in closed form.
    # It is only valid across runs that reported the SAME contact count n.
    usable = [r for r in icf_runs if r.get("icf_contact_stiffness")]
    if mj is not None and len(usable) >= 2:
        a, b = usable[0], usable[-1]
        ka, kb = a["icf_contact_stiffness"], b["icf_contact_stiffness"]
        da, db = a["delta_rest_mean"], b["delta_rest_mean"]
        same_n = a["contacts_penetrating_median"] == b["contacts_penetrating_median"]
        denom = (1.0 / ka) - (1.0 / kb)
        print("\n---- STEP 3b: two-point solve on the root-drop observable ----")
        if abs(denom) < 1e-15:
            print("  NOT AVAILABLE: the two ICF runs used the same stiffness.")
        else:
            C = (da - db) / denom
            d0 = da - C / ka
            need = mj["delta_rest_mean"] - d0
            print(
                f"  from k={ka:.6g} (d={da:.6g} m, n={a['contacts_penetrating_median']}) and "
                f"k={kb:.6g} (d={db:.6g} m, n={b['contacts_penetrating_median']}):"
            )
            print(f"    geometric offset d0 = {d0:.6g} m,  C = W/n_eff = {C:.6g} N")
            if need <= 0.0:
                print("    the MuJoCo target lies below the offset; no positive k reproduces it.")
            else:
                print(f"    k solving d0 + C/k = d_MJ = {mj['delta_rest_mean']:.6g} m  ->  k = {C / need:.6g} N/m")
            if not same_n:
                print("    !! the two runs reported DIFFERENT contact counts, so d0+C/k is not one")
                print("       model across them and this solve is an extrapolation, not a fit.")

    print("\n---- ACCEPTANCE ----")
    verdicts = {}
    if mj is None or icf is None:
        verdicts["AC1"] = ("SKIP", "need both a 1g MuJoCo and a 1g ICF run")
        verdicts["AC2"] = ("SKIP", "need both a 1g MuJoCo and a 1g ICF run")
    else:
        settled = mj["settle_dz_max"] < 1e-6 and icf["settle_dz_max"] < 1e-6
        pm, pi = mj.get("penetration_mean"), icf.get("penetration_mean")
        rel_pen = abs(pi - pm) / abs(pm) if (pm and pi) else float("inf")
        # The root drop is the acceptance observable: it is a pose, defined the
        # same way on both arms, and it does not depend on which contacts the
        # census counted. The penetration match is reported beside it because it
        # is the quantity the contact law actually sets.
        rel = (
            abs(icf["delta_rest_mean"] - mj["delta_rest_mean"]) / abs(mj["delta_rest_mean"])
            if mj["delta_rest_mean"]
            else float("inf")
        )
        ok = settled and rel <= 0.05
        verdicts["AC1"] = (
            "PASS" if ok else "FAIL",
            f"k={_fmt(icf.get('icf_contact_stiffness'), ' N/m', 6)}: "
            f"|d_ICF - d_MJ|/d_MJ = {rel:.3g} (<= 0.05 required); mean-penetration match "
            f"{rel_pen:.3g}; settled: MJ dz_max={mj['settle_dz_max']:.3g} m, "
            f"ICF dz_max={icf['settle_dz_max']:.3g} m (< 1e-6 m required)",
        )
        same_n = mj["contacts_penetrating_median"] == icf["contacts_penetrating_median"]
        verdicts["AC2"] = (
            "PASS" if same_n else "FAIL",
            f"penetrating object-slab contacts per world: MJ={mj['contacts_penetrating_median']}, "
            f"ICF={icf['contacts_penetrating_median']}",
        )
    verdicts["AC3"] = (
        "NOT RUN",
        "pad<->object transferability is not implemented by this probe; it needs a scripted "
        "gripper close and is the check that the mass-normalization model is right. Until it "
        "runs, the calibration is untested outside the object<->slab pair.",
    )
    if mj is None or mj.get("keff_spread_R") is None:
        verdicts["AC4"] = ("SKIP", "no fixed-MuJoCo run to read body_invweight0 from")
    else:
        R = mj["keff_spread_R"]
        Ro = mj.get("keff_spread_R_object_pairs")
        verdicts["AC4"] = (
            "PASS" if R <= 10.0 else "REPORTED-FAIL",
            f"K_eff spread R = {R:.4g} over all load-carrying pairs "
            f"({mj.get('keff_min_pair')} .. {mj.get('keff_max_pair')}); object pairs alone "
            f"R = {_fmt(Ro, digits=4)}"
            + (
                ""
                if R <= 10.0
                else ". The fixed-MuJoCo arm is NOT a stiffness-matched control "
                "outside the object-limited pairs -- ICF's single global k cannot follow K_eff's "
                "mass scaling, so scope every arm-1 claim to the object pairs"
            ),
        )
    for key in ("AC1", "AC2", "AC3", "AC4"):
        status, why = verdicts[key]
        print(f"  {key}: {status:<13} {why}")

    print("\n---- LOAD NON-LINEARITY (observable C, static half) ----")
    for arm in ("mujoco", "icf"):
        base = one_g.get(arm)
        heavy = [r for r in runs if r["arm"] == arm and r["gravity_scale"] > 1.0 + 1e-9]
        if base is None or not heavy or not base.get("penetration_mean"):
            print(f"  {arm}: NOT RUN (needs a 1g and a >1g run of this arm)")
            continue
        for h in heavy:
            if not h.get("penetration_mean"):
                print(f"  {arm}: {h['gravity_scale']}g run reported no penetrating contact")
                continue
            # A ratio taken from an unsettled state is not a static observable at
            # all, so it is reported as invalid rather than as a number.
            if h["settle_dz_max"] >= 1e-6 or base["settle_dz_max"] >= 1e-6:
                print(
                    f"  {arm}: INVALID at {h['gravity_scale']:.2f}g -- NOT SETTLED "
                    f"(dz_max {h['settle_dz_max']:.3g} m; rest depth {h['delta_rest_mean']:.4g} m, "
                    f"spread over worlds {h['delta_rest_std']:.3g} m). The scene moved, so this is "
                    f"not a static load test."
                )
                continue
            ratio = h["penetration_mean"] / base["penetration_mean"]
            print(
                f"  {arm}: pen({h['gravity_scale']:.2f}g)/pen(1g) = {ratio:.4f} "
                f"(ICF's linear spring predicts exactly {h['gravity_scale']:.2f}; MuJoCo predicts "
                f"strictly less because imp stiffens with depth)"
            )
    print("=====================================================\n")
    return 0


# --------------------------------------------------------------------------- CLI

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--verdict",
    type=str,
    nargs="+",
    default=None,
    help="Offline mode: print the acceptance verdict from result JSON files and exit.",
)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument(
    "--arm",
    type=str,
    default="mujoco",
    choices=["mujoco", "icf"],
    help="Fixed-step arm to measure. The adaptive arm is not a calibration target.",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600, help="Env steps (control ticks) to run.")
parser.add_argument("--tail", type=int, default=300, help="Trailing env steps averaged into the rest depth.")
parser.add_argument(
    "--settle_win", type=int, default=100, help="Trailing env steps over which max per-step |dz| certifies settledness."
)
parser.add_argument(
    "--gravity_scale",
    type=float,
    default=1.0,
    help="Multiplies the gravity vector. 3.0 gives the load-non-linearity observable.",
)
parser.add_argument("--object_asset", type=str, default="object", help="Scene entity name of the manipulated body.")
parser.add_argument("--slab_asset", type=str, default="table_guard", help="Scene entity name of the tabletop slab.")
parser.add_argument(
    "--arm_body_hint",
    type=str,
    default="follower_left",
    help="Substring selecting the ACTIVE arm's bodies for the K_eff spread.",
)
parser.add_argument("--out", type=str, default=None, help="Write the result JSON here.")
parser.add_argument("--seed", type=int, default=42)

if "--verdict" in sys.argv:
    _args, _ = parser.parse_known_args()
    raise SystemExit(print_verdict(_args.verdict))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import warp as wp  # noqa: E402

# --------------------------------------------------------------------------- census kernel


@wp.kernel(enable_backward=False)
def _contact_census(
    contact_count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    margin0: wp.array(dtype=wp.float32),
    margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    shape_role: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    n_all: wp.array(dtype=wp.int32),
    n_pen: wp.array(dtype=wp.int32),
    depth_max: wp.array(dtype=wp.float32),
    depth_sum: wp.array(dtype=wp.float32),
):
    """Per-world census of the pipeline contact set, reusing ICF's own gap definition.

    The separation is ``phi = dot(x1 - x0, n) - margin0 - margin1`` with the
    witness points transformed by the current body poses -- the same expression
    the ICF patch builder evaluates, so "penetrating" here means force-bearing on
    the ICF arm and (margin 0, gap discarded on the injected MuJoCo path)
    force-bearing on the MuJoCo arm too. Both arms consume the SAME contact
    buffer, so a difference in this count between arms means one of them dropped
    contacts on its own per-world budget.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = shape0[tid]
    s1 = shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    w = shape_world[s0]
    if w < 0:
        w = shape_world[s1]
    if w < 0:
        return
    wp.atomic_add(n_all, w, 1)
    r0 = shape_role[s0]
    r1 = shape_role[s1]
    # Role 1 is the manipulated object, role 2 the slab; only that pair carries
    # the reference load whose depth the calibration matches.
    if not ((r0 == 1 and r1 == 2) or (r0 == 2 and r1 == 1)):
        return
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    x0 = point0[tid]
    x1 = point1[tid]
    if b0 >= 0:
        x0 = wp.transform_point(body_q[b0], x0)
    if b1 >= 0:
        x1 = wp.transform_point(body_q[b1], x1)
    n = wp.normalize(normal[tid])
    phi = wp.dot(x1 - x0, n) - margin0[tid] - margin1[tid]
    if phi < 0.0:
        wp.atomic_add(n_pen, w, 1)
        wp.atomic_max(depth_max, w, -phi)
        wp.atomic_add(depth_sum, w, -phi)


# --------------------------------------------------------------------------- build-time read


def read_mujoco_compliance(solver, arm_body_hint: str) -> dict:
    """Step 1: derive ``m_eff`` and ``K_eff`` per body pair from MuJoCo model constants.

    Costs no simulation. ``K_eff`` follows from the efc row the solver builds:
    ``D = imp / ((1 - imp) * invweight)`` and ``aref = -k * imp * pos``, so at
    static equilibrium ``f = D * k * imp * pos``, i.e.
    ``K_eff = imp^2 * k / ((1 - imp) * invweight)`` with
    ``k = 1 / (dmax^2 * timeconst^2 * dampratio^2)`` and ``invweight`` the sum of
    the two bodies' ``body_invweight0[..][0]``. Saturated (``imp = dmax``) this is
    ``m_eff / (timeconst^2 * dampratio^2 * (1 - dmax))``.

    Must be called AFTER at least one step: ``opt.timestep`` is filled per step
    from the dt the manager hands ``SolverMuJoCo.step``, so before the first step
    it still holds the construction default and the REFSAFE verdict derived from
    it would be about a timestep the simulation never runs.

    The contact law is read PER GEOM CLASS rather than globally: shapes that
    author no ``mjc:solref`` take the scene's fallback ke/kd conversion, which is
    a different response time from the one the reference pair authors, so a single
    scene-wide min/max would describe no actual contact.
    """
    import mujoco
    import numpy as np

    out: dict = {"keff_pairs": [], "keff_spread_R": None}
    mjm = getattr(solver, "mj_model", None)
    mjw = getattr(solver, "mjw_model", None)
    if mjm is None or mjw is None:
        out["error"] = "solver exposes no mj_model/mjw_model; not a MuJoCo backend"
        return out

    invw = mjw.body_invweight0.numpy()  # (nworld_or_1, nbody, 2)
    timestep = float(np.asarray(mjw.opt.timestep.numpy()).reshape(-1)[0])
    solref = mjw.geom_solref.numpy()  # (nworld_or_1, ngeom, 2)
    solimp = mjw.geom_solimp.numpy()  # (nworld_or_1, ngeom, 5)
    out["mj_timestep"] = timestep

    nbody = mjm.nbody
    bnames = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"#body{i}" for i in range(nbody)]
    ngeom = mjm.ngeom
    gnames = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) or f"#geom{i}" for i in range(ngeom)]
    geom_body = np.asarray(mjm.geom_bodyid).reshape(-1)

    def geom_class(i: int) -> str:
        low = (gnames[i] + "|" + bnames[int(geom_body[i])]).lower()
        if "object" in low:
            return "object"
        if "tableguard" in low:
            return "slab"
        if "gripper" in low:
            return "gripper"
        if "link" in low:
            return "link"
        return "other"

    ref0 = solref[0]
    imp0 = solimp[0]
    classes: dict[str, dict] = {}
    for i in range(ngeom):
        c = geom_class(i)
        key = (
            round(float(ref0[i][0]), 12),
            round(float(ref0[i][1]), 12),
            round(float(imp0[i][0]), 12),
            round(float(imp0[i][1]), 12),
            round(float(imp0[i][2]), 12),
        )
        entry = classes.setdefault(c, {})
        entry[key] = entry.get(key, 0) + 1
    out["geom_law_by_class"] = {
        c: [
            {"timeconst": k[0], "dampratio": k[1], "dmin": k[2], "dmax": k[3], "width": k[4], "geoms": n}
            for k, n in sorted(v.items())
        ]
        for c, v in sorted(classes.items())
    }

    # The reference pair's law: object geoms against slab geoms. Both are expected
    # to author the same solref, in which case MuJoCo's solmix blend is that value
    # whatever the mixing weight; a split is reported instead of averaged.
    def _single(c):
        rows = out["geom_law_by_class"].get(c, [])
        return rows[0] if len(rows) == 1 else None

    obj_law, slab_law = _single("object"), _single("slab")
    if obj_law is None or slab_law is None:
        out["reference_law"] = None
        out["reference_law_note"] = (
            "object and/or slab geoms do not share one solref/solimp; the reference contact has no "
            "single law and K_eff below is not defined for it."
        )
        return out

    # PAIR MIXING, as the solver resolves it: both solref[0] > 0 gives the
    # solmix-weighted blend of the two geoms' solref and solimp, so a pair's law
    # is neither geom's authored value unless the two agree.
    solmix = mjw.geom_solmix.numpy()[0] if hasattr(mjw, "geom_solmix") else None
    obj_geom = next(i for i in range(ngeom) if geom_class(i) == "object")
    slab_geom = next(i for i in range(ngeom) if geom_class(i) == "slab")
    if solmix is None:
        mix = 0.5
    else:
        m1, m2 = float(solmix[obj_geom]), float(solmix[slab_geom])
        mix = 0.5 if (m1 + m2) <= 0.0 else m1 / (m1 + m2)
    out["reference_pair_solmix_weight"] = mix
    if obj_law["timeconst"] > 0.0 and slab_law["timeconst"] > 0.0:
        tc_authored = mix * obj_law["timeconst"] + (1.0 - mix) * slab_law["timeconst"]
        dr = mix * obj_law["dampratio"] + (1.0 - mix) * slab_law["dampratio"]
    else:
        tc_authored = min(obj_law["timeconst"], slab_law["timeconst"])
        dr = min(obj_law["dampratio"], slab_law["dampratio"])
    dmax = mix * obj_law["dmax"] + (1.0 - mix) * slab_law["dmax"]
    width = mix * obj_law["width"] + (1.0 - mix) * slab_law["width"]
    tc_eff = max(tc_authored, 2.0 * timestep)
    out["reference_law"] = {"timeconst": tc_authored, "dampratio": dr, "dmax": dmax, "width": width}
    out["solref_timeconst_authored"] = tc_authored
    out["solref_dampratio"] = dr
    out["solimp_dmax"] = dmax
    out["solimp_width"] = width
    out["solref_timeconst_effective"] = tc_eff
    out["refsafe_binding"] = bool(tc_eff > tc_authored + 1e-15)

    inv0 = invw[0]  # world 0; this task randomizes no per-world inertia
    obj_ids = [i for i, nm in enumerate(bnames) if "object" in nm.lower()]
    out["mj_object_bodies"] = [bnames[i] for i in obj_ids]
    # LOAD-CARRYING PAIRS ONLY: the object against worldbody (which the slab is
    # welded to) and against the ACTIVE arm's own gripper/link bodies. Including
    # the idle arm would inflate the spread with pairs the task never loads.
    # A body with no collision geom can never carry a contact, so it is not a
    # load-carrying pair and must not inflate the spread.
    bodies_with_geoms = set(int(b) for b in geom_body.tolist())
    partners = [0] + [
        i
        for i, nm in enumerate(bnames)
        if i in bodies_with_geoms
        and arm_body_hint.lower() in nm.lower()
        and ("gripper" in nm.lower() or "link" in nm.lower())
    ]
    rows = []
    for oi in obj_ids:
        for pi in partners:
            if pi == oi:
                continue
            iw = float(inv0[oi][0]) + float(inv0[pi][0])
            if iw <= 0.0:
                continue
            m_eff = 1.0 / iw
            # Saturated impedance: imp -> dmax once the depth passes `width`.
            k_eff = m_eff / (tc_eff * tc_eff * dr * dr * (1.0 - dmax))
            rows.append(
                {
                    "body_a": bnames[oi],
                    "body_b": bnames[pi],
                    "m_eff": m_eff,
                    "k_eff_saturated": k_eff,
                }
            )
    # The OTHER load-carrying group: the active arm's own colliders against the
    # slab. ICF applies its ONE global k to these pairs too, so their K_eff must
    # be in the spread even though the calibration reference is an object pair.
    for pi in partners:
        if pi == 0:
            continue
        iw = float(inv0[pi][0]) + float(inv0[0][0])
        if iw <= 0.0:
            continue
        m_eff = 1.0 / iw
        rows.append(
            {
                "body_a": bnames[pi],
                "body_b": bnames[0],
                "m_eff": m_eff,
                "k_eff_saturated": m_eff / (tc_eff * tc_eff * dr * dr * (1.0 - dmax)),
            }
        )
    obj_rows = [r for r in rows if any(r["body_a"] == bnames[i] or r["body_b"] == bnames[i] for i in obj_ids)]
    if obj_rows:
        oks = [r["k_eff_saturated"] for r in obj_rows]
        out["keff_spread_R_object_pairs"] = max(oks) / min(oks)
        out["keff_object_pairs_count"] = len(obj_rows)
    out["keff_pairs"] = rows
    if rows:
        ks = [r["k_eff_saturated"] for r in rows]
        lo = min(rows, key=lambda r: r["k_eff_saturated"])
        hi = max(rows, key=lambda r: r["k_eff_saturated"])
        out["keff_spread_R"] = max(ks) / min(ks)
        out["keff_min_pair"] = f"{lo['body_a']} <-> {lo['body_b']}"
        out["keff_max_pair"] = f"{hi['body_a']} <-> {hi['body_b']}"
    return out


# --------------------------------------------------------------------------- main


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        apply_solver_choice(env_cfg, args_cli.arm)
        env_cfg.sim.gravity = tuple(g * args_cli.gravity_scale for g in env_cfg.sim.gravity)
        # One uninterrupted settle: an env reset mid-run would restart the drop
        # and poison the tail average, so the episode must outlast the probe.
        control_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        env_cfg.episode_length_s = (args_cli.steps + 10) / control_hz
        # Contact-pool capacity scales with world count; the task's authored caps
        # are sized for its production count and would dominate memory here. The
        # census reports the realized peak so an undersized pool is visible
        # rather than silent.
        ccfg = getattr(env_cfg.sim.physics, "collision_cfg", None)
        if ccfg is not None:
            scale = max(args_cli.num_envs / 8192.0, 1.0 / 512.0)
            ccfg.rigid_contact_max = max(65536, int(ccfg.rigid_contact_max * scale))
            ccfg.max_triangle_pairs = max(1_000_000, int(ccfg.max_triangle_pairs * scale))

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped

        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver = NewtonManager._solver
        model = NewtonManager._model
        contacts = NewtonManager._contacts
        device = unwrapped.device

        result: dict = {
            "task": args_cli.task,
            "arm": args_cli.arm,
            "solver_class": type(solver).__name__,
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "tail": int(args_cli.tail),
            "settle_win": int(args_cli.settle_win),
            "gravity_scale": float(args_cli.gravity_scale),
            "sim_dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "num_substeps": int(getattr(env_cfg.sim.physics, "num_substeps", 1)),
            "gravity": list(env_cfg.sim.gravity),
            "icf_contact_stiffness": None,
            "icf_max_rigid_contact": None,
        }
        params = getattr(solver, "params", None)
        if params is not None:
            result["icf_contact_stiffness"] = float(getattr(params, "contact_stiffness", float("nan")))
            result["icf_hc_dissipation"] = float(getattr(params, "contact_hc_dissipation", float("nan")))
            result["icf_max_rigid_contact"] = int(getattr(params, "max_rigid_contact", 0))

        # ---- geometry of the reference contact, re-derived from the cfg -------
        slab_cfg = getattr(env_cfg.scene, args_cli.slab_asset)
        slab_top = float(slab_cfg.init_state.pos[2]) + float(slab_cfg.spawn.size[2]) / 2.0
        result["slab_top_z"] = slab_top

        obj = unwrapped.scene[args_cli.object_asset]
        obj_mass = float(obj.data.default_mass.reshape(-1)[0])
        result["object_mass_kg"] = obj_mass
        result["object_weight_N"] = obj_mass * abs(env_cfg.sim.gravity[2])

        # ---- shape roles, host-side, once ------------------------------------
        labels = list(getattr(model, "shape_label", []) or [])
        roles_np = np.zeros(max(len(labels), 1), dtype=np.int32)
        for i, lab in enumerate(labels):
            low = lab.lower()
            if "/object/" in low or low.endswith("/object"):
                roles_np[i] = 1
            elif "tableguard" in low:
                roles_np[i] = 2
        result["shapes_object"] = int((roles_np == 1).sum())
        result["shapes_slab"] = int((roles_np == 2).sum())
        if result["shapes_object"] == 0 or result["shapes_slab"] == 0:
            print(
                "[probe] WARNING: could not identify object/slab shapes from shape_label; "
                "the contact census will report zero. First 20 labels:"
            )
            for lab in labels[:20]:
                print(f"    {lab}")

        shape_role = wp.array(roles_np, dtype=wp.int32, device=str(device))
        shape_world = getattr(model, "shape_world", None)
        if shape_world is None:
            raise RuntimeError("model.shape_world is not populated; the per-world census cannot attribute contacts.")

        n_worlds = int(model.world_count)
        n_all = wp.zeros(n_worlds, dtype=wp.int32, device=str(device))
        n_pen = wp.zeros(n_worlds, dtype=wp.int32, device=str(device))
        depth_max = wp.zeros(n_worlds, dtype=wp.float32, device=str(device))
        depth_sum = wp.zeros(n_worlds, dtype=wp.float32, device=str(device))
        census_dim = int(contacts.rigid_contact_shape0.shape[0]) if contacts is not None else 0

        # ---- device accumulators for the pose observable ----------------------
        zero_action = torch.zeros(unwrapped.action_space.shape, device=device, dtype=torch.float32)
        env.reset()
        z_prev = None
        z_sum = torch.zeros(args_cli.num_envs, device=device, dtype=torch.float64)
        z_min = torch.full((args_cli.num_envs,), 1e9, device=device, dtype=torch.float32)
        dz_max = torch.zeros(args_cli.num_envs, device=device, dtype=torch.float32)
        done_count = torch.zeros(args_cli.num_envs, device=device, dtype=torch.int32)
        n_tail = 0

        tail_start = args_cli.steps - args_cli.tail
        settle_start = args_cli.steps - args_cli.settle_win

        for i in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            z = obj.data.root_pos_w[:, 2] - unwrapped.scene.env_origins[:, 2]
            done_count += (terminated | truncated).to(torch.int32)
            z_min = torch.minimum(z_min, z)
            if i >= tail_start:
                z_sum += z.to(torch.float64)
                n_tail += 1
            if z_prev is not None and i >= settle_start:
                dz_max = torch.maximum(dz_max, (z - z_prev).abs())
            z_prev = z.clone()
            if census_dim and i >= settle_start:
                wp.launch(
                    _contact_census,
                    dim=census_dim,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        shape_world,
                        shape_role,
                        NewtonManager._state_0.body_q,
                        n_all,
                        n_pen,
                        depth_max,
                        depth_sum,
                    ],
                    device=str(device),
                )

        # ---- ONE readback, after the loop ------------------------------------
        z_mean = (z_sum / max(n_tail, 1)).cpu().numpy()
        z_min_np = z_min.cpu().numpy()
        dz_np = dz_max.cpu().numpy()
        done_np = done_count.cpu().numpy()
        n_all_np = n_all.numpy()
        n_pen_np = n_pen.numpy()
        depth_np = depth_max.numpy()
        depth_sum_np = depth_sum.numpy()

        # The census accumulated over settle_win launches, so divide back out.
        per_step = max(args_cli.settle_win, 1)
        pen_per_world = n_pen_np / per_step
        all_per_world = n_all_np / per_step

        delta = slab_top - z_mean
        result["delta_rest_per_world"] = [float(v) for v in delta]
        result["delta_rest_mean"] = float(np.mean(delta))
        result["delta_rest_std"] = float(np.std(delta))
        result["z_min_per_world"] = [float(v) for v in z_min_np]
        result["settle_dz_max"] = float(np.max(dz_np))
        result["episode_dones"] = int(done_np.sum())
        result["contacts_penetrating_median"] = int(round(float(np.median(pen_per_world))))
        result["contacts_penetrating_mean"] = float(np.mean(pen_per_world))
        result["contacts_all_per_world_mean"] = float(np.mean(all_per_world))
        result["contact_depth_max"] = float(np.max(depth_np))
        # THE CALIBRATION OBSERVABLE. Static equilibrium is sum_i k*delta_i = W,
        # so the MEAN penetration over the n force-bearing object<->slab contacts
        # is exactly W/(n*k) on the ICF arm -- strictly proportional to 1/k, which
        # is what makes the one-shot rescale a division rather than a fit. The
        # root-frame drop below is the same quantity plus any fixed geometric
        # offset between the body origin and the collision mesh's lowest witness,
        # so it is reported as a cross-check rather than used for the rescale.
        with np.errstate(invalid="ignore", divide="ignore"):
            pen_mean_world = np.where(n_pen_np > 0, depth_sum_np / np.maximum(n_pen_np, 1), np.nan)
        result["penetration_mean"] = float(np.nanmean(pen_mean_world)) if np.any(n_pen_np > 0) else None
        result["penetration_mean_std"] = float(np.nanstd(pen_mean_world)) if np.any(n_pen_np > 0) else None
        # Derived diagnostic, not a model constant: the linear spring constant that
        # would produce this rest depth if the object's whole weight were split
        # equally over n vertical contacts. On the ICF arm it IS the running k (to
        # the extent the normals are vertical and the load is shared); on the
        # MuJoCo arm it is the SECANT stiffness of its depth-stiffening law at this
        # one depth, which is the number an ICF k has to match.
        n_med = result["contacts_penetrating_median"]
        if result["penetration_mean"] and n_med > 0:
            result["k_secant_measured"] = result["object_weight_N"] / (n_med * result["penetration_mean"])
        else:
            result["k_secant_measured"] = None
        result["rigid_contact_max"] = int(getattr(ccfg, "rigid_contact_max", 0)) if ccfg else 0

        # AFTER the loop: opt.timestep only holds the running dt once the solver
        # has stepped at least once.
        if args_cli.arm == "mujoco":
            result.update(read_mujoco_compliance(solver, args_cli.arm_body_hint))

        env.close()

    # ---- report ----------------------------------------------------------
    print("\n================ CALIBRATION PROBE ================")
    print(f"  arm                 : {result['arm']} ({result['solver_class']})")
    print(
        f"  num_envs            : {result['num_envs']}   steps: {result['steps']}   "
        f"gravity_scale: {result['gravity_scale']}"
    )
    print(f"  sim.dt              : {result['sim_dt']:.6g} s   substeps: {result['num_substeps']}")
    print(f"  object mass         : {result['object_mass_kg']:.6g} kg   weight: {result['object_weight_N']:.6g} N")
    print(f"  slab top z          : {result['slab_top_z']:.6g} m")
    if result["icf_contact_stiffness"] is not None:
        print(
            f"  ICF k               : {result['icf_contact_stiffness']:.6g} N/m   "
            f"d: {result.get('icf_hc_dissipation')}   max_rigid_contact: {result['icf_max_rigid_contact']}"
        )
    print(
        f"  rest depth (mean)   : {result['delta_rest_mean']:.6g} m  (std over worlds {result['delta_rest_std']:.3g} m)"
    )
    print(
        f"  settled certificate : max per-step |dz| over last {result['settle_win']} steps = "
        f"{result['settle_dz_max']:.3g} m  ({'SETTLED' if result['settle_dz_max'] < 1e-6 else 'NOT SETTLED'})"
    )
    print(f"  episode dones       : {result['episode_dones']}  (must be 0 for a clean settle)")
    print(f"  MEAN penetration    : {_fmt(result['penetration_mean'], ' m')}  <- the calibration observable")
    print(f"  secant k (derived)  : {_fmt(result.get('k_secant_measured'), ' N/m')}  = W/(n*pen), n from the census")
    print(
        f"  object<->slab contacts per world: penetrating {result['contacts_penetrating_mean']:.3f}, "
        f"deepest {result['contact_depth_max']:.6g} m"
    )
    print(
        f"  all pipeline contacts per world : {result['contacts_all_per_world_mean']:.1f} "
        f"(pool {result['rigid_contact_max']})"
    )
    if result.get("geom_law_by_class"):
        print("  MuJoCo contact law per geom class (timeconst s, dampratio, dmin, dmax, width m):")
        for cls, rows in result["geom_law_by_class"].items():
            for row in rows:
                print(
                    f"    {cls:<8} {row['timeconst']:.6g}  {row['dampratio']:.4g}  {row['dmin']:.4g}  "
                    f"{row['dmax']:.4g}  {row['width']:.4g}   ({row['geoms']} geoms)"
                )
    if result.get("reference_law") is None and "reference_law_note" in result:
        print(f"  !! reference law    : {result['reference_law_note']}")
    if result.get("keff_pairs"):
        print(f"  MuJoCo dt (running) : {result['mj_timestep']:.6g} s")
        print(
            f"  MuJoCo timeconst    : authored {result['solref_timeconst_authored']:.6g} s -> effective "
            f"{result['solref_timeconst_effective']:.6g} s  "
            f"(REFSAFE {'BINDING' if result['refsafe_binding'] else 'not binding'})"
        )
        print(
            f"  K_eff spread R      : {result['keff_spread_R']:.4g} over "
            f"{len(result['keff_pairs'])} load-carrying pairs (object pairs + active-arm/slab pairs)"
        )
        print(
            f"    restricted to OBJECT pairs only: R = "
            f"{_fmt(result.get('keff_spread_R_object_pairs'), digits=4)} over "
            f"{result.get('keff_object_pairs_count')} pairs"
        )
        print("    NOTE: K_eff above is the SATURATED value (imp -> dmax). At the measured rest")
        print("    depth the impedance ramp has not saturated, so the running stiffness is softer;")
        print("    the secant k above is the number the ICF arm has to match.")
        print(f"    softest: {result['keff_min_pair']}")
        print(f"    stiffest: {result['keff_max_pair']}")
    print("===================================================\n")

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[probe] wrote {args_cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
