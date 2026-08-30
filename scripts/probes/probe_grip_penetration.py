# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grip-phase penetration probe for the Trossen mug lift, with built-in instrument validation.

Companion to ``probe_contact_compliance.py`` (rest-contact calibration). That probe
measures the object at rest under zero action; this one measures the two things it
cannot: the INSTRUMENT itself against a posed, known overlap, and penetration UNDER
GRIP while a frozen trained policy performs the task.

WHAT IT MEASURES
  pose-check  Instrument validation per the telemetry contract: the mug is teleported
              (direct ``body_q`` write, NO dynamics stepped) so its collision mesh's
              lowest vertex sits a KNOWN depth below the slab top, the collision
              pipeline is run once at that frozen state, and the census must report
              that depth. The expected depth is computed INDEPENDENTLY of the census,
              from the USD collision-mesh vertices. Runs at several depths.
  rollout     A frozen policy checkpoint drives the arm. At EVERY solver tick, after
              the collide and BEFORE the solve — so the census reads exactly the
              contact set and poses the solver consumes, not a re-query one substep
              off — per-world penetration is recorded for two pair classes:
              pad<->object (the grip preload case) and object<->slab (the rest case).
              Per env step, the pad contact-force magnitude (the grip force) and the
              object height are recorded. Everything is read back once, at the end.

PENETRATION DEFINITION: point contact, ``phi = dot(x1 - x0, n) - margin0 - margin1``
with witness points transformed by the tick's body poses — the same expression the
ICF patch builder evaluates. Penetration is ``-phi`` where ``phi < 0``. This probe is
for the point-contact arms (icf, mujoco-injected); it does not define a hydroelastic
penetration.

SAMPLING: every solver tick of every accepted step. The fixed-step arms reject
nothing, so tick sampling IS accepted-step sampling and no downsampling proof is
owed. CUDA graph capture is disabled for the probe (the tick hook and per-tick
buffer indexing are host-side); this probe measures penetration, never wall time.

USAGE (from the IsaacLab root; single lines only)
  export VIRTUAL_ENV=$HOME/Documents/code/icra2027/.venv
  export P="scripts/probes/probe_grip_penetration.py"
  ./isaaclab.sh -p $P --arm icf --pose_check_only --viz none --out /tmp/pose_check.json
  ./isaaclab.sh -p $P --arm icf --checkpoint <model.pt> --num_envs 16 --steps 150 --viz none --out /tmp/grip.json
"""

from __future__ import annotations

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="IsaacContrib-Lift-Mug-Trossen-v0")
parser.add_argument("--arm", type=str, default="icf", choices=["mujoco", "icf"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150, help="Env steps (control ticks) of policy rollout.")
parser.add_argument("--checkpoint", type=str, default=None, help="rsl_rl model_*.pt to drive the rollout.")
parser.add_argument(
    "--pose_depths",
    type=float,
    nargs="+",
    default=[200e-6, 1e-3, 5e-3],
    help="Known overlaps [m] for the frozen-pose instrument validation.",
)
parser.add_argument("--pose_check_only", action="store_true", help="Run only the instrument validation.")
parser.add_argument(
    "--pose_tol_rel", type=float, default=0.05, help="Relative tolerance on the pose-check depth match."
)
parser.add_argument(
    "--pose_tol_abs", type=float, default=5e-6, help="Absolute tolerance floor [m] on the pose-check depth match."
)
parser.add_argument("--object_asset", type=str, default="object")
parser.add_argument("--slab_asset", type=str, default="table_guard")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import warp as wp  # noqa: E402

# ------------------------------------------------------------------- census kernel


@wp.kernel(enable_backward=False)
def _pair_census(
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
    role_a: int,
    role_b: int,
    tick: int,
    n_pen: wp.array2d(dtype=wp.int32),
    depth_max: wp.array2d(dtype=wp.float32),
    depth_sum: wp.array2d(dtype=wp.float32),
):
    """Per-world penetration census for one role pair, written at row ``tick``.

    Same gap expression as ``probe_contact_compliance._contact_census`` (and the ICF
    patch builder): ``phi = dot(x1 - x0, n) - margin0 - margin1`` at the given poses.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = shape0[tid]
    s1 = shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    r0 = shape_role[s0]
    r1 = shape_role[s1]
    if not ((r0 == role_a and r1 == role_b) or (r0 == role_b and r1 == role_a)):
        return
    w = shape_world[s0]
    if w < 0:
        w = shape_world[s1]
    if w < 0:
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
        wp.atomic_add(n_pen, tick, w, 1)
        wp.atomic_max(depth_max, tick, w, -phi)
        wp.atomic_add(depth_sum, tick, w, -phi)


# ------------------------------------------------------------------- helpers


def _mesh_lowest_local_z(usd_path: str) -> float:
    """Lowest collision-mesh vertex z in the mug's body frame, straight off the USD.

    Independent of the contact census: this is the probe's reference geometry for the
    known-overlap validation. The collision pieces are authored with their vertices
    already in the body frame (convert_mug.py writes points verbatim under /Mug).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    z_min = None
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if "/collisions_" not in path or prim.GetTypeName() != "Mesh":
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        for p in pts:
            z = float(p[2])
            if z_min is None or z < z_min:
                z_min = z
    if z_min is None:
        raise RuntimeError(f"no collision mesh points found in {usd_path}")
    return z_min


def _stats(depth_max_np, n_pen_np):
    """Episode max / p99 / median over contacting samples, from [T, W] arrays."""
    import numpy as np

    mask = n_pen_np > 0
    if not mask.any():
        return {"episode_max": None, "p99": None, "median_contacting": None, "contacting_fraction": 0.0}
    d = depth_max_np[mask]
    return {
        "episode_max": float(d.max()),
        "p99": float(np.percentile(d, 99)),
        "median_contacting": float(np.median(d)),
        "contacting_fraction": float(mask.mean()),
    }


# ------------------------------------------------------------------- main


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        apply_solver_choice(env_cfg, args_cli.arm)
        # The tick hook and its per-tick host-side buffer indexing cannot live inside
        # a captured graph; this probe never measures wall time, so capture is off.
        env_cfg.sim.physics.use_cuda_graph = False
        # One uninterrupted rollout: the episode must outlast the probe.
        control_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        env_cfg.episode_length_s = (args_cli.steps + 10) / control_hz
        ccfg = getattr(env_cfg.sim.physics, "collision_cfg", None)
        if ccfg is not None:
            scale = max(args_cli.num_envs / 8192.0, 1.0 / 512.0)
            ccfg.rigid_contact_max = max(65536, int(ccfg.rigid_contact_max * scale))
            ccfg.max_triangle_pairs = max(1_000_000, int(ccfg.max_triangle_pairs * scale))

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped

        from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
        from isaaclab_newton.physics.newton_manager import NewtonManager

        solver = NewtonManager._solver
        model = NewtonManager._model
        contacts = NewtonManager._contacts
        device = unwrapped.device
        n_worlds = int(model.world_count)

        result: dict = {
            "task": args_cli.task,
            "arm": args_cli.arm,
            "solver_class": type(solver).__name__,
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "seed": int(args_cli.seed),
            "sim_dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "checkpoint": args_cli.checkpoint,
        }
        params = getattr(solver, "params", None)
        if params is not None:
            result["icf_contact_stiffness"] = float(getattr(params, "contact_stiffness", float("nan")))

        # ---- shape roles: 1 = object, 2 = slab, 3 = finger pads -------------
        labels = list(getattr(model, "shape_label", []) or [])
        roles_np = np.zeros(max(len(labels), 1), dtype=np.int32)
        for i, lab in enumerate(labels):
            low = lab.lower()
            if "/object/" in low or low.endswith("/object"):
                roles_np[i] = 1
            elif "tableguard" in low:
                roles_np[i] = 2
            elif "follower_left_gripper" in low:
                roles_np[i] = 3
        result["shapes_object"] = int((roles_np == 1).sum())
        result["shapes_slab"] = int((roles_np == 2).sum())
        result["shapes_pad"] = int((roles_np == 3).sum())
        if min(result["shapes_object"], result["shapes_slab"], result["shapes_pad"]) == 0:
            print("[probe] WARNING: a shape role matched nothing. First 30 labels:")
            for lab in labels[:30]:
                print(f"    {lab}")

        shape_role = wp.array(roles_np, dtype=wp.int32, device=str(device))
        shape_world = model.shape_world
        shape_body_np = model.shape_body.numpy()

        # ---- geometry references --------------------------------------------
        slab_cfg = getattr(env_cfg.scene, args_cli.slab_asset)
        slab_top = float(slab_cfg.init_state.pos[2]) + float(slab_cfg.spawn.size[2]) / 2.0
        obj_cfg = getattr(env_cfg.scene, args_cli.object_asset)
        z_min_local = _mesh_lowest_local_z(obj_cfg.spawn.usd_path)
        result["slab_top_z"] = slab_top
        result["mesh_lowest_local_z"] = z_min_local

        env.reset()

        def census_args(role_a, role_b, tick, n_pen, depth_max, depth_sum, body_q):
            return dict(
                dim=int(contacts.rigid_contact_shape0.shape[0]),
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
                    body_q,
                    role_a,
                    role_b,
                    tick,
                    n_pen,
                    depth_max,
                    depth_sum,
                ],
                device=str(device),
            )

        # ================= pose-check: instrument validation =================
        # Teleport the mug so its lowest collision vertex sits a KNOWN depth below
        # the slab top. Nothing is stepped: one collide at the frozen state, one
        # census, and the reported max depth must equal the posed overlap.
        obj = unwrapped.scene[args_cli.object_asset]
        origins = unwrapped.scene.env_origins.cpu().numpy()
        mug_bodies = np.unique(shape_body_np[roles_np == 1])
        result["mug_bodies"] = [int(b) for b in mug_bodies]
        if len(mug_bodies) != n_worlds:
            print(f"[probe] WARNING: {len(mug_bodies)} mug bodies for {n_worlds} worlds")

        # body index -> world, to place each mug over its own env origin.
        shape_world_np = shape_world.numpy()
        body_world = np.full(int(mug_bodies.max()) + 1, -1)
        for s, b in enumerate(shape_body_np):
            if roles_np[s] == 1 and b >= 0:
                body_world[b] = int(shape_world_np[s])

        pose_rows = []
        state0 = NewtonManager._state_0
        bq_saved = state0.body_q.numpy().copy()
        for overlap in args_cli.pose_depths:
            bq = bq_saved.copy()
            for b in mug_bodies:
                w = body_world[b]
                # identity orientation: z-extent of the mesh is yaw-invariant anyway,
                # but the saved pose's orientation is kept to stay on the reset state.
                bq[b][2] = origins[w][2] + slab_top - overlap - z_min_local
            wp.copy(state0.body_q, wp.array(bq, dtype=wp.transform, device=str(device)))
            NewtonManager._collision_pipeline.collide(state0, contacts)
            n_pen = wp.zeros((1, n_worlds), dtype=wp.int32, device=str(device))
            d_max = wp.zeros((1, n_worlds), dtype=wp.float32, device=str(device))
            d_sum = wp.zeros((1, n_worlds), dtype=wp.float32, device=str(device))
            wp.launch(_pair_census, **census_args(1, 2, 0, n_pen, d_max, d_sum, state0.body_q))
            dm = d_max.numpy()[0]
            np_ = n_pen.numpy()[0]
            tol = max(args_cli.pose_tol_rel * overlap, args_cli.pose_tol_abs)
            err = np.abs(dm - overlap)
            ok = bool(np_.min() > 0 and err.max() <= tol)
            pose_rows.append(
                {
                    "posed_overlap": float(overlap),
                    "depth_max_mean": float(dm.mean()),
                    "depth_max_spread": float(dm.max() - dm.min()),
                    "abs_error_max": float(err.max()),
                    "n_pen_min": int(np_.min()),
                    "tol": float(tol),
                    "pass": ok,
                }
            )
            print(
                f"[pose-check] overlap {overlap * 1e6:9.1f} um -> census depth_max mean "
                f"{dm.mean() * 1e6:9.1f} um (max |err| {err.max() * 1e6:7.2f} um, n_pen_min {np_.min()}) "
                f"{'PASS' if ok else 'FAIL'}"
            )
        wp.copy(state0.body_q, wp.array(bq_saved, dtype=wp.transform, device=str(device)))
        result["pose_check"] = pose_rows
        result["pose_check_pass"] = bool(all(r["pass"] for r in pose_rows))

        if args_cli.pose_check_only or args_cli.checkpoint is None:
            if not args_cli.pose_check_only:
                print("[probe] no --checkpoint given; rollout skipped.")
            env.close()
            if args_cli.out:
                with open(args_cli.out, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"[probe] wrote {args_cli.out}")
            return 0 if result["pose_check_pass"] else 1

        # ================= rollout under the frozen policy ====================
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, filter_unsupported_rsl_rl_kwargs

        env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env_w, filter_unsupported_rsl_rl_kwargs(agent_cfg.to_dict()), log_dir=None, device=str(device)
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(device))

        ticks_max = (args_cli.steps + 2) * int(env_cfg.decimation)
        bufs = {}
        for name in ("pad_obj", "obj_slab"):
            bufs[name] = {
                "n_pen": wp.zeros((ticks_max, n_worlds), dtype=wp.int32, device=str(device)),
                "d_max": wp.zeros((ticks_max, n_worlds), dtype=wp.float32, device=str(device)),
                "d_sum": wp.zeros((ticks_max, n_worlds), dtype=wp.float32, device=str(device)),
            }
        tick_counter = {"i": 0, "armed": False}

        orig_substeps = NewtonMJWarpManager._run_solver_substeps.__func__

        def hooked(cls, contacts_arg):
            # Census FIRST: contacts were just collided at state_0, which is exactly
            # what the solver consumes this tick. After the solve, poses have moved.
            if tick_counter["armed"] and tick_counter["i"] < ticks_max:
                t = tick_counter["i"]
                b = bufs["pad_obj"]
                wp.launch(_pair_census, **census_args(3, 1, t, b["n_pen"], b["d_max"], b["d_sum"], state0.body_q))
                b = bufs["obj_slab"]
                wp.launch(_pair_census, **census_args(1, 2, t, b["n_pen"], b["d_max"], b["d_sum"], state0.body_q))
                tick_counter["i"] += 1
            return orig_substeps(cls, contacts_arg)

        NewtonMJWarpManager._run_solver_substeps = classmethod(hooked)

        obs, _ = env_w.reset()
        tick_counter["armed"] = True

        sensor = unwrapped.scene.sensors["pad_object_contact"]
        grip = torch.zeros((args_cli.steps, args_cli.num_envs), device=device)
        obj_z = torch.zeros((args_cli.steps, args_cli.num_envs), device=device)
        dones_total = 0
        with torch.inference_mode():
            for i in range(args_cli.steps):
                actions = policy(obs)
                obs, _, dones, _ = env_w.step(actions)
                dones_total += int(dones.sum().item())
                forces = sensor.data.force_matrix_w
                if forces is not None:
                    net = forces.torch.sum(dim=2)
                    mag = torch.linalg.vector_norm(net, dim=-1)
                    grip[i] = mag.reshape(args_cli.num_envs, -1).sum(dim=-1).nan_to_num(0.0)
                obj_z[i] = obj.data.root_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]
        tick_counter["armed"] = False
        NewtonMJWarpManager._run_solver_substeps = classmethod(orig_substeps)

        # ---- ONE readback ----------------------------------------------------
        ticks = tick_counter["i"]
        result["ticks_recorded"] = ticks
        result["episode_dones"] = dones_total
        grip_np = grip.cpu().numpy()
        obj_z_np = obj_z.cpu().numpy()
        for name, roles in (("pad_obj", "pads<->object"), ("obj_slab", "object<->slab")):
            b = bufs[name]
            dm = b["d_max"].numpy()[:ticks]
            npn = b["n_pen"].numpy()[:ticks]
            s = _stats(dm, npn)
            s["n_pen_mean_contacting"] = float(npn[npn > 0].mean()) if (npn > 0).any() else None
            result[name] = s

        pad_touch = grip_np > 0.1
        result["grip_force_N"] = {
            "mean_while_touching": float(grip_np[pad_touch].mean()) if pad_touch.any() else None,
            "p95_while_touching": float(np.percentile(grip_np[pad_touch], 95)) if pad_touch.any() else None,
            "max": float(grip_np.max()),
            "touching_fraction_of_steps": float(pad_touch.mean()),
        }
        result["object_z_max"] = float(obj_z_np.max())
        result["lifted_fraction_final_step"] = float((obj_z_np[-1] > 0.1).mean())

        # Adaptive-arm telemetry: controller counters, per-world dt, and the
        # L-inf argmax coordinate histogram (the step-size attribution
        # diagnostic), labeled by joint where the model names them.
        if hasattr(solver, "status_summary"):
            adaptive_info = solver.status_summary()
            adaptive_info["dt_per_world"] = [float(x) for x in solver.dt.numpy()]
            # Per-world demand, for the §2.1 break-even ratio (batch pays the
            # max over worlds; compaction converts it to the mean) and the
            # tail-env identity check (recurring tail worlds indict the scene
            # or reset distribution, not the solver).
            adaptive_info["accepted_per_world"] = [int(x) for x in solver.accepted_count.numpy()]
            adaptive_info["rejected_per_world"] = [int(x) for x in solver.rejected_count.numpy()]
            adaptive_info["floor_per_world"] = [int(x) for x in solver.floor_count.numpy()]
            hist = solver.argmax_hist.numpy()
            adaptive_info["argmax_hist"] = [int(x) for x in hist]
            try:
                names = getattr(model, "joint_key", None)
                if names is None:
                    jt = model.joint_type.numpy()
                    names = [f"joint{j}_type{int(t)}" for j, t in enumerate(jt)]
                names = list(names)
                q_start = model.joint_q_start.numpy()
                nq0 = int(getattr(solver, "nq_per_env", len(hist)))
                labels = ["?"] * nq0
                for j in range(len(names)):
                    s0 = int(q_start[j])
                    s1 = int(q_start[j + 1]) if j + 1 < len(q_start) else nq0
                    for c in range(s0, min(s1, nq0)):
                        labels[c] = names[j]
                by_joint: dict[str, int] = {}
                for c, n in enumerate(hist[:nq0]):
                    by_joint[labels[c]] = by_joint.get(labels[c], 0) + int(n)
                adaptive_info["argmax_by_joint"] = dict(
                    sorted(by_joint.items(), key=lambda kv: -kv[1])
                )
            except Exception as exc:  # labeling is best-effort; the raw hist stands
                adaptive_info["argmax_label_error"] = str(exc)
            try:
                adaptive_info["robot_joint_names"] = list(unwrapped.scene["robot"].joint_names)
            except Exception:
                pass
            result["adaptive"] = adaptive_info

        # Prediction cross-check: steady grip indentation F/(n*k) on the ICF arm.
        k = result.get("icf_contact_stiffness")
        po = result["pad_obj"]
        if k and po["median_contacting"] and result["grip_force_N"]["mean_while_touching"] and po["n_pen_mean_contacting"]:
            pred = result["grip_force_N"]["mean_while_touching"] / (po["n_pen_mean_contacting"] * k)
            result["predicted_grip_indentation"] = pred

        env.close()

    # ---- report --------------------------------------------------------------
    print("\n================ GRIP PENETRATION PROBE ================")
    print(f"  arm: {result['arm']} ({result['solver_class']})   envs: {result['num_envs']}   steps: {result['steps']}")
    print(f"  pose-check: {'PASS' if result['pose_check_pass'] else 'FAIL'}")
    if "pad_obj" in result:
        k = result.get("icf_contact_stiffness")
        if k:
            print(f"  ICF k: {k:.6g} N/m")
        for name, label in (("pad_obj", "pads<->object (GRIP)"), ("obj_slab", "object<->slab (REST)")):
            s = result[name]
            if s["episode_max"] is None:
                print(f"  {label}: no contacting samples")
                continue
            print(
                f"  {label}: max {s['episode_max'] * 1e6:9.1f} um   p99 {s['p99'] * 1e6:9.1f} um   "
                f"median(contacting) {s['median_contacting'] * 1e6:9.1f} um   "
                f"contacting {s['contacting_fraction'] * 100:5.1f}% of tick-world samples, "
                f"n_pen(mean) {s['n_pen_mean_contacting']:.2f}"
            )
        g = result["grip_force_N"]
        print(
            f"  grip force: mean {g['mean_while_touching']} N  p95 {g['p95_while_touching']} N  "
            f"max {g['max']:.2f} N  (touching {g['touching_fraction_of_steps'] * 100:.1f}% of steps)"
        )
        if result.get("predicted_grip_indentation"):
            print(
                f"  F/(n*k) predicted grip indentation: {result['predicted_grip_indentation'] * 1e6:.1f} um  "
                f"vs measured median {result['pad_obj']['median_contacting'] * 1e6:.1f} um"
            )
        print(f"  lifted at final step: {result['lifted_fraction_final_step'] * 100:.0f}% of envs")
        print(f"  episode dones during rollout: {result['episode_dones']} (should be 0)")
    print("========================================================\n")

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[probe] wrote {args_cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
