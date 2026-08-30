# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate and VALIDATE an IK pre-grasp bank pose for a Trossen scene.

Solves damped-least-squares IK for the arm joints so the TCP reaches a
requested hover point with the gripper's closing axis aligned to a requested
direction — using the simulator itself as the kinematics oracle (parallel
envs evaluate the numerical Jacobian in one step per iteration, so the
solve is exact for the scene's actual rig, offsets and all).

The solved pose is then subjected to the same existence proof the mug's
teleop pose passed before it was ever trusted: teleport, scripted close,
scripted raise — the object must be held at the end. A pose that fails is
printed with FAILED and must not be pasted into any cfg; the scene keeps
running home-only until a passing pose exists (generated fingers-down poses
for the mug measured kinematically infeasible once — generation is cheap,
validation is the gate).

On success prints a paste-ready GRASP_BANK_POSE dict and, with
``--xy_jacobian``, the 6x2 placement-tracking Jacobian from re-solves at
perturbed object placements.

USAGE (single line, from the IsaacLab root)
  ./isaaclab.sh -p scripts/probes/probe_generate_bank.py --task <task-id>
      --tcp_pos -0.037 0.0 0.27 --close_axis 1 0 0 --hover_back 0.02 --xy_jacobian
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--tcp_pos", type=float, nargs=3, required=True, help="Grasp point, env frame [m].")
parser.add_argument("--close_axis", type=float, nargs=3, default=(1.0, 0.0, 0.0), help="Closing axis, env frame.")
parser.add_argument("--approach_axis", type=float, nargs=3, default=(0.0, 0.0, -1.0), help="TCP approach, env frame.")
parser.add_argument("--hover_back", type=float, default=0.02, help="Hover offset back along approach [m].")
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--pos_tol", type=float, default=0.004, help="[m] IK position acceptance.")
parser.add_argument(
    "--axis_tol",
    type=float,
    default=0.15,
    help="Closing-axis alignment acceptance; the existence proof judges whether a tilted jaw seats the grip.",
)
parser.add_argument("--close_steps", type=int, default=15)
parser.add_argument("--raise_steps", type=int, default=45)
parser.add_argument("--raise_rad", type=float, default=0.4)
parser.add_argument("--held_dz", type=float, default=0.04, help="[m] raise height that counts as held.")
parser.add_argument("--xy_jacobian", action="store_true")
parser.add_argument("--xy_delta", type=float, default=0.01)
parser.add_argument("--q_init", type=float, nargs=6, default=None, help="IK seed for the six arm joints [rad] (default: home)")
parser.add_argument("--object_shift", type=float, nargs=3, default=None, help="shift the mug spawn by (dx, dy, dz) [m] for this probe")
parser.add_argument("--fix_joints", type=int, nargs="*", default=[], help="arm joint indices held at their seed value during the solve")

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
parser.set_defaults(visualizer=[])

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

ARM = [f"follower_left_joint_{i}" for i in range(6)]
GRIP = ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"]
FD_EPS = 0.02  # [rad] finite-difference step per joint


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab_tasks.utils.hydra import resolve_task_config
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        n_env = 1 + 2 * len(ARM)  # nominal + central differences per arm joint
        env_cfg.scene.num_envs = n_env
        apply_solver_choice(env_cfg, "icf")
        env_cfg.sim.physics.collision_cfg.rigid_contact_max = 100_000
        env_cfg.sim.physics.collision_cfg.max_triangle_pairs = 2_000_000
        env_cfg.observations.policy.enable_corruption = False
        # Kinematic queries only need the events cleared of arm scatter.
        if hasattr(env_cfg.events, "randomize_arm_start"):
            env_cfg.events.randomize_arm_start = None
        if hasattr(env_cfg.events, "reset_arm_grasp_bank"):
            env_cfg.events.reset_arm_grasp_bank = None
        if hasattr(env_cfg.events, "reset_reference"):  # the flip's reference-state RSI (2026-08-29): home only here
            env_cfg.events.reset_reference.params["home_fraction"] = 1.0
            env_cfg.events.reset_reference.params["approach_fraction"] = 0.0
            env_cfg.events.reset_reference.params["noise"] = 0.0
        # The bedrock curriculum references the bank event just nulled above;
        # it must go with it or the anneal term throws at every reset.
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "grow_approach"):
            env_cfg.curriculum.grow_approach = None

        if args_cli.object_shift is not None:
            pos = list(env_cfg.scene.object.init_state.pos)
            env_cfg.scene.object.init_state.pos = [pos[0] + args_cli.object_shift[0], pos[1] + args_cli.object_shift[1], pos[2] + args_cli.object_shift[2]]
            print(f"[bank-gen] mug spawn shifted to {env_cfg.scene.object.init_state.pos}", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        robot = u.scene["robot"]
        obj = u.scene["object"]
        arm_ids, _ = robot.find_joints(ARM, preserve_order=True)
        grip_ids, _ = robot.find_joints(GRIP, preserve_order=True)
        env.reset()
        q_home = robot.data.joint_pos.torch[0].clone()
        limits = robot.data.joint_pos_limits.torch[0]

        approach = np.array(args_cli.approach_axis, dtype=np.float64)
        approach /= np.linalg.norm(approach)
        target = np.array(args_cli.tcp_pos, dtype=np.float64) - args_cli.hover_back * approach
        close_axis = np.array(args_cli.close_axis, dtype=np.float64)
        close_axis /= np.linalg.norm(close_axis)

        def fk_batch(q_arm_batch: np.ndarray):
            """Write per-env arm joints, one physics step, read TCP per env.

            The step is the FK oracle: one 1/90 s step under PD moves the
            written pose by microradians, and the scene update refreshes the
            frame transformer from the stepped state (a forward() alone
            leaves the sensor stale -- measured as meter-scale TCP error).
            """
            q = q_home.unsqueeze(0).repeat(n_env, 1).clone()
            q[:, arm_ids] = torch.tensor(q_arm_batch, dtype=torch.float32, device=u.device)
            q[:, grip_ids] = 0.044  # full open
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
            ee = u.scene["ee_frame"]
            pos = ee.data.target_pos_w.torch[..., 0, :].cpu().numpy() - u.scene.env_origins.cpu().numpy()
            # Closing axis MEASURED as the pad-to-pad direction: assuming a
            # TCP-frame axis put the solver to work aligning a fictitious
            # frame (forensics: pads separated along world Z while the
            # "aligned" axis read 16 degrees off X).
            pads = robot.data.body_pos_w.torch[:, pad_body_ids]
            sep = (pads[:, 0] - pads[:, 1]).cpu().numpy()
            sep = sep / np.linalg.norm(sep, axis=1, keepdims=True)
            l6 = robot.data.body_pos_w.torch[:, link6_id].cpu().numpy() - u.scene.env_origins.cpu().numpy()
            tool = pos - l6
            tool = tool / np.linalg.norm(tool, axis=1, keepdims=True)
            return pos, sep, tool

        pad_body_ids, _ = robot.find_bodies("follower_left_gripper_.*", preserve_order=True)
        link6_id = robot.find_bodies("follower_left_link_6")[0][0]

        q = q_home[arm_ids].cpu().numpy().astype(np.float64)
        if args_cli.q_init is not None:
            q = np.array(args_cli.q_init, dtype=np.float64)
            print(f"[bank-gen] IK seed {q.round(3).tolist()}", flush=True)

        def solve(target_pos: np.ndarray, q0: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
            qq = (q if q0 is None else q0).copy()
            err_p = err_a = np.inf
            for _ in range(args_cli.iters):
                batch = np.repeat(qq[None, :], n_env, axis=0)
                for j in range(len(ARM)):
                    batch[1 + 2 * j, j] += FD_EPS
                    batch[2 + 2 * j, j] -= FD_EPS
                pos, seps, tools = fk_batch(batch)
                e_pos = target_pos - pos[0]
                d0 = seps[0]
                e_axis = close_axis - np.dot(close_axis, d0) * d0  # align, sign-free
                # TOOL AXIS (finger direction) must point along the approach, sign-SENSITIVE:
                # with only the jaw axis constrained the fingers were free to point down and
                # closed 5 cm behind a side-on handle bar (2026-08-28 flip forensics).
                e_tool = approach - tools[0]
                err_p, err_a = float(np.linalg.norm(e_pos)), float(max(np.linalg.norm(e_axis), np.linalg.norm(e_tool)))
                if err_p < args_cli.pos_tol and err_a < args_cli.axis_tol:
                    break
                J = np.zeros((9, len(ARM)))
                for j in range(len(ARM)):
                    J[:3, j] = (pos[1 + 2 * j] - pos[2 + 2 * j]) / (2 * FD_EPS)
                    J[3:6, j] = (seps[1 + 2 * j] - seps[2 + 2 * j]) / (2 * FD_EPS)
                    J[6:, j] = (tools[1 + 2 * j] - tools[2 + 2 * j]) / (2 * FD_EPS)
                e = np.concatenate([e_pos, 0.12 * e_axis, 0.12 * e_tool])
                lam = args_cli.damping
                for fj in args_cli.fix_joints:
                    J[:, fj] = 0.0
                dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(9), e)
                qq = qq + np.clip(dq, -0.2, 0.2)
                lo = limits[arm_ids, 0].cpu().numpy()
                hi = limits[arm_ids, 1].cpu().numpy()
                qq = np.clip(qq, lo + 1e-3, hi - 1e-3)
            return qq, err_p, err_a

        q_sol, err_p, err_a = solve(target)
        print(f"[bank-gen] IK: pos err {err_p * 1000:.1f} mm, axis err {err_a:.3f}")
        print(f"[bank-gen] q_sol (hover) = {[round(float(v), 4) for v in q_sol]}")
        if err_p > args_cli.pos_tol or err_a > args_cli.axis_tol:
            lo = limits[arm_ids, 0].cpu().numpy(); hi = limits[arm_ids, 1].cpu().numpy()
            at = ["lo" if q_sol[i] - lo[i] < 0.02 else ("hi" if hi[i] - q_sol[i] < 0.02 else "--") for i in range(len(ARM))]
            print(f"[bank-gen]   q_sol {np.round(q_sol, 3)} limits lo {np.round(lo, 2)} hi {np.round(hi, 2)} at-limit {at}")
            print("[bank-gen] FAILED: IK did not reach the hover point. No pose emitted.")
            env.close()
            return 1
        # Second solve AT the grasp point (6 mm standoff): the existence proof
        # descends hover -> grasp before closing -- closing at a hover tens of
        # millimeters out just grabs air (measured: object dz exactly 0).
        grasp_target = np.array(args_cli.tcp_pos, dtype=np.float64) - 0.006 * approach
        q_grasp, err_g, _ = solve(grasp_target, q0=q_sol)
        print(f"[bank-gen] IK grasp point: pos err {err_g * 1000:.1f} mm")
        print(f"[bank-gen] q_grasp = {[round(float(v), 4) for v in q_grasp]}")
        if err_g > args_cli.pos_tol:
            print("[bank-gen] FAILED: grasp point unreachable. No pose emitted.")
            env.close()
            return 1

        # ---- existence proof: teleport, clearance check, close, raise -----
        q_full = q_home.unsqueeze(0).repeat(n_env, 1).clone()
        q_full[:, arm_ids] = torch.tensor(q_sol, dtype=torch.float32, device=u.device)
        q_full[:, grip_ids] = 0.044
        robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
        # CLEARANCE GATE: the pose must be a hover -- zero pad-object contact
        # at spawn. A pose that starts touching (or inside) the object can
        # pass the hold check through penetration-seated grip, which is the
        # artifact class this program exists to expose, not a validation.
        def _scale_vec(t):
            sc = getattr(t, "_scale", None)
            if torch.is_tensor(sc):
                sc = sc.reshape(-1, sc.shape[-1])[0] if sc.dim() > 1 else sc.reshape(-1)
                return sc.cpu().numpy() if sc.numel() > 1 else float(sc[0])
            return float(t.cfg.scale)

        am0 = u.action_manager
        term0 = am0.get_term("arm_action")
        arm_dim0 = term0.action_dim
        scale0 = _scale_vec(term0)
        q_def0 = robot.data.default_joint_pos.torch[0, arm_ids].cpu().numpy()
        hold = torch.zeros(n_env, am0.total_action_dim, device=u.device)
        # HOLD THE HOVER: zero action on this offset-based term commands the
        # DEFAULT pose, and three steps of the PD lurching home from the
        # hover measured 1.5 kN of mid-flight contact that the hover itself
        # never makes.
        hold[:, :arm_dim0] = torch.tensor((q_sol - q_def0) / scale0, dtype=torch.float32, device=u.device)
        with torch.inference_mode():
            for _ in range(3):
                env.step(hold)
        cf = u.scene.sensors["pad_object_contact"].data.force_matrix_w
        f0 = 0.0
        if cf is not None:
            f0 = float(torch.linalg.vector_norm(cf.torch.sum(dim=2), dim=-1).nan_to_num(0.0).max())
        pad_ids, pad_names = robot.find_bodies("follower_left_gripper_.*")
        pp = robot.data.body_pos_w.torch[0, pad_ids] - obj.data.root_pos_w.torch[0:1]
        for nm, d in zip(pad_names, pp.cpu().numpy()):
            print(f"[bank-gen]   pad {nm} rel plate: [{d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}] m")
        ee0 = u.scene["ee_frame"].data.target_pos_w.torch[0, 0] - u.scene.env_origins[0]
        rq = robot.data.root_quat_w.torch[0].cpu().numpy(); oq = obj.data.root_quat_w.torch[0].cpu().numpy(); op = (obj.data.root_pos_w.torch[0] - u.scene.env_origins[0]).cpu().numpy()
        from scipy.spatial.transform import Rotation as _R
        h_b = np.array([0.062, 0.0, 0.058])
        h_xyzw = op + _R.from_quat(oq).apply(h_b)                      # buffer read as (x,y,z,w)
        h_wxyz = op + _R.from_quat([oq[1], oq[2], oq[3], oq[0]]).apply(h_b)  # buffer read as (w,x,y,z)
        print(f"[bank-gen]   robot root quat (buffer order) {rq.round(3)}  -> identity is (0,0,0,1) if xyzw, (1,0,0,0) if wxyz")
        print(f"[bank-gen]   mug root pos {op.round(3)} quat {oq.round(3)}; handle_middle if xyzw {h_xyzw.round(3)} | if wxyz {h_wxyz.round(3)}")
        print(f"[bank-gen]   TCP env-frame: {ee0.cpu().numpy().round(3)}")
        print(f"[bank-gen] spawn clearance: max pad-object force {f0:.3f} N")
        if f0 > 0.1:
            print("[bank-gen] FAILED: pose touches the object at spawn (not a hover). No pose emitted.")
            env.close()
            return 1
        obj_spawn = obj.data.root_pos_w.torch.clone()
        am = u.action_manager
        term = am.get_term("arm_action")
        arm_dim = term.action_dim
        scale = _scale_vec(term)
        # Retarget the action OFFSETS to the hover, exactly as the bank event
        # does: home-anchored offsets make every scripted target a clipped
        # (q - default)/scale command, and the arm sags toward home instead
        # of holding the pose (measured: the whole close ran off-pose, dz 0).
        q_sol_t = torch.tensor(q_sol, dtype=torch.float32, device=u.device)
        off = getattr(term, "_offset", None)
        if torch.is_tensor(off):
            off[:] = q_sol_t.unsqueeze(0)
        goff = getattr(am.get_term("gripper_action"), "_offset", None)
        if torch.is_tensor(goff):
            goff[:] = 0.044
        act = torch.zeros(n_env, am.total_action_dim, device=u.device)
        approach_steps = 25
        with torch.inference_mode():
            for step in range(approach_steps + args_cli.close_steps + args_cli.raise_steps):
                # All arm commands are DELTAS from the retargeted hover.
                a = min(step / approach_steps, 1.0)
                q_cmd = (1.0 - a) * q_sol + a * q_grasp
                if step == 0:
                    q_int = np.zeros_like(q_cmd)
                q_meas = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy()
                if step < approach_steps + args_cli.close_steps:
                    q_int = np.clip(q_int + 0.5 * (q_cmd - q_meas), -0.3, 0.3)  # integral action: PD sag under gravity
                act[:, :arm_dim] = torch.tensor((q_cmd + q_int - q_sol) / scale, dtype=torch.float32, device=u.device)
                if step >= approach_steps:
                    act[:, arm_dim:] = -1.0  # close
                if step >= approach_steps + args_cli.close_steps:
                    ramp = (step - approach_steps - args_cli.close_steps + 1) / args_cli.raise_steps
                    j1 = term._joint_names.index("follower_left_joint_1")
                    act[:, j1] = (q_grasp[j1] + q_int[j1] - q_sol[j1] - args_cli.raise_rad * ramp) / (scale[j1] if hasattr(scale, '__len__') else scale)
                env.step(act)
                if step % 10 == 9:
                    tcp = u.scene["ee_frame"].data.target_pos_w.torch[0, 0] - u.scene.env_origins[0]
                    car = robot.data.joint_pos.torch[0, grip_ids].cpu().numpy()
                    cf2 = u.scene.sensors["pad_object_contact"].data.force_matrix_w
                    fmax = 0.0
                    if cf2 is not None:
                        fmax = float(torch.linalg.vector_norm(cf2.torch.sum(dim=2), dim=-1).nan_to_num(0.0).max())
                    qa = robot.data.joint_pos.torch[0, arm_ids].cpu().numpy()
                    qerr = np.abs(qa - q_cmd)
                    tgt = am.get_term("arm_action").processed_actions[0].cpu().numpy() if hasattr(am.get_term("arm_action"), "processed_actions") else None
                    def _fm(n):
                        if n not in u.scene.sensors:
                            return None
                        fmw = u.scene.sensors[n].data.force_matrix_w
                        return 0.0 if fmw is None else round(float(torch.linalg.vector_norm(fmw.torch.sum(dim=2), dim=-1).nan_to_num(0.0)[0].max()), 2)
                    fh = [_fm(nm) for nm in ("pad_left_handle", "pad_right_handle", "pad_body_contact")]
                    try:
                        tq = robot.data.applied_torque.torch[0, arm_ids].cpu().numpy().round(2)
                    except Exception:
                        tq = None
                    print(
                        f"[proof {step:03d}] torque {tq} |"
                        f" tcp {tcp.cpu().numpy().round(3)} carriage {car[0] * 1000:.1f}/{car[1] * 1000:.1f} mm"
                        f" pad force {fmax:7.2f} N  obj dz {float((obj.data.root_pos_w.torch[:, 2] - obj_spawn[:, 2]).max()) * 1000:+.1f} mm"
                        f"  |q-q_cmd| max {qerr.max():.3f} rad (j{int(qerr.argmax())})  handleL/R/body {fh}"
                    )
                    if step == approach_steps + args_cli.close_steps - 1:
                        print(f"[proof]   PD-sag integral at the grasp: q_int {np.round(q_int, 4)}  -> compensated bank {np.round(q_sol + q_int, 4)}")
                    if step == 9 and tgt is not None:
                        print(f"[proof]   q_cmd {np.round(q_cmd, 3)}\n[proof]   q_now {np.round(qa, 3)}\n[proof]   processed target {np.round(tgt, 3)}")
        dz = float((obj.data.root_pos_w.torch[:, 2] - obj_spawn[:, 2]).max())
        _q = obj.data.root_quat_w.torch[0]
        print(f"[bank-gen] object up_cos after the raise (env 0): {float(1 - 2 * (_q[0] ** 2 + _q[1] ** 2)):+.3f}  (+1 upright, -1 inverted)")
        def _fmag(name):
            if name not in u.scene.sensors:
                return None
            fm = u.scene.sensors[name].data.force_matrix_w
            if fm is None:
                return 0.0
            return float(torch.linalg.vector_norm(fm.torch.sum(dim=2), dim=-1).nan_to_num(0.0)[0].max())
        fl, fr, fb = _fmag("pad_left_handle"), _fmag("pad_right_handle"), _fmag("pad_body_contact")
        if fl is not None:
            print(f"[bank-gen] end-of-raise forces: left pad on HANDLE {fl:.2f} N, right pad on HANDLE {fr:.2f} N, pads on BODY {(fb if fb is not None else float("nan")):.2f} N -> "
                  f"{'HANDLE-ONLY PINCH' if (fl > 0.5 and fr > 0.5 and fb < 1.0) else 'NOT a clean handle pinch'}")
        held = dz > args_cli.held_dz
        print(f"[bank-gen] existence proof: max object dz {dz * 1000:.1f} mm -> {'HELD' if held else 'FAILED'}")
        if not held:
            print("[bank-gen] FAILED: pose does not produce a physical hold. No pose emitted.")
            env.close()
            return 1

        names = ARM + GRIP
        vals = list(q_sol) + [0.044, 0.044]
        print("GRASP_BANK_POSE = {")
        for n, v in zip(names, vals):
            print(f'    "{n}": {v:.4f},')
        print("}")

        if args_cli.xy_jacobian:
            rows = np.zeros((len(ARM), 2))
            for k, d in enumerate((np.array([args_cli.xy_delta, 0, 0]), np.array([0, args_cli.xy_delta, 0]))):
                qp, ep, _ = solve(target + d)
                qm, em, _ = solve(target - d)
                if max(ep, em) > args_cli.pos_tol:
                    print("[bank-gen] xy_jacobian FAILED at perturbed placement; omit tracking.")
                    break
                rows[:, k] = (qp - qm) / (2 * args_cli.xy_delta)
            else:
                print("BANK_POSE_XY_JACOBIAN = [")
                for r in rows:
                    print(f"    [{r[0]:+.6f}, {r[1]:+.6f}],")
                print("]")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
