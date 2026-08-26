# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive grasp-pose lab for IsaacContrib-Lift-Mug-Trossen-v0.

Drag the six arm joints (and the gripper) on SLIDERS in the Newton viewer's side
panel; the pose is HELD every step, so the arm stays where you put it instead of
springing back. Read the joint vector live, and save it with one button.

Same control model as the G1 ``g1_spatula_lift/assets/pose_lab.py``: an owned
value dict is the single source of truth, the panel edits it in immediate mode,
and the step loop rewrites the PD targets from it.

Run::

    VIRTUAL_ENV=$PWD/../IsaacLabRubato/.venv MUG_COLLISION=hull ./isaaclab.sh -p \
      source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_plate_rack/pose_lab.py --viz newton

    ... --selftest      # headless: drive the panel through a stub, non-zero on failure
    ... NEWTON_SOLVER=icf   # ICF physics instead of MuJoCo-Warp

``S`` (or the SAVE button) writes ``grasp_pose_authored.py``, shaped as the
``pose`` dict ``mdp.reset_arm_reverse_curriculum`` takes -- the constant the env cfg
documents as "the grasp-bank reset pose" but never defines.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import sys
import termios
import tty

# imgui_bundle must load BEFORE the simulation stack. Its native module binds
# GLFW symbols against whichever ``libglfw.so.3`` is already in the process, and
# the sim stack can drag in a Wayland-only GLFW that lacks ``glfwGetX11Window``;
# the import then fails inside the Newton viewer, which SILENTLY disables its
# whole ImGui layer -- scene renders, no side panel. (Lifted verbatim from the
# G1 pose_lab, which paid for this discovery.)
with contextlib.suppress(ImportError):
    import imgui_bundle  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Author a pre-grasp pose for the Trossen plate-pick task.")
parser.add_argument(
    "--selftest", action="store_true", help="Drive every control surface headlessly; exit non-zero on failure."
)
parser.add_argument(
    "--max_target_speed", type=float, default=1.5, help="rad/s cap between slider value and PD target (0 = snap)."
)
parser.add_argument(
    "--task", type=str, default="IsaacContrib-PlatePick-Trossen-v0", help="Gym id of the scene to open (any rig-family task)."
)
parser.add_argument(
    "--solver",
    type=str,
    default="icf",
    choices=["icf", "icf-adaptive", "mujoco", "mujoco-adaptive", "sap", "sap-adaptive"],
    help="Physics preset for the session (training's --solver): fixed-step icf by default.",
)
parser.add_argument(
    "--object-at-goal",
    dest="object_at_goal",
    action="store_true",
    help="After reset, drop the object at the task's commanded GOAL pose (position + rpy) instead of its spawn.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.selftest:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (  # noqa: E402
    ARM_JOINTS,
    GRIPPER_JOINT,
    GRIPPER_JOINT_R,
)
from isaaclab_tasks.contrib.trossen_plate_rack.trossen_plate_rack_env_cfg import (  # noqa: E402
    PLATE_BANK_POSE as GRASP_BANK_POSE,
)

# The plate's settled REST height above its env origin: read from the task
# cfg at session start (see below), so scene revisions cannot strand it.
OBJECT_REST_Z = None  # resolved from cfg.scene.object.init_state.pos[2]
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# 1:1 WITH TRAINING: the training launches export these; a bare lab launch
# would otherwise build a DIFFERENT scene (mesh mug, no rails) and any pose
# authored in it would be authored against geometry the task never sees.
# setdefault, so an explicit override still wins.
os.environ.setdefault("MUG_COLLISION", "hull")
os.environ.setdefault("PLATE_COLLISION", "hull")
os.environ.setdefault("RACK_COLLISION", "slabs")
os.environ.setdefault("TROSSEN_RAILS", "1")

TASK = args_cli.task
_RENDER = not args_cli.selftest  # selftest is the only path that must not draw
OUT_PATH = os.path.join(os.path.dirname(__file__), "grasp_pose_authored.py")

# The task's collision caps are GLOBAL and sized for 4096 envs (8M contacts /
# 192M triangle pairs ~ 15 GB); a one-env posing session OOMs a 16 GB card on a
# 32 MB allocation without this. Scaled down HERE ONLY -- the task cfg keeps its
# exact caps so training and the adaptive-vs-fixed comparison are unaffected.
POSE_RIGID_CONTACT_MAX = 200_000
POSE_MAX_TRIANGLE_PAIRS = 4_000_000

# Finger gap at carriage=0 (assets docstring): the jaws bottom out here.
CLOSED_GAP_M = 0.0483

# Sessions START at the Trossen READY pose: the vendor controller's home position
# (joints 0, pi/2, pi/2, 0, 0, 0; carriages open at 0.044), taken verbatim from Trossen's
# own demo scripts in assets.py -- the pose the driver brings the arm to from sleep
# before any operation. That IS the articulation's init_state, so it is read from
# default_joint_pos rather than restated here, and cannot drift from the rig config.
#
# POSE_START=bank starts from GRASP_BANK_POSE instead -- the teleop-authored pre-grasp
# the task's own reset event uses. It is IMPORTED, never copied: a local copy would
# silently stop matching the scene the moment the constant was retuned, and then this
# tool would be showing a pose the env no longer resets to.


def _t(x):
    """Scene fields are sometimes wrapped (``.torch``), sometimes plain tensors."""
    return x.torch if hasattr(x, "torch") else x


def _short(name: str) -> str:
    """Compact slider label: follower_left_joint_3 -> joint_3."""
    return name.replace("follower_left_", "")


class _StubImgui:
    """Minimal ImGui stand-in so --selftest can drive the panel with no GUI.

    ImGui is immediate mode: a widget reports interaction through its RETURN
    VALUE, so "click" means :meth:`button` returns True and "drag" means
    :meth:`slider_float` reports a change. Feeding this to the real panel
    callback exercises the same code the mouse would.
    """

    def __init__(self, click: str | None = None, drag: tuple[str, float] | None = None):
        self._click, self._drag = click, drag
        self.buttons: list[str] = []
        self.sliders: list[str] = []
        self.texts: list[str] = []

    def separator(self) -> None:
        """The stub draws nothing."""

    def text(self, label: str) -> None:
        self.texts.append(label)

    def button(self, label: str) -> bool:
        self.buttons.append(label)
        return label == self._click

    def slider_float(self, label: str, value: float, lo: float, hi: float):
        self.sliders.append(label)
        if self._drag is not None and self._drag[0] == label:
            return True, min(max(self._drag[1], lo), hi)
        return False, value


class RawKeys:
    """Non-blocking single-keypress stdin. No-op when stdin is not a tty."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else None
        self._saved = None

    def __enter__(self):
        if self.enabled:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self.enabled and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self) -> list[str]:
        keys = []
        while self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch)
        return keys


def build_env(num_envs: int = 1):
    from isaaclab_tasks.utils.physics_presets import apply_solver_choice

    cfg = parse_env_cfg(TASK, num_envs=num_envs)
    # Same solver as the training runs (--solver icf): the realized pose is a
    # PD equilibrium under the solver, so posing under a different one would
    # hand back vectors authored against different sag.
    apply_solver_choice(cfg, args_cli.solver)
    cfg.sim.physics.collision_cfg.rigid_contact_max = POSE_RIGID_CONTACT_MAX
    cfg.sim.physics.collision_cfg.max_triangle_pairs = POSE_MAX_TRIANGLE_PAIRS

    # WHY THE NEWTON WINDOW NEVER OPENS WITHOUT THIS: the task's __post_init__ sets
    # sim.default_visualizer_cfg = NewtonVisualizerCfg(headless=True, ...) for offline
    # video recording. SimulationContext builds a fresh NewtonVisualizerCfg for
    # '--viz newton', then _apply_default_visualizer_cfg copies over every field still
    # at its factory value -- dragging headless=True across. The viewer becomes a
    # pyglet HeadlessWindow (EGL) and no window maps, silently (the "No display found"
    # warning only fires when headless was NOT already requested). Setting
    # headless=False on the RESOLVED cfg does not survive, because that copier only
    # preserves fields DIFFERING from the factory default -- and False is the default.
    # Do NOT set default_visualizer_cfg = None: validate_config() writes .eye/.lookat.
    if cfg.sim.default_visualizer_cfg is not None:
        cfg.sim.default_visualizer_cfg.headless = False

    # PLATE CONTACTS ON by default (POSE_MUG_GHOST=1 re-enables the ghost):
    # the pinch must be feelable -- pads engage the physical plate and the
    # panel's plate sliders re-place it between attempts.
    if os.environ.get("POSE_MUG_GHOST", "0") != "0":
        for _c in cfg.scene.object.spawn.collision_props:
            if hasattr(_c, "collision_enabled"):
                _c.collision_enabled = False
    return gym.make(TASK, cfg=cfg).unwrapped


def main() -> int:  # noqa: C901
    env = build_env()
    robot = env.scene["robot"]
    device = env.device

    arm_ids, arm_names = robot.find_joints([ARM_JOINTS], preserve_order=True)
    grip_ids, grip_names = robot.find_joints([GRIPPER_JOINT, GRIPPER_JOINT_R], preserve_order=True)
    editable = list(arm_names)

    env.reset()
    for _ in range(2):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    lim_all = _t(robot.data.soft_joint_pos_limits)[0]
    lim = {n: (float(lim_all[i][0]), float(lim_all[i][1])) for n, i in zip(arm_names, arm_ids)}

    # The single source of truth. The panel edits this; the loop commands it.
    grip_open = float(_t(robot.data.default_joint_pos)[0, grip_ids[0]])
    _use_bank = os.environ.get("POSE_START", "ready") == "bank"
    if _use_bank:
        # Clamped, not trusted: the constant is retuned by hand, so an unclamped write
        # could land outside the soft limits and be silently rejected by the solver.
        vals = {n: min(max(GRASP_BANK_POSE[n], lim[n][0]), lim[n][1]) for n in arm_names}
        vals[GRIPPER_JOINT] = min(max(float(GRASP_BANK_POSE[GRIPPER_JOINT]), 0.0), grip_open)
    else:
        # From default_joint_pos (the authored init_state), NOT the measured joint_pos:
        # by the time we read it the arm has already sagged a hair under gravity
        # (measured j1 = 1.5641 vs the authored pi/2 = 1.5708), and starting from the
        # sag would silently make "ready" mean "ready, minus whatever gravity took".
        _dflt = _t(robot.data.default_joint_pos)[0]
        vals = {n: float(_dflt[i]) for n, i in zip(arm_names, arm_ids)}
        vals[GRIPPER_JOINT] = float(_dflt[grip_ids[0]])
    # Teleport there rather than letting the PD crawl over from home.
    _start_q = _t(robot.data.joint_pos).clone()
    for n, i in zip(arm_names, arm_ids):
        _start_q[0, i] = vals[n]
    for gi in grip_ids:
        _start_q[0, gi] = vals[GRIPPER_JOINT]
    robot.write_joint_position_to_sim_index(position=_start_q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(_t(robot.data.joint_vel)))
    # Stage the PD targets too: a teleport alone leaves the targets at their old values,
    # so the very next step sags the arm off the pose it was just placed in.
    robot.set_joint_position_target_index(
        target=torch.tensor([[vals[n] for n in arm_names]], device=device, dtype=torch.float32),
        joint_ids=arm_ids,
    )
    robot.set_joint_position_target_index(
        target=torch.full((1, len(grip_ids)), vals[GRIPPER_JOINT], device=device, dtype=torch.float32),
        joint_ids=grip_ids,
    )
    env.scene.write_data_to_sim()
    for _ in range(2):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    # cmd trails vals through a rate limit, so a slider yanked across its range
    # does not command a step the PD answers with a whip-crack.
    cmd = dict(vals)

    _ghost = os.environ.get("POSE_MUG_GHOST", "0") != "0"
    obj = env.scene["object"]
    if args_cli.object_at_goal:
        # The task's commanded goal pose (the range minima: every rig task pins ONE
        # goal), placed as the object's pose. Contacts stay on, so a wrong goal shows
        # itself: the object falls, jams, or rests exactly where it was put.
        from isaaclab.utils.math import quat_from_euler_xyz  # noqa: PLC0415

        rg = env.cfg.commands.object_pose.ranges
        org = _t(env.scene.env_origins)[0]
        g_pos = torch.tensor([[rg.pos_x[0], rg.pos_y[0], rg.pos_z[0]]], device=device, dtype=torch.float32) + org
        eul = torch.tensor([[rg.roll[0], rg.pitch[0], rg.yaw[0]]], device=device, dtype=torch.float32)
        g_quat = quat_from_euler_xyz(eul[:, 0], eul[:, 1], eul[:, 2])  # (x, y, z, w)
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([g_pos, g_quat], dim=-1))
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(_t(obj.data.root_vel_w)))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        print(
            f"[pose_lab] object placed at the GOAL pose: env pos=({rg.pos_x[0]:+.4f}, {rg.pos_y[0]:+.4f}, {rg.pos_z[0]:+.4f})"
            f" rpy=({rg.roll[0]:+.4f}, {rg.pitch[0]:+.4f}, {rg.yaw[0]:+.4f})"
        )
    # Pin to the SPAWN pose, not wherever it is now: with collisions off the ghost sinks
    # a little on every step before the first pin, and freezing that leaves the mug
    # buried in the slab (measured z = 0.0089 against a 0.021 spawn).
    mug_pose0 = _t(obj.data.root_pose_w).clone()
    if not args_cli.object_at_goal:
        mug_pose0[0, 2] = float(_t(env.scene.env_origins)[0][2]) + float(env.cfg.scene.object.init_state.pos[2])
    _org = _t(env.scene.env_origins)[0]
    from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: PLC0415

    # Panel-editable OBJECT POSE (env frame): position AND roll/pitch/yaw.
    # HOLD ON  -> the object is frozen at the slider pose every step (sim paused
    #             for the object; the arm still moves). Adjust freely.
    # HOLD OFF -> physics owns the object: it falls, settles, rests.
    # PRINT OBJECT POSE reads the LIVE pose (where it actually rests);
    # SNAP pulls the sliders to that live pose so it can be held again.
    _q0 = mug_pose0[0, 3:7].unsqueeze(0)
    _r0, _p0, _y0 = euler_xyz_from_quat(_q0)
    obj_pose = {
        "pos": [float(mug_pose0[0, 0] - _org[0]), float(mug_pose0[0, 1] - _org[1]), float(mug_pose0[0, 2] - _org[2])],
        "rpy": [float(_r0[0]), float(_p0[0]), float(_y0[0])],
    }
    hold = {"on": True, "released": False}
    _watch = {"n": 0}
    plate_pos = obj_pose["pos"]  # alias kept for the selftest's slider names

    def _slider_pose_tensor() -> torch.Tensor:
        pos = torch.tensor([[_org[k] + obj_pose["pos"][k] for k in range(3)]], device=device, dtype=torch.float32)
        e = torch.tensor([obj_pose["rpy"]], device=device, dtype=torch.float32)
        q = quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2])  # (x, y, z, w)
        return torch.cat([pos, q], dim=-1)

    def place_plate() -> None:
        pose = _slider_pose_tensor()
        mug_pose0.copy_(pose)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(_t(obj.data.root_vel_w)))

    def live_object_pose() -> tuple[list[float], list[float], list[float]]:
        """(pos_env, rpy, quat_xyzw) of the object as simulated right now."""
        pw = _t(obj.data.root_pos_w)[0] - _org
        q = _t(obj.data.root_quat_w)[0].unsqueeze(0)
        r, pch, y = euler_xyz_from_quat(q)
        return [float(v) for v in pw], [float(r[0]), float(pch[0]), float(y[0])], [float(v) for v in q[0]]

    def print_object_pose() -> None:
        pos, rpy, q = live_object_pose()
        msg = (
            f"OBJECT POSE (env frame, live): pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
            f"  rpy=({rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}) rad"
            f"  quat_xyzw=({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})"
            f"  hold={'ON' if hold['on'] else 'OFF'}"
        )
        sys.stdout.write("\r\n  " + msg + "\r\n")
        sys.stdout.flush()
        pending["status"] = msg

    def snap_sliders_to_object() -> None:
        pos, rpy, _ = live_object_pose()
        obj_pose["pos"][:] = pos
        obj_pose["rpy"][:] = rpy
        pending["status"] = "sliders snapped to the live object pose"

    def joint_dict() -> dict[str, float]:
        return {n: vals[n] for n in editable}

    _pad_ids, _pad_names = robot.find_bodies("follower_left_gripper_.*")

    def _pad_gauge() -> str:
        """Pad-vs-mug gauge from REALIZED body poses, so a vector cannot look
        right on screen while being somewhere else in numbers.

        Raw numbers only, honestly labeled: the pad rows read the pad MOUNT
        origins, which sit several cm above the rubber surfaces — classifying
        them against rim height mislabels a correct rim pinch as too high.
        The TCP row (finger midpoint) is the graspedness gauge: a pinch-ready
        pose has TCP radial inside the wall and TCP height near the rim."""
        lines = []
        mug = mug_pose0[0, 0:3]
        pads = _t(robot.data.body_pos_w)[0, _pad_ids]
        for name, p in zip(_pad_names, pads):
            rel = p - mug
            radial = float(torch.linalg.vector_norm(rel[:2])) * 1000
            height = float(rel[2]) * 1000
            lines.append(
                f"  {name.replace('follower_left_', '') + ' (mount origin)':38s}"
                f" radial {radial:6.1f} mm  height {height:6.1f} mm\r\n"
            )
        tcp = _t(env.scene["ee_frame"].data.target_pos_w)[0, 0]
        rel = tcp - mug
        radial = float(torch.linalg.vector_norm(rel[:2])) * 1000
        height = float(rel[2]) * 1000
        lines.append(
            f"  {'TCP (finger midpoint)':38s} radial {radial:6.1f} mm  height {height:6.1f} mm"
            f"   [mug: wall R 35-39, rim height 97]\r\n"
        )
        return "".join(lines)

    def report(tag: str = "pose") -> dict[str, float]:
        jd = joint_dict()
        g = vals[GRIPPER_JOINT]
        sys.stdout.write(
            f"\r\n[{tag}]\r\n"
            + "".join(f"  {n:28s} {v:+.6f}\r\n" for n, v in jd.items())
            + f"  gripper {g * 1000:.1f} mm/side ({100.0 * g / grip_open:.0f}% open,"
            f" gap ~{(CLOSED_GAP_M + 2 * g) * 1000:.1f} mm)\r\n"
            + _pad_gauge()
            + "  vector ["
            + ", ".join(f"{v:+.6f}" for v in jd.values())
            + "]\r\n"
        )
        sys.stdout.flush()
        return jd

    def save() -> str:
        jd = joint_dict()
        g = vals[GRIPPER_JOINT]
        body = "\n".join(f'    "{k}": {v:.6f},' for k, v in jd.items())
        text = (
            '"""Grasp pose authored with pose_grasp.py.\n\n'
            "Shaped as the ``pose`` argument of ``mdp.reset_arm_reverse_curriculum``.\n"
            '"""\n\n'
            f"# Gripper when authored: {g:.4f} m per carriage ({100.0 * g / grip_open:.0f}% open,\n"
            f"# finger gap ~{(CLOSED_GAP_M + 2 * g) * 1000:.1f} mm). NOT part of the dict:\n"
            "# reset_arm_reverse_curriculum leaves the carriages at their default open state\n"
            "# so closing the grasp stays the policy's own first action.\n\n"
            f"GRASP_STRADDLE_POSE = {{\n{body}\n}}\n"
        )
        with open(OUT_PATH, "w") as fh:
            fh.write(text)
        sys.stdout.write(f"\r\n  saved -> {OUT_PATH}\r\n")
        sys.stdout.flush()
        return OUT_PATH

    pending = {"status": "ready"}

    _home_vals = dict(vals)  # the pose this session started from

    def do_reset_arm() -> None:
        for n in arm_names:
            vals[n] = cmd[n] = _home_vals[n]
        vals[GRIPPER_JOINT] = cmd[GRIPPER_JOINT] = _home_vals[GRIPPER_JOINT]
        pending["status"] = "arm reset to start pose"

    # ---- the Newton side panel ----------------------------------------------
    panel = {"registered": False, "reason": "no Newton visualizer in this session"}

    def draw_panel(imgui) -> None:
        """Draw the pose controls in the Newton viewer's side panel (immediate mode)."""
        imgui.separator()
        imgui.text("Trossen Mug Grasp Pose")
        if imgui.button("RESET ARM##mugpose"):
            do_reset_arm()
        if imgui.button("PRINT##mugpose"):
            report("print")
        if imgui.button("SAVE##mugpose"):
            save()
            pending["status"] = "saved grasp_pose_authored.py"
        imgui.text(pending["status"])
        # Re-reading vals every frame IS the slider resync.
        for n in editable:
            lo, hi = lim[n]
            changed, v = imgui.slider_float(f"{_short(n)}##mugpose", vals[n], lo, hi)
            if changed:
                vals[n] = v
        changed, v = imgui.slider_float("gripper##mugpose", vals[GRIPPER_JOINT], 0.0, grip_open)
        if changed:
            vals[GRIPPER_JOINT] = v
        imgui.separator()
        imgui.text(f"Object pose (env frame)  --  HOLD {'ON: frozen at sliders' if hold['on'] else 'OFF: physics live'}")
        if imgui.button(f"HOLD OBJECT: {'ON -> release' if hold['on'] else 'OFF -> hold'}##mugpose"):
            if not hold["on"]:
                snap_sliders_to_object()  # hold it where it is, not where the sliders were
            hold["on"] = not hold["on"]
            if hold["on"]:
                place_plate()
            else:
                hold["released"] = True
        if imgui.button("PRINT OBJECT POSE##mugpose"):
            print_object_pose()
        if imgui.button("SNAP SLIDERS TO OBJECT##mugpose"):
            snap_sliders_to_object()
        _moved = False
        for k, (axn, lo2, hi2) in enumerate(
            (("plate_x", -0.35, 0.35), ("plate_y", -0.35, 0.35), ("plate_z", 0.0, 0.45))
        ):
            changed, v = imgui.slider_float(f"{axn}##mugpose", obj_pose["pos"][k], lo2, hi2)
            if changed:
                obj_pose["pos"][k] = v
                _moved = True
        for k, axn in enumerate(("plate_roll", "plate_pitch", "plate_yaw")):
            changed, v = imgui.slider_float(f"{axn}##mugpose", obj_pose["rpy"][k], -3.1416, 3.1416)
            if changed:
                obj_pose["rpy"][k] = v
                _moved = True
        if _moved:
            place_plate()  # HOLD ON: the pin follows; HOLD OFF: one teleport, then physics

    def try_register_panel() -> bool:
        """Attach draw_panel to the Newton viewer; no-op once attached.

        Retried each iteration so it cannot lose a race with viewer creation.
        """
        if panel["registered"]:
            return True
        for viz in getattr(env.sim, "visualizers", None) or getattr(env.sim, "_visualizers", []):
            if "newton" not in type(viz).__name__.lower():
                continue
            viewer = getattr(viz, "_viewer", None)  # no public accessor exists
            if viewer is None:
                continue  # not built yet; retry
            if not hasattr(viewer, "register_ui_callback"):
                panel["reason"] = f"{type(viewer).__name__} has no register_ui_callback(); the Newton API moved"
                continue
            if getattr(viewer, "gui", None) is None:
                # ViewerGL builds its ImGui layer only outside headless mode, and
                # register_ui_callback() is a SILENT no-op without it.
                panel["reason"] = "viewer has no ImGui layer (headless: no DISPLAY, or cfg.headless)"
                continue
            if not getattr(viewer.gui, "is_available", True):
                panel["reason"] = "viewer's ImGui layer failed to init (see 'imgui_bundle UI unavailable' above)"
                continue
            viewer.register_ui_callback(draw_panel, position="side")
            panel["registered"] = True
            panel["reason"] = 'left panel > "Example Options" > "Trossen Mug Grasp Pose"'
            return True
        return False

    try_register_panel()

    # ---- one lab iteration (interactive AND selftest go through this) --------
    def step_once() -> None:
        if not panel["registered"]:
            try_register_panel()
        limit = float("inf") if args_cli.max_target_speed <= 0 else args_cli.max_target_speed * env.physics_dt
        for n in list(cmd):
            cmd[n] += max(-limit, min(limit, vals[n] - cmd[n]))
        arm_t = torch.tensor([[cmd[n] for n in arm_names]], device=device, dtype=torch.float32)
        grip_t = torch.full((1, len(grip_ids)), cmd[GRIPPER_JOINT], device=device, dtype=torch.float32)
        robot.set_joint_position_target_index(target=arm_t, joint_ids=arm_ids)
        robot.set_joint_position_target_index(target=grip_t, joint_ids=grip_ids)
        if _ghost:
            # A ghost has no collisions and would free-fall: pin it every step.
            place_plate()
        if hold["on"] and not _ghost:
            # HOLD = the sim is PAUSED for everything: the object sits at the slider
            # pose with no physics step, so no contact state can build up under a
            # pin and unload as a kick on release (measured: a mug pinned in contact
            # for 60 steps and then freed falls off the branch; the same pose
            # written once and stepped hangs). Arm sliders take effect on release.
            place_plate()
            env.scene.write_data_to_sim()
            if _RENDER:
                env.sim.render()
        else:
            if hold["released"]:
                # First live step after HOLD: one clean pose + zero-velocity write.
                place_plate()
                hold["released"] = False
            else:
                # Physics live: print the object's pose at most once a second, and
                # only when it has MOVED since the last print (> 1 mm or > 1 deg),
                # so a resting object goes quiet instead of spamming the console.
                _watch["n"] += 1
                if _watch["n"] % 90 == 0:
                    _pos, _rpy, _ = live_object_pose()
                    _last = _watch.get("last")
                    _moved = _last is None or max(abs(a - b) for a, b in zip(_pos, _last[0])) > 0.001 or max(
                        abs(a - b) for a, b in zip(_rpy, _last[1])
                    ) > 0.0175
                    if _moved:
                        print_object_pose()
                        _watch["last"] = (_pos, _rpy)
            env.scene.write_data_to_sim()
            # NOT `not args_cli.headless`: AppLauncher OVERWRITES args_cli.headless to True
            # for '--viz newton' (only the Kit visualizer implies non-headless), so that
            # spelling renders nothing and the Newton window stays blank. Render whenever
            # this is not the selftest.
            env.sim.step(render=_RENDER)
            env.scene.update(env.physics_dt)

    # ---- selftest ------------------------------------------------------------
    if args_cli.selftest:
        fails: list[str] = []

        def check(name: str, ok: bool, detail: str = "") -> None:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}")
            if not ok:
                fails.append(name)

        print("[selftest] driving the panel through a stub (no GUI)")
        mug_env = (_t(obj.data.root_pos_w)[0] - _t(env.scene.env_origins)[0]).tolist()
        # The plate's authored spawn, read from the TASK CFG so this check
        # follows every scene revision. Contacts are ON in this lab, so
        # allow the few mm the free plate settles by before this read.
        _sp = list(env.cfg.scene.object.init_state.pos)
        check(
            "plate at its authored spawn",
            all(abs(mug_env[i] - _sp[i]) < 0.02 for i in range(3)),
            f"plate env = ({mug_env[0]:+.4f}, {mug_env[1]:+.4f}, {mug_env[2]:+.4f}) vs cfg {tuple(round(v, 4) for v in _sp)}",
        )
        q_now = _t(robot.data.joint_pos)[0]
        worst = max(abs(float(q_now[i]) - vals[n]) for n, i in zip(arm_names, arm_ids))
        check("session starts AT the configured start pose", worst < 0.02, f"worst joint error {worst:.4f} rad")
        if not _use_bank:
            # The READY pose must equal the rig's own init_state, not a restated copy.
            dflt = _t(robot.data.default_joint_pos)[0]
            worst_d = max(abs(vals[n] - float(dflt[i])) for n, i in zip(arm_names, arm_ids))
            check(
                "start pose IS the Trossen ready pose (articulation default)",
                worst_d < 1e-6,
                f"worst delta {worst_d:.2e} rad; j1={vals[arm_names[1]]:.4f} (pi/2={3.14159265 / 2:.4f})",
            )
            check(
                "ready pose opens the carriages",
                abs(vals[GRIPPER_JOINT] - grip_open) < 1e-6,
                f"carriage {vals[GRIPPER_JOINT]:.4f}",
            )
        stub = _StubImgui()
        draw_panel(stub)
        check(
            "panel draws a slider per arm joint + gripper + object xyz + rpy",
            len(stub.sliders) == len(editable) + 7,
            f"{len(stub.sliders)} sliders: {stub.sliders}",
        )
        check("panel draws RESET/PRINT/SAVE + object HOLD/PRINT/SNAP buttons", len(stub.buttons) == 6, f"{stub.buttons}")

        probe = editable[3]
        lo, hi = lim[probe]
        want = min(max(vals[probe] + 0.35, lo), hi)
        draw_panel(_StubImgui(drag=(f"{_short(probe)}##mugpose", want)))
        check(
            "dragging a slider updates the value dict",
            abs(vals[probe] - want) < 1e-6,
            f"{probe} = {vals[probe]:+.4f}, wanted {want:+.4f}",
        )

        before = float(_t(robot.data.joint_pos)[0, arm_ids[3]])
        for _ in range(240):
            step_once()
        after = float(_t(robot.data.joint_pos)[0, arm_ids[3]])
        check(
            "the slider actually MOVES the joint",
            abs(after - want) < 0.05,
            f"{probe}: {before:+.4f} -> {after:+.4f}, target {want:+.4f}",
        )

        gwant = grip_open * 0.5
        draw_panel(_StubImgui(drag=("gripper##mugpose", gwant)))
        for _ in range(240):
            step_once()
        gafter = float(_t(robot.data.joint_pos)[0, grip_ids[0]])
        check(
            "the gripper slider reaches a PARTIAL opening",
            abs(gafter - gwant) < 0.006,
            f"carriage {gafter:.4f}, wanted {gwant:.4f}",
        )
        check("both carriages track together", abs(float(_t(robot.data.joint_pos)[0, grip_ids[1]]) - gafter) < 0.006)

        if _ghost:
            moved = float(torch.linalg.vector_norm(_t(obj.data.root_pos_w)[0] - mug_pose0[0, 0:3]))
            check("ghost mug stays pinned", moved < 0.02, f"moved {moved:.4f} m")

        draw_panel(_StubImgui(click="RESET ARM##mugpose"))
        check(
            "RESET ARM restores the session start pose",
            abs(vals[probe] - _home_vals[probe]) < 1e-6,
            f"{probe} = {vals[probe]:+.4f}, start {_home_vals[probe]:+.4f}",
        )

        path = save()
        with open(path) as f:
            ok = os.path.exists(path) and "GRASP_STRADDLE_POSE" in f.read()
        check("SAVE writes a loadable pose file", ok, path)

        print(f"[selftest] {'ALL PASSED' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
        env.close()
        simulation_app.close()
        return 1 if fails else 0

    # ---- interactive ---------------------------------------------------------
    print(f"\n[pose_grasp] Newton side panel: {panel['reason']}")
    print("  Drag the joint sliders in the viewer's LEFT panel. The pose is held.")
    print("  Terminal keys (optional):  P print   S save   x quit\n")
    sys.stdout.flush()

    keys = RawKeys()
    with keys:
        while simulation_app.is_running():
            for ch in keys.poll():
                if ch == "P":
                    report("print")
                elif ch == "S":
                    save()
                elif ch == "x":
                    env.close()
                    simulation_app.close()
                    return 0
            step_once()

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
