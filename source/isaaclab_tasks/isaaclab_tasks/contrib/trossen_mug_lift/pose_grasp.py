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
      source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_mug_lift/pose_grasp.py --viz newton

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

parser = argparse.ArgumentParser(description="Author a grasp pose for the Trossen mug-lift task.")
parser.add_argument(
    "--selftest", action="store_true", help="Drive every control surface headlessly; exit non-zero on failure."
)
parser.add_argument(
    "--max_target_speed", type=float, default=1.5, help="rad/s cap between slider value and PD target (0 = snap)."
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
    GRASP_BANK_POSE,
    GRIPPER_JOINT,
    GRIPPER_JOINT_R,
    OBJECT_REST_Z,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# 1:1 WITH TRAINING: the training launches export these; a bare lab launch
# would otherwise build a DIFFERENT scene (mesh mug, no rails) and any pose
# authored in it would be authored against geometry the task never sees.
# setdefault, so an explicit override still wins.
os.environ.setdefault("MUG_COLLISION", "hull")
os.environ.setdefault("TROSSEN_RAILS", "1")

TASK = "IsaacContrib-Lift-Mug-Trossen-v0"
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
    apply_solver_choice(cfg, "icf")
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

    # GHOST MUG (POSE_MUG_GHOST=0 to disable): you pose the gripper straight through
    # where the mug is, and a physical mug just gets batted across the table.
    if os.environ.get("POSE_MUG_GHOST", "1") != "0":
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

    _ghost = os.environ.get("POSE_MUG_GHOST", "1") != "0"
    obj = env.scene["object"]
    # Pin to the SPAWN pose, not wherever it is now: with collisions off the ghost sinks
    # a little on every step before the first pin, and freezing that leaves the mug
    # buried in the slab (measured z = 0.0089 against a 0.021 spawn).
    mug_pose0 = _t(obj.data.root_pose_w).clone()
    mug_pose0[0, 2] = float(_t(env.scene.env_origins)[0][2]) + OBJECT_REST_Z

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
            # A ghost mug has no collisions, so nothing holds it up: without this pin
            # it free-falls through the tabletop. RigidObject's *_index setters are
            # keyword-only and take root_pose/root_velocity (NOT position=/velocity=).
            obj.write_root_pose_to_sim_index(root_pose=mug_pose0)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(_t(obj.data.root_vel_w)))
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
        check(
            "mug spawns 45.75 cm forward: env y = 0.000",
            abs(mug_env[1]) < 1e-3,
            f"mug env = ({mug_env[0]:+.4f}, {mug_env[1]:+.4f}, {mug_env[2]:+.4f})",
        )
        check("mug on the centerline: env x = -0.020", abs(mug_env[0] - (-0.020)) < 1e-3)
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
            "panel draws a slider per arm joint + gripper",
            len(stub.sliders) == len(editable) + 1,
            f"{len(stub.sliders)} sliders: {stub.sliders}",
        )
        check("panel draws RESET/PRINT/SAVE buttons", len(stub.buttons) == 3, f"{stub.buttons}")

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
