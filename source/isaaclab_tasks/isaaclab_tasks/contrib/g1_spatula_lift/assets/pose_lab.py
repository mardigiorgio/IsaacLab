# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive pose lab: pose the hand in the GUI, read the joint values live.

Opens the spatula scene with a viewer, teleports the robot to a starting pose,
then holds that pose: every step the right arm/hand PD targets are rewritten
from an owned value dictionary, so the arm and fingers stay wherever you put
them instead of springing back. Right-hand joint angles and fingertip heights
print on a timer so you can read off the pose you want.

There are three interchangeable control surfaces, all writing the same values:

* the **Newton side panel** (``--visualizer newton``), inside the viewer's left
  panel under "Example Options";
* the **omni.ui window** "Spatula Pose Lab" (``--visualizer kit``, the default) --
  ``omni.ui`` ships only as a Kit extension, so it cannot exist under Newton;
* the **JSON pose file** (both backends): edit it, save, and the pose follows.

Run::

    ./isaaclab.sh -p .../assets/pose_lab.py                     # neutral hover, Kit window
    ./isaaclab.sh -p .../assets/pose_lab.py --visualizer newton # neutral hover, Newton side panel
    ./isaaclab.sh -p .../assets/pose_lab.py --pose dict         # PREGRASP_JOINT_POS claw
    ./isaaclab.sh -p .../assets/pose_lab.py --pose map          # grasp_map.pt claw (authored on the OLD table)
    ./isaaclab.sh -p .../assets/pose_lab.py --selftest          # drive every control surface, non-zero on failure
    ./isaaclab.sh -p .../assets/pose_lab.py --max_target_speed 0  # snap targets (pre-limiter behaviour)

``--selftest`` runs headless on its own and opens (and keeps pumping) a live
window when combined with ``--visualizer``, which is how the Kit-only surfaces
get exercised. Press Ctrl+C (or close the viewer) to exit the interactive run;
with ``--save`` the last pose is written as a one-stage grasp map, ready to drop
in as ``grasp_map.pt`` -- unless the run diverged past recovery or the pose is
non-finite, in which case :func:`save_pose` refuses, writes nothing, and the
process exits 1 so a ``pose_lab.py --save p.pt && train.py p.pt`` chain stops
instead of reading a stale file.
"""

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable

from isaaclab.app import add_launcher_args

# The constants sit ABOVE the parser on purpose: several of them are CLI
# defaults, and a default that duplicates a literal is a default that drifts.
TASK = "IsaacContrib-Lift-Spatula-G1-v0"
RIGHT_HAND = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_RESET_TOL = 3e-2
"""Post-reset right arm/hand tolerance [rad], sized from measurement: the
residual one frame after the write is ~1.5e-3 from the free-space hover and
~6.5e-3 from the claw poses, whose thumb is pressed into the tabletop. A reset
that never reached the solver reads the full ~3e-1 displacement, so this gate
sits ~5x above the worst true residual and ~10x below a dead reset.

Deliberately NOT measured over all 43 joints: the robot is fixed-base, its legs
dangle, and the right knee falls ~4e-1 rad within a fifth of a second -- folding
that in made a working reset report FAIL."""
ARM_DISPLACED_TOL = 1.5e-1
"""Selftest gate [rad]: how far the arm must visibly move before a reset is
asked to undo it. Measured displacement is ~3e-1."""
ARM_DISPLACE_BY = 3e-1
"""Selftest: how far off the start pose to drive each right-arm joint [rad]."""
MAX_TARGET_SPEED_DEFAULT = 2.0
"""Default cap on how fast a commanded PD target is allowed to move [rad/s].

Every control surface writes an authored pose into ``vals``; the loop walks the
*commanded* target toward it at this rate rather than stepping straight to it.

A step change in a position target is a force impulse. Measured on this scene: a
0.3 rad step on the seven right-arm joints drives the arm at ~12 rad/s -- and the
peak scales with the step, reaching ~16 rad/s at 0.5 rad and 62 rad/s on a full
limit-to-limit jump (see :data:`RUNAWAY_JOINT_SPEED`) -- while
the light spatula leaves the resulting contact at 50-190 m/s -- far more energy
than the strike carried, i.e. an under-converged contact solve injecting energy
rather than dissipating it. Driven through 25 perturb/reset cycles of that step
input the scene reached a non-finite state within 2 cycles (and 6 cycles with
the solver's iteration budget doubled, so convergence alone does not fix it).
Rate limited, 25/25 cycles stayed finite and the first cycle's count of frames
above 10 m/s fell from 175 to 6.

2 rad/s is about the speed this arm moves in manipulation. It is only ever felt
on a large jump -- a slider drag moves far less than this per frame."""
CONTROL_RATE_HZ = 240.0
"""Physics rate of this scene [Hz]. Used only to derive :data:`MIN_TARGET_SPEED`.

Hardcoded because ``--max_target_speed`` is validated at parse time, before a
simulation exists to ask. :func:`run_selftest` re-derives the bound from the
LIVE ``env.cfg.sim.dt`` and the LIVE joint limits and FAILs if this drifts."""
TARGET_MAGNITUDE_BOUND = 4.0
"""Bound on |commanded PD target| [rad] over the editable joints.

Every authored value is clamped into its joint limits before it reaches the
limiter, so this bounds the joint limits themselves. Widest editable stop
measured on this robot: 3.089 rad (``right_shoulder_pitch_joint``). Rounded UP
to the next binade, which makes :data:`MIN_TARGET_SPEED` conservative rather
than marginal, because ULP is monotone in magnitude."""
MIN_TARGET_SPEED = math.ulp(TARGET_MAGNITUDE_BOUND) * CONTROL_RATE_HZ
"""Smallest non-zero rate the parser accepts [rad/s]; 2.132e-13 at 240 Hz.

The limiter walks the commanded target by ``rate * sim_dt`` per frame, so a rate
is usable only if that step is big enough to CHANGE a double. ``x + ulp(x) != x``
holds for every finite ``x`` and ULP is monotone, so a per-frame allowance of one
``ulp(TARGET_MAGNITUDE_BOUND)`` is guaranteed to move every target this tool can
author, and anything smaller rounds straight back on the widest joints.

Rejecting only SUBNORMAL rates -- the previous bound, ``sys.float_info.min`` --
did not implement that rationale at all: it left ~295 orders of magnitude of
ordinary normal doubles accepted, ``--max_target_speed 1e-16`` among them, whose
per-frame allowance of 4.2e-19 rad added to an O(1e-1) rad target rounds back to
the target. Every control surface then goes dead and the interactive loop renders
a frozen arm forever with no diagnostic. The subnormal bound is subsumed, and so
is the slew-window overflow of :data:`SELFTEST_MAX_SLEW_FRAMES`, since
2.1e-13 is far above both."""
SPATULA_RESET_TOL = 1e-2
"""Post-reset spatula tolerance [m]: it settles ~2 mm below spawn under gravity."""
RESET_SETTLE_FRAMES = 1
"""Steps between a reset and its PASS/FAIL verdict.

Not zero: the position readback is a zero-copy view of the write buffer, so a
same-frame check reports success even when the solver discards the write -- the
very bug this tool shipped with. One full step is enough for a discarded write
to show up, because the solver's writeback lands on the next step.

Not more than one either: a claw start pose is not a force equilibrium, so its
thumb is levered out of contact at a steady ~1.7e-2 rad per frame and never
settles. Waiting longer measures that divergence rather than the reset."""
SELFTEST_MAX_SLEW_FRAMES = 2400
"""Cap on the selftest's derived slew window [frames], 10 s at 240 Hz.

Without it ``--max_target_speed 0.001`` would silently spin for 72000 frames and
read as a hang. Past the cap the arm simply has not arrived and
"slider path moved the arm off start" FAILs, which is the truthful verdict.

Applied BEFORE the ``math.ceil`` that turns the derived window into a frame
count. That ordering is the only one that is correct for every rate: below about
4e-307 rad/s the raw window ``ARM_DISPLACE_BY / (rate * dt)`` overflows to
``inf`` (measured at 240 Hz: 1e-305 still divides to 7.2e306, 4e-307 gives
``inf``), and ``math.ceil(inf)`` raises OverflowError -- so a cap applied to the
ceil's RESULT could never run, because the ceil is what raises.

Nothing on the CLI reaches that overflow any more: :data:`MIN_TARGET_SPEED`
(2.1e-13 rad/s) is ~294 decades above 4e-307, and the slowest rate the parser now
accepts derives a finite 3.4e14-frame window that this cap turns into 2400. The
ordering stays because it costs nothing and the two constants are independent --
a future floor is not required to keep the division finite for the cap to work."""
RUNAWAY_JOINT_SPEED = 200.0
"""Runaway gate on |joint_vel| [rad/s] over the watched (non-finger) joints.

3.2x the worst LEGITIMATE transient measured on this scene. "Legitimate" is
decided WITHOUT consulting this gate, or the sizing would be circular: a sample
counts only if its whole 150-frame hold window stayed finite AND kept the spatula
inside :data:`RUNAWAY_OBJECT_SPEED` and :data:`RUNAWAY_OBJECT_RANGE`, i.e. nothing
else said the scene had blown up. Every jump below is a limit-to-limit snap under
``--max_target_speed 0`` -- the snap behaviour a user can select, which makes each
authored jump a step input -- from the hover start, restored between samples.
Peak watched joint speed [rad/s]:

* all seven right-arm sliders to their far stops at once, 3.27 rad on the widest
  joint and the largest pose any control surface can author: **62.3**;
* one arm slider limit-to-limit: 36.0 wrist_yaw, 28.4 elbow, 25.3 shoulder_roll,
  23.1 wrist_pitch, 15.7 shoulder_yaw, 1.6 shoulder_pitch;
* any finger slider alone, and all seven fingers together: under 0.3;
* the selftest's own 0.3 rad arm step, which it prints every run: 11.2.

62.3 rad/s is the number this gate is sized from, and the previous 40.0 sat below
it. In that clean full-amplitude sample the watched speed crossed 40 on frame 13
and spent 37 of its 150 frames above it, while the spatula never exceeded 11.4 m/s
and no value ever went non-finite -- so 40.0 fired SIMULATION RUNAWAY 37 times on
motion produced by an advertised feature, on a scene that was fine. That misfire
is not harmless: a recovery overwrites ``vals`` and ``cmd`` with the start pose
and teleports the spatula, destroying the pose the user was authoring.

The detection cost of 40 -> 200, stated plainly. Four samples DID diverge, and on
three of them the watched joints never reached even 40 (peaks 47.5, 36.5, 11.8
while finite): :data:`RUNAWAY_OBJECT_SPEED` caught all three, on frames 37, 80 and
22, with the joints reading 28.2, 7.6 and 6.2 rad/s. The failure mode of this
scene is an ejected spatula, not a spinning arm, and the object gate is what sees
it. The fourth sample is the same all-seven-arm snap as above, re-run: it crossed
40 on frame 13 again, but this time went on to trip the object gate on frame 49,
200 rad/s on frame 93 and non-finite on frame 125, peaking at 579.8 rad/s. So 40
would have flagged that one 36 frames before the object gate did -- on a
trajectory whose first 13 frames are indistinguishable from the clean run's. A
gate at 40 cannot buy that 0.15 s without also destroying the pose in the clean
case; 200 still fires on the same divergence, and this gate is not dead.

200 is also 5x below the 1000 rad/s state :func:`run_selftest` asserts must still
classify as a runaway, which pins the constant from both sides: 62.3 must stay
healthy, 1000 must not. Under the default 2 rad/s limit the arm tracks the
commanded target and the selftest's own peak is 2.6-2.8 rad/s across its four
modes -- ~70x under this gate, so nothing routine comes near it.

Fingers are excluded for the reason the task's own ``TerminationsCfg.robot_exploded``
excludes them -- their tiny links spike legitimately on contact. Nothing is lost:
a finger runaway shows up on :data:`RUNAWAY_OBJECT_SPEED`, because it is the
66 g spatula that gets ejected, and it reaches the arm within a few frames."""
RUNAWAY_JOINT_NAMES = ["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
"""Joints the speed gate watches, copied from ``TerminationsCfg.robot_exploded``.

That termination never fires here: this loop drives ``env.sim.step`` directly and
runs no manager, so the pose lab needs its own copy of the guard."""
RUNAWAY_OBJECT_SPEED = 50.0
"""Runaway gate on the spatula's linear speed [m/s].

The bottom of the measured ejection band: a step target change ejects the spatula
at 50-190 m/s (see :data:`MAX_TARGET_SPEED_DEFAULT`), and from there the state
reached non-finite within two perturb/reset cycles. The trace this gate exists
for read 5.9e7 m/s WHILE STILL FINITE three frames before overflow -- six decades
above this bound -- so the trip lands with a large margin rather than three frames.

The healthy side is bounded from below, not pinned: rate limited, the first cycle
still spends 6 frames above 10 m/s and that run's PEAK was never recorded. 50 is
the lowest value the evidence rules out as healthy, not a tight gate. The selftest
now prints the measured peak so the next run can tighten this to ~3x the real
healthy maximum."""
RUNAWAY_OBJECT_RANGE = 10.0
"""Runaway gate on the spatula's distance from spawn [m].

Catches a body that has already left the world on a frame whose sampled speed
happens to be small, and catches the overflow case in the frame it happens: at
5.9e7 m/s the spatula covers 2.5e5 m in one 1/240 s step.

~14x the task's own out-of-bounds box (``TerminationsCfg.spatula_dropped``:
x in [-0.7, 0.7], y in [-0.43, 0.43]) so it cannot fire on a spatula merely
knocked off the 1.22 x 0.762 m table -- legitimate here, since the pose lab runs
no termination manager and no episode reset."""
RECOVERY_MUTE_FRAMES = 12
"""Frames the health gate is muted after a recovery, ~0.05 s at 240 Hz.

Long enough for the teleport's own transient to pass (a reset verdict is taken
after :data:`RESET_SETTLE_FRAMES`) and short enough that a recovery which did not
take is caught within a twentieth of a second."""
MAX_RECOVERIES = 5
"""Runaway recoveries before the lab stops instead of recovering again.

A recovery that keeps re-tripping is not a recovery. Five rides out a user
re-authoring the same bad jump twice while still terminating."""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pose",
    default="hover",
    choices=["hover", "default", "dict", "map"],
    help=(
        "starting pose: 'hover' (env default, fingers open; alias 'default'), "
        "'dict' (PREGRASP_JOINT_POS claw), or 'map' (grasp_map.pt stage 'pre_grasp_caged')"
    ),
)
parser.add_argument("--save", default=None, help="write the final pose here as a {'stages': [...]} file")
parser.add_argument("--stage_name", default="pre_grasp_caged", help="stage name written by --save")
parser.add_argument("--print_every", type=float, default=2.0, help="seconds between joint printouts")
parser.add_argument("--pose_file", default="/tmp/spatula_pose.json", help="edit this file to move joints")
parser.add_argument(
    "--selftest", action="store_true", help="drive every control surface non-interactively and exit non-zero on failure"
)
parser.add_argument(
    "--max_target_speed",
    type=float,
    default=MAX_TARGET_SPEED_DEFAULT,
    help=(
        f"cap on how fast a commanded PD target moves [rad/s] (default {MAX_TARGET_SPEED_DEFAULT});"
        " 0 disables the limit so a large authored jump snaps -- the pre-limiter behaviour, and the"
        " one measured to diverge this scene"
    ),
)
add_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Validated HERE, at parse time, so a bad value costs a one-line error instead of
# a whole simulation launch. ``< 0`` alone is not a validation: every comparison
# against NaN is False, so ``nan`` passed straight through -- and NaN is truthy,
# so the ``or math.inf`` fallback below never fired either. That left
# MAX_TARGET_SPEED = nan, which makes the per-frame allowance nan and turns
# EVERY editable PD target non-finite on the first frame: the rate limiter would
# poison the very simulation it exists to protect.
if not math.isfinite(args_cli.max_target_speed) or args_cli.max_target_speed < 0:
    parser.error(
        f"--max_target_speed must be a finite value >= 0 (0 disables the limit), not {args_cli.max_target_speed}"
    )
# Too-slow rates are rejected for a behavioural reason, not a stylistic one: the
# limiter's per-frame step is `rate * sim_dt`, and below MIN_TARGET_SPEED that
# step is smaller than one ULP of the widest authorable target, so `cmd += step`
# returns cmd unchanged. Every control surface goes dead while the loop keeps
# rendering, i.e. the tool reads as a hang with no diagnostic. Bounding this at
# `sys.float_info.min` did NOT cover that: subnormals are ~295 orders of
# magnitude below where the arithmetic actually stalls.
if 0.0 < args_cli.max_target_speed < MIN_TARGET_SPEED:
    parser.error(
        f"--max_target_speed {args_cli.max_target_speed} is below {MIN_TARGET_SPEED:.4g} rad/s, the rate whose"
        f" per-frame step at {CONTROL_RATE_HZ:.0f} Hz is one ULP of the widest target this tool can author"
        f" ({TARGET_MAGNITUDE_BOUND} rad); a smaller step rounds back to the value it started from, so the arm"
        " would never arrive. Pass 0 to disable the limit instead."
    )
MAX_TARGET_SPEED = args_cli.max_target_speed or math.inf
"""The rate limit in force this run [rad/s], or ``math.inf`` when disabled.

``math.inf`` rather than ``None`` on purpose: it makes both consumers correct
with no branch. In :func:`main` the per-frame allowance ``MAX_TARGET_SPEED *
sim_dt`` becomes ``inf`` and the clamp is the identity; in :func:`run_selftest`
the derived slew window ``ARM_DISPLACE_BY / (MAX_TARGET_SPEED * dt)`` becomes
``0.0`` -- the target snaps, so zero slew frames are needed -- instead of a
ZeroDivisionError."""
# ``--visualizer`` parses to a LIST (or None); always hand the launcher a list --
# the kitless path joins the value with " ", which would shred a bare string.
_headless = False
if getattr(args_cli, "visualizer", None) is None:
    if args_cli.selftest:
        # A bare selftest needs no window. This is the framework's own "--viz none"
        # spelling: an EMPTY selection marked explicit. Handing it the literal
        # ["none"] instead trips SimulationContext's "explicitly requested
        # visualizer(s) ['none'] could not be configured" on the kitless path,
        # which never normalises "none" away the way AppLauncher does.
        args_cli.visualizer_explicit = True
        _headless = True
    else:
        # the interactive tool is useless without a window
        args_cli.visualizer = ["kit"]
args_cli.headless = _headless
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _short(joint_name: str) -> str:
    """Return the compact slider label for a joint name."""
    return joint_name.replace("right_", "").replace("_joint", "")


@dataclasses.dataclass
class LabHooks:
    """Live entry points into the running lab, handed to :func:`run_selftest`.

    The selftest drives these rather than re-implementing the loop, so every
    check goes through the exact code a button click goes through. Without this
    the selftest could pass while the buttons were dead -- the failure mode this
    tool actually shipped with.

    Attributes:
        tick: Advance the lab one iteration: drain queued resets, drive the arm
            from ``vals``, step, then report any verdict that has come due.
        click: Reset name to the zero-argument callable the buttons are bound to.
        vals: The single source of truth for the editable joint values [rad].
        pending: Shared command/verdict state, including the ``"log"`` of every
            status line the lab has emitted, ``"diverged"`` (the frame of the
            first health-gate trip of any kind, ``None`` if never, never cleared
            by a recovery), ``"recoveries"``, and ``"peaks"``, the running max of
            (joint speed [rad/s], spatula speed [m/s], spatula range [m]).
        draw_panel: The Newton side-panel callback; takes an ImGui-like object.
        buttons: Reset name to its ``omni.ui`` button widget. Empty under Newton.
        editable: The editable joint names, in PD-target column order.
    """

    tick: Callable[[], None]
    click: dict[str, Callable[[], None]]
    vals: dict[str, float]
    pending: dict[str, object]
    draw_panel: Callable[[object], None]
    buttons: dict[str, object]
    editable: list[str]


class _StubImgui:
    """Minimal ImGui stand-in so the selftest can drive the Newton side panel.

    ImGui is immediate mode: a widget reports interaction through its *return
    value*, so "click" means :meth:`button` returns ``True`` and "drag" means
    :meth:`slider_float` reports a change. Feeding this to the real panel
    callback exercises it on both backends without needing a GUI.

    Args:
        click: Label whose button reports a click, or ``None`` for no click.
        drag: ``(label, value)`` for the one slider that reports a change.
    """

    def __init__(self, click: str | None = None, drag: tuple[str, float] | None = None):
        self._click = click
        self._drag = drag
        self.buttons: list[str] = []
        self.sliders: list[str] = []
        self.texts: list[str] = []

    def separator(self) -> None:
        """Ignore the separator; the stub draws nothing."""

    def text(self, label: str) -> None:
        """Record a text label the panel drew."""
        self.texts.append(label)

    def button(self, label: str) -> bool:
        """Record a button and report whether this is the one being clicked."""
        self.buttons.append(label)
        return label == self._click

    def slider_float(self, label: str, value: float, lo: float, hi: float) -> tuple[bool, float]:
        """Record a slider and report the staged drag, clamped like ImGui's own."""
        self.sliders.append(label)
        if self._drag is not None and self._drag[0] == label:
            return True, min(max(self._drag[1], lo), hi)
        return False, value


def reset_spatula(env, eid, dev) -> list[float]:
    """Teleport the spatula back to its spawn pose, at rest.

    Module-level so a test can drive it directly rather than only through a
    button. Deliberately does not read the pose back: ``root_pos_w`` is a
    zero-copy view of the write buffer, so a pre-step readback reports success
    even when the solver discards the write. The caller verifies after stepping.

    Args:
        env: The running environment.
        eid: Environment indices to reset. Shape is (1,).
        dev: Torch device the simulation runs on.

    Returns:
        The pre-teleport spatula position in the env frame [m].
    """
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
        SPATULA_SPAWN_POS,
        SPATULA_SPAWN_QUAT,
    )

    obj = env.scene["spatula"]
    before = (obj.data.root_pos_w.torch[eid[0]] - env.scene.env_origins[eid[0]]).tolist()
    pose = torch.tensor([*SPATULA_SPAWN_POS, *SPATULA_SPAWN_QUAT], device=dev).unsqueeze(0)
    pose[:, :3] += env.scene.env_origins[eid]
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=eid)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=eid)
    # no write_data_to_sim(): on a rigid object that only composes external
    # wrenches, and the canonical reset events do not call it either
    return before


def reset_arm(env, eid, start_q, joint_ids) -> torch.Tensor:
    """Put the right arm and hand back to the pose the lab started from.

    Writes the joint state AND the PD target together, then flushes, so the arm
    snaps back instead of being dragged there by the controller.

    Only the editable columns are touched. A full-width restore also re-snapped
    the dangling legs of this fixed-base robot to their t=0 angles, which they
    then lost again to gravity within a few frames -- a whole-body twitch on a
    button labelled RESET ARM, and a moving target for the verdict.

    Args:
        env: The running environment.
        eid: Environment indices to reset. Shape is (1,).
        start_q: Joint positions to restore [rad]. Shape is (1, num_joints).
        joint_ids: Columns of ``start_q`` to restore.

    Returns:
        The restored joint positions [rad]. Shape is (1, len(joint_ids)).
    """
    robot = env.scene["robot"]
    q = start_q[:, joint_ids].clone()
    robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), joint_ids=joint_ids, env_ids=eid)
    # set_joint_position_target_index only stages the value; this flush is required
    robot.set_joint_position_target_index(target=q, joint_ids=joint_ids, env_ids=eid)
    robot.write_data_to_sim()
    return q


def health_report(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    watched_ids: list[int],
    object_pos: torch.Tensor,
    object_vel: torch.Tensor,
    object_spawn: torch.Tensor,
) -> tuple[str | None, str, tuple[float, float, float]]:
    """Classify the simulation state as healthy, runaway, or non-finite.

    ``isfinite`` alone is not enough. In a measured trace the spatula reached
    5.9e7 m/s while every value was still finite, three frames before the state
    overflowed; a selftest that ended inside that window printed ALL CHECKS
    PASSED on a scene that was already destroyed.

    The two kinds are reported separately because only one of them is
    recoverable: at ``"runaway"`` the state is numerically clean and a teleport
    can restore it, at ``"nonfinite"`` a NaN has already reached buffers a state
    write does not touch. Non-finite is tested FIRST, so a state that is both
    huge and NaN is reported as the unrecoverable one.

    Everything is reduced on device and synced ONCE, as a single stacked vector,
    so the classification is plain Python arithmetic on seven scalars -- one
    device sync per call, as before, and the same code path for live tensors and
    for the selftest's synthetic ones.

    Args:
        joint_pos: Robot joint positions [rad]. Shape is (num_envs, num_joints).
        joint_vel: Robot joint velocities [rad/s]. Shape is (num_envs, num_joints).
        watched_ids: Columns of ``joint_vel`` the speed bound applies to; the
            finiteness check always covers the full width.
        object_pos: Object position in the ENV frame [m]. Shape is (num_envs, 3).
        object_vel: Object spatial velocity [m/s, rad/s]. Shape is (num_envs, 6).
        object_spawn: Object spawn position in the env frame [m]. Shape is (3,).

    Returns:
        ``(kind, detail, peaks)``. ``kind`` is ``None`` when healthy,
        ``"runaway"`` when finite but out of bounds, ``"nonfinite"`` otherwise.
        ``peaks`` is this frame's ``(joint speed [rad/s], object speed [m/s],
        object range [m])``, for the caller to accumulate; entries may be NaN
        when ``kind`` is ``"nonfinite"``.
    """
    dtype = joint_pos.dtype
    measured = torch.stack(
        [
            (~torch.isfinite(joint_pos).all()).to(dtype),
            (~torch.isfinite(joint_vel).all()).to(dtype),
            (~torch.isfinite(object_pos).all()).to(dtype),
            (~torch.isfinite(object_vel).all()).to(dtype),
            joint_vel[:, watched_ids].abs().max(),
            object_vel[:, :3].norm(dim=-1).max(),
            (object_pos - object_spawn).norm(dim=-1).max(),
        ]
    )
    bad_q, bad_qd, bad_p, bad_v, q_speed, o_speed, o_range = measured.tolist()  # the one sync
    peaks = (q_speed, o_speed, o_range)
    named = (("joint_pos", bad_q), ("joint_vel", bad_qd), ("object_pos", bad_p), ("object_vel", bad_v))
    bad = [name for name, flag in named if flag]
    if bad:
        return "nonfinite", f"non-finite {', '.join(bad)}", peaks
    if q_speed > RUNAWAY_JOINT_SPEED:
        return "runaway", f"watched joint speed {q_speed:.1f} rad/s over the {RUNAWAY_JOINT_SPEED} rad/s bound", peaks
    if o_speed > RUNAWAY_OBJECT_SPEED:
        return "runaway", f"spatula speed {o_speed:.1f} m/s over the {RUNAWAY_OBJECT_SPEED} m/s bound", peaks
    if o_range > RUNAWAY_OBJECT_RANGE:
        return "runaway", f"spatula {o_range:.1f} m from spawn, over the {RUNAWAY_OBJECT_RANGE} m bound", peaks
    return None, "", peaks


def save_pose(
    path: str,
    stage_name: str,
    joint_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    fatal: int | None = None,
) -> bool:
    """Write the final pose to *path* as a one-stage grasp map, unless it is poisoned.

    ``--save`` is the one thing this tool leaves behind, and the module docstring
    offers the result as a drop-in ``grasp_map.pt``. So the run that ends worst --
    a divergence, which breaks the interactive loop and falls straight through to
    here -- is exactly the run whose pose must never reach disk: every downstream
    consumer would read NaN out of a file the tool said was ready to use.

    Two INDEPENDENT refusals, deliberately not folded into one. ``fatal`` is the
    lab's own verdict that the run cannot be recovered; the finiteness check
    re-measures the tensors about to be serialized. Either alone refuses, so a
    NaN pose is still blocked if the flag is somehow clear, and a flagged run is
    still blocked even though a runaway can leave every value finite.

    Args:
        path: Destination file.
        stage_name: Name recorded for the single stage.
        joint_pos: Full-width joint positions [rad]. Shape is (num_joints,).
        object_pos: Object position in the env frame [m]. Shape is (3,).
        object_quat: Object orientation as a wxyz quaternion. Shape is (4,).
        fatal: Frame of the unrecoverable health-gate trip, or ``None`` when the
            run never diverged past recovery.

    Returns:
        Whether the file was written.
    """
    refusal = None
    if fatal is not None:
        refusal = (
            f"the simulation diverged past recovery at frame {fatal}, so the live pose is whatever the"
            " divergence left behind rather than a pose you authored"
        )
    else:
        bad = [
            name
            for name, tensor in (
                ("joint_pos", joint_pos),
                ("object_pos", object_pos),
                ("object_quat", object_quat),
            )
            if not torch.isfinite(tensor).all()
        ]
        if bad:
            refusal = f"non-finite {', '.join(bad)} in the pose about to be serialized"
    if refusal is not None:
        print(
            f"\n[pose_lab] REFUSING TO SAVE: {refusal}. NOTHING was written to {path} and any existing file"
            " there is untouched -- a saved NaN would be read back as a pose by everything downstream.",
            flush=True,
        )
        return False
    torch.save(
        {
            "stages": [
                {
                    "name": stage_name,
                    "joint_pos": joint_pos,
                    "joint_vel": torch.zeros_like(joint_pos),
                    "object_pos": object_pos,
                    "object_quat": object_quat,
                }
            ]
        },
        path,
    )
    print(f"\n[pose_lab] saved pose '{stage_name}' -> {path}", flush=True)
    return True


def run_selftest(env, eid, dev, start_q, idx, lab: LabHooks) -> int:  # noqa: C901
    """Drive every control surface non-interactively and report PASS/FAIL per check.

    Each reset is checked in two halves: first that a deliberate off-pose write
    actually reaches the solver, then that the reset brings it back. Checking
    only the "came back" half would pass trivially on a simulation that discards
    every write, because nothing ever left -- which is exactly the failure mode
    this tool suffered from.

    Every reset here is fired through :attr:`LabHooks.click`, the same callable
    object the buttons are bound to, and every step goes through
    :attr:`LabHooks.tick`, the same iteration the interactive loop runs. The
    panel callback is driven through :class:`_StubImgui`, and under Kit the
    ``omni.ui`` button is asked to dispatch its own click.

    Args:
        env: The running environment.
        eid: Environment indices to exercise. Shape is (1,).
        dev: Torch device the simulation runs on.
        start_q: The pose the lab started from [rad]. Shape is (1, num_joints).
        idx: Joint name to articulation joint index.
        lab: Live entry points into the running lab.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
        SPATULA_SPAWN_POS,
        SPATULA_SPAWN_QUAT,
    )

    robot = env.scene["robot"]
    spatula = env.scene["spatula"]
    limits = robot.data.joint_pos_limits.torch[0]
    log = lab.pending["log"]
    results = []

    def _tick(count=1):
        for _ in range(count):
            lab.tick()

    def _clamp(name, value):
        i = idx[name]
        return min(max(float(value), limits[i, 0].item()), limits[i, 1].item())

    def _spatula_pos():
        return (spatula.data.root_pos_w.torch[eid[0]] - env.scene.env_origins[eid[0]]).tolist()

    editable_ids = [idx[n] for n in lab.editable]

    def _arm_err():
        """Return the worst right arm/hand deviation from the start pose [rad].

        Restricted to the editable columns for the same reason
        :data:`ARM_RESET_TOL` is: the dangling legs of this fixed-base robot
        drift far and fast, and have nothing to do with either reset.
        """
        return (robot.data.joint_pos.torch[:1][:, editable_ids] - start_q[:, editable_ids]).abs().max().item()

    def _record(name, ok, detail):
        results.append(bool(ok))
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    def _check(name, measured, tol, greater):
        # Non-finite is failed EXPLICITLY, never left to comparison semantics:
        # ``nan < tol`` is False so a "must be small" check happens to fail, but
        # ``nan > tol`` is False too, so a "must be large" check fails for the
        # wrong reason and the message would blame the measurement instead of
        # the divergence that produced it.
        if not math.isfinite(measured):
            _record(name, False, f"{measured} -- non-finite, the simulation has diverged")
            return
        ok = measured > tol if greater else measured < tol
        _record(name, ok, f"{measured:.5f} (need {'>' if greater else '<'} {tol})")

    def _lines_since(mark, needle):
        return [line for line in log[mark:] if needle in line]

    def _verdicts_since(mark):
        return [line for line in log[mark:] if " PASS" in line or " FAIL" in line]

    def _fire(name, frames=RESET_SETTLE_FRAMES):
        """Press a reset button the way a user does, then let the verdict land.

        Stops on the frame the lab reports its verdict, so an independent
        measurement taken straight after is of the same instant the lab judged --
        which is the only way the two numbers can be compared at all.
        """
        mark = len(log)
        lab.click[name]()
        _tick(frames)
        return _verdicts_since(mark)

    _tick(5)
    spawn = list(SPATULA_SPAWN_POS)

    # -- spatula, half 1: prove the write channel reaches the solver at all ----
    shove = torch.tensor([[spawn[0] + 0.15, spawn[1] + 0.10, spawn[2] + 0.12, *SPATULA_SPAWN_QUAT]], device=dev)
    shove[:, :3] += env.scene.env_origins[eid]
    spatula.write_root_pose_to_sim_index(root_pose=shove, env_ids=eid)
    spatula.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=eid)
    _tick(2)
    _check("spatula write channel live", math.dist(_spatula_pos(), spawn), 0.05, greater=True)

    # -- spatula, half 2: RESET SPATULA, through the button's own callable -----
    verdicts = _fire("reset_spatula")
    _check("spatula returned to spawn", math.dist(_spatula_pos(), spawn), SPATULA_RESET_TOL, greater=False)
    _record(
        "RESET SPATULA reported its own verdict",
        len(verdicts) == 1 and verdicts[0].endswith("PASS"),
        verdicts[0] if verdicts else "the button produced no verdict line",
    )

    # -- arm, half 1: the slider path drives the arm off the start pose --------
    # This is the partial-column write the interactive loop uses every frame; a
    # full-width write would not cover it, and a column-order slip there would
    # move the wrong joints.
    targets_before = robot.data.joint_pos_target.torch[:1].clone()
    for name in RIGHT_ARM:
        lab.vals[name] = _clamp(name, start_q[0, idx[name]].item() + ARM_DISPLACE_BY)
    # The commanded target is rate limited (see MAX_TARGET_SPEED_DEFAULT), so the
    # window is DERIVED from the rate actually in force rather than hardcoded: the
    # target needs ARM_DISPLACE_BY / MAX_TARGET_SPEED seconds to arrive, and the arm
    # then has 60 frames to track it. With --max_target_speed 0 the rate is math.inf,
    # so the slew is zero frames -- the target snaps -- and nothing divides by zero.
    # Only the window moves with the rate; the gate it must clear, ARM_DISPLACED_TOL,
    # is untouched.
    # The cap goes INSIDE the ceil: see SELFTEST_MAX_SLEW_FRAMES. Capping the
    # ceil's RESULT cannot guard it, because the ceil is what raises.
    if math.isinf(MAX_TARGET_SPEED):
        slew = 0
    else:
        raw_slew = ARM_DISPLACE_BY / (MAX_TARGET_SPEED * env.cfg.sim.dt)
        slew = math.ceil(min(raw_slew, float(SELFTEST_MAX_SLEW_FRAMES)))
    # printed, not merely computed: it is the only externally visible proof that
    # the window tracks --max_target_speed instead of being hardcoded
    print(f"[selftest] slew window {slew} + 60 frames at {MAX_TARGET_SPEED} rad/s", flush=True)
    _tick(slew + 60)
    _check("slider path moved the arm off start", _arm_err(), ARM_DISPLACED_TOL, greater=True)

    # -- arm, half 1b: that write must not touch the legs / waist / left arm ---
    others = torch.ones(targets_before.shape[1], dtype=torch.bool, device=dev)
    others[editable_ids] = False
    drift = (robot.data.joint_pos_target.torch[:1][:, others] - targets_before[:, others]).abs().max().item()
    _check("non-editable PD targets untouched", drift, 1e-6, greater=False)

    # -- arm, half 2: RESET ARM, through the button's own callable -------------
    verdicts = _fire("reset_arm")
    _check("arm returned to start", _arm_err(), ARM_RESET_TOL, greater=False)
    _record(
        "RESET ARM reported its own verdict",
        len(verdicts) == 1 and verdicts[0].endswith("PASS"),
        verdicts[0] if verdicts else "the button produced no verdict line",
    )

    # -- arm, half 3: and the pose must then STAY put ---------------------------
    # The tool's whole promise is that the arm holds where you put it. Only
    # meaningful from a free-space start: the claw poses press the hand into the
    # tabletop, which is not an equilibrium the PD can hold, so the arm drifts
    # out of contact by design rather than by defect.
    if args_cli.pose in ("hover", "default"):
        _tick(30)
        _check("arm still on target 30 frames later", _arm_err(), ARM_RESET_TOL, greater=False)
    else:
        print(
            f"[selftest] SKIP arm still on target 30 frames later: the '{args_cli.pose}' start pose"
            " presses the hand into the tabletop, so it is not a PD equilibrium to hold",
            flush=True,
        )

    # -- both resets in ONE frame: both must still report ----------------------
    # Regression guard: a single shared verdict slot silently dropped whichever
    # reset was queued first, which looks exactly like a dead button.
    mark = len(log)
    lab.click["reset_arm"]()
    lab.click["reset_spatula"]()
    _tick(RESET_SETTLE_FRAMES + 2)
    verdicts = _verdicts_since(mark)
    arm_v = [v for v in verdicts if " arm:" in v]
    spatula_v = [v for v in verdicts if " spatula:" in v]
    _record(
        # Both halves must PASS, not merely be EMITTED. Asserting only that the
        # lines exist let this check report PASS while quoting two failing
        # sub-verdicts -- which is how a fully diverged run once printed
        # "ALL CHECKS PASSED".
        "both resets in one frame both report and both pass",
        len(arm_v) == 1 and arm_v[0].endswith("PASS") and len(spatula_v) == 1 and spatula_v[0].endswith("PASS"),
        " | ".join(verdicts) if verdicts else "neither reset produced a verdict line",
    )

    # -- the Newton side-panel callback ----------------------------------------
    mark = len(log)
    stub = _StubImgui(click="RESET ARM##poselab")
    lab.draw_panel(stub)
    _record(
        "Newton panel button queues RESET ARM",
        bool(_lines_since(mark, "queued: RESET ARM")) and "RESET ARM##poselab" in stub.buttons,
        f"drew {len(stub.buttons)} buttons and {len(stub.sliders)} sliders",
    )
    _tick(RESET_SETTLE_FRAMES + 2)

    probe = "right_elbow_joint"
    want = _clamp(probe, lab.vals[probe] + 0.10)
    lab.draw_panel(_StubImgui(drag=(f"{_short(probe)}##poselab", want)))
    _record(
        "Newton panel slider writes the pose",
        abs(lab.vals[probe] - want) < 1e-6,
        f"{probe} = {lab.vals[probe]:+.4f}, dragged to {want:+.4f}",
    )
    _fire("reset_arm")

    # -- the omni.ui button's own dispatch (Kit sessions only) -----------------
    button = lab.buttons.get("reset_spatula")
    if button is None:
        print("[selftest] SKIP omni.ui button dispatch: no Kit runtime in this session", flush=True)
    elif not hasattr(button, "call_clicked_fn"):
        print("[selftest] SKIP omni.ui button dispatch: this omni.ui build has no Button.call_clicked_fn()", flush=True)
    else:
        mark = len(log)
        button.call_clicked_fn()
        _record(
            "omni.ui button dispatch queues RESET SPATULA",
            bool(_lines_since(mark, "queued: RESET SPATULA")),
            "the widget's own clicked_fn fired",
        )
        _tick(RESET_SETTLE_FRAMES + 2)

    # -- the health gate itself: a gate that can never fire also "passes" -------
    # Pure-function positive control, no simulation involved: feed
    # :func:`health_report` states it MUST classify, including the exact 5.9e7 m/s
    # pre-overflow reading the old isfinite-only latch let through while printing
    # ALL CHECKS PASSED, the legitimate 12 rad/s this selftest itself produces, and
    # the 62.3 rad/s peak measured on a full-amplitude snap that stayed clean for
    # its whole window -- that last case pins RUNAWAY_JOINT_SPEED above the worst
    # legitimate transient, so lowering the gate back under it fails here instead
    # of in a user's session.
    ids = [0, 1, 2, 3]
    ok_q = torch.zeros(1, 4, device=dev)
    ok_p = torch.tensor([list(SPATULA_SPAWN_POS)], device=dev)
    ok_v = torch.zeros(1, 6, device=dev)
    spawn_t = torch.tensor(SPATULA_SPAWN_POS, device=dev)

    def _kind(qp=None, qv=None, op=None, ov=None):
        return health_report(
            ok_q if qp is None else qp,
            ok_q if qv is None else qv,
            ids,
            ok_p if op is None else op,
            ok_v if ov is None else ov,
            spawn_t,
        )[0]

    fast_v = ok_v.clone()
    fast_v[0, 0] = 5.9e7
    far_p = ok_p.clone()
    far_p[0, 0] += 1.0e3
    fast_q = ok_q.clone()
    fast_q[0, 0] = 1.0e3
    nan_q = ok_q.clone()
    nan_q[0, 0] = float("nan")
    both = fast_v.clone()
    both[0, 1] = float("inf")
    cases = {
        "at rest": (_kind(), None),
        "12 rad/s arm (what this selftest produces)": (_kind(qv=torch.full((1, 4), 12.0, device=dev)), None),
        "62.3 rad/s arm (measured peak of a clean full-amplitude snap)": (
            _kind(qv=torch.full((1, 4), 62.3, device=dev)),
            None,
        ),
        "5.9e7 m/s spatula, still finite": (_kind(ov=fast_v), "runaway"),
        "spatula 1 km from spawn": (_kind(op=far_p), "runaway"),
        "1000 rad/s joint": (_kind(qv=fast_q), "runaway"),
        "NaN joint position": (_kind(qp=nan_q), "nonfinite"),
        "non-finite outranks runaway": (_kind(ov=both), "nonfinite"),
    }
    wrong = [f"{k}: got {got!r}, want {want!r}" for k, (got, want) in cases.items() if got != want]
    _record(
        "health gate classifies known states correctly",
        not wrong,
        " | ".join(wrong) if wrong else f"{len(cases)}/{len(cases)} states classified",
    )

    # -- the rate-limit floor, against the LIVE dt and the LIVE joint limits ----
    # :data:`MIN_TARGET_SPEED` is derived at parse time from two assumptions --
    # :data:`CONTROL_RATE_HZ` and :data:`TARGET_MAGNITUDE_BOUND` -- because no
    # simulation exists yet to ask. One exists here, so the derivation is checked
    # against it instead of trusted, and pinned on BOTH sides: a floor too low
    # would go on accepting rates that freeze every control surface, and a floor
    # raised arbitrarily would reject rates that work.
    worst_target = limits[editable_ids].abs().max().item()
    floor_step = MIN_TARGET_SPEED * env.cfg.sim.dt
    _record(
        "rate-limit floor moves the widest authorable target",
        worst_target <= TARGET_MAGNITUDE_BOUND and worst_target + floor_step != worst_target,
        f"widest editable stop {worst_target:.4f} rad (assumed bound {TARGET_MAGNITUDE_BOUND}); the floor steps"
        f" {floor_step:.3e} rad at dt {env.cfg.sim.dt:.6f} s",
    )
    _record(
        "rate-limit floor is not set higher than the arithmetic needs",
        worst_target + floor_step / 10.0 == worst_target,
        f"a rate 10x under the floor steps {floor_step / 10.0:.3e} rad, which rounds back into"
        f" {worst_target:.4f} rad -- so the rejected side really is a stall",
    )

    # -- and the simulation must have stayed healthy THROUGHOUT -----------------
    # The catch-all. Every check above samples one quantity at one instant, and
    # the checks that read no numbers at all (UI plumbing) pass happily on a dead
    # simulation. ``diverged`` latches the FIRST trip of any kind and is never
    # cleared, so a run that auto-recovered still fails here -- recovery must
    # never turn a diverging selftest green.
    diverged = lab.pending["diverged"]
    q_peak, o_peak, r_peak = lab.pending["peaks"]
    _record(
        "simulation stayed healthy for the whole run",
        diverged is None,
        (
            f"no bound tripped; peak spatula range {r_peak:.3f} m"
            if diverged is None
            else f"health gate tripped at frame {diverged}, {lab.pending['recoveries']} recovery(ies)"
        ),
    )
    _check("peak watched joint speed under its gate", q_peak, RUNAWAY_JOINT_SPEED, greater=False)
    _check("peak spatula speed under its gate", o_peak, RUNAWAY_OBJECT_SPEED, greater=False)

    passed = all(results)
    print(f"[selftest] {'ALL CHECKS PASSED' if passed else 'FAILURES DETECTED'}", flush=True)
    return 0 if passed else 1


def main():  # noqa: C901
    """Open the viewer, hold the pose, and stream joint values.

    Complexity is over the linter default: this is an interactive authoring tool
    whose body is a UI layout plus an event loop, and splitting it would spread
    shared closure state across helpers for no readability gain.
    """
    import json
    import os
    import signal
    import time
    import traceback

    env_cfg = parse_env_cfg(TASK, num_envs=1)
    env_cfg.scene.num_envs = 1
    # Deterministic authoring: nothing may yank the pose away mid-edit.
    # ``reset_pregrasp`` is the ONLY randomizing term EventCfg still has -- the
    # two ``randomize_*`` terms were removed from the task, so there is no
    # exploration noise left to disable here.
    env_cfg.events.reset_pregrasp = None
    # Newton, not the PhysX default: the spatula's raw triangle colliders are
    # rejected on a dynamic body by PhysX, and the right-hand contact sensors
    # fail to initialise there. This script has no `presets=` CLI hook.
    from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import PhysicsCfg

    env_cfg.sim.physics = PhysicsCfg().newton_mjwarp
    # MJWarp pushes Newton state -> MuJoCo qpos only when
    # ``solver._step % update_data_interval == 0``. The whole step is CUDA-graph
    # captured, so that branch is evaluated ONCE at capture time and baked in --
    # and the Kit path warms up one step before capturing, landing on an odd
    # _step where the sync is omitted from the graph forever. Every teleport is
    # then overwritten by the next step's qpos -> state writeback, while the
    # readback (a zero-copy view of the write buffer) still reports success.
    # Interval 1 is true at any capture step, so it is parity-proof. This is an
    # authoring tool; throughput is irrelevant. PhysicsCfg() deep-copies its
    # members (configclass post-init), so this cannot reach a training run.
    #
    # SCOPE (measured, not inferred -- see the probes referenced below). This is
    # NOT a pose-lab bug and this line does not fix it anywhere else:
    #   * It is a SOLVER-level fault, not an event-level one. Manager resets
    #     (``reset_scene_to_default``, ``reset_root_state_uniform``,
    #     ``reset_joints_by_offset``) and bare ``write_*_to_sim_index`` calls with
    #     no manager involved were ALL measured dead, so direct-workflow
    #     ``_reset_idx``, ``env.reset_to()`` and teleop teleports are dead too.
    #   * It bites exactly the GUI path: ``--visualizer kit`` (play, record,
    #     teleop, any interactive tool). Headless, ``--visualizer newton`` and
    #     ``--video`` training were each measured UNAFFECTED -- they capture
    #     eagerly at ``_step == 0``, where the sync is recorded at any interval.
    #   * 12 configs in this repo ship ``update_data_interval = 2``, including
    #     this task's own preset (which is why the override is needed here). Only
    #     the spatula task was measured; the other 11 share the code path.
    #   * Under CUDA-graph capture the setting is binary, not a frequency: sync
    #     every step (even capture parity) or never (odd parity).
    # The durable fix is upstream in ``isaaclab_newton``: make the Kit warm-up in
    # ``NewtonManager._capture_relaxed_graph`` step-count-neutral so capture always
    # lands on ``_step == 0``. That is an out-of-scope change for an authoring tool.
    env_cfg.sim.physics.solver_cfg.update_data_interval = 1

    viz_label = ",".join(args_cli.visualizer) if args_cli.visualizer else "none"

    with launch_simulation(env_cfg, args_cli):
        from isaaclab_tasks.contrib.g1_spatula_lift.g1_spatula_lift_env_cfg import (
            FINGERTIP_BODY_NAMES,
            HANDLE_SEGMENT_P0_B,
            HANDLE_SEGMENT_P1_B,
            LAB_TABLE_HEIGHT,
            PREGRASP_JOINT_POS,
            SPATULA_SPAWN_POS,
        )
        from isaaclab_tasks.contrib.g1_spatula_lift.mdp.functions import _handle_segment_geometry

        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        dev = env.device
        eid = torch.tensor([0], device=dev)
        sim_dt = env.cfg.sim.dt

        fc = SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)
        fc.resolve(env.scene)
        oc = SceneEntityCfg("spatula")
        oc.resolve(env.scene)
        tip_names = [robot.body_names[i] for i in fc.body_ids]

        editable = RIGHT_ARM + RIGHT_HAND
        idx = {n: robot.joint_names.index(n) for n in editable}
        editable_ids = [idx[n] for n in editable]
        joint_limits = robot.data.joint_pos_limits.torch[0]
        lim = {n: (joint_limits[idx[n], 0].item(), joint_limits[idx[n], 1].item()) for n in editable}

        def _clamp_into_limits(name, joint_index, value):
            """Clamp a joint value [rad] into its limits, announcing any change."""
            lo, hi = joint_limits[joint_index, 0].item(), joint_limits[joint_index, 1].item()
            clamped = min(max(float(value), lo), hi)
            if abs(clamped - float(value)) > 1e-6:
                print(
                    f"[pose_lab] {name} clamped {float(value):+.4f} -> {clamped:+.4f} (outside its joint limits)",
                    flush=True,
                )
            return clamped

        # ---- starting pose -------------------------------------------------
        q = robot.data.default_joint_pos.torch[:1].clone()
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args_cli.pose == "map":
            gm = torch.load(os.path.join(here, "grasp_map.pt"), weights_only=True)
            stage = gm["stages"][0]
            gmq = stage["joint_pos"].to(dev)
            obj_z = float(stage["object_pos"][2])
            print("[pose_lab] START POSE: grasp_map.pt stage 'pre_grasp_caged' (CURLED claw)")
            print(
                f"[pose_lab] WARNING: grasp_map.pt is STALE -- its object z is {obj_z:.4f} m,"
                f" {100 * (LAB_TABLE_HEIGHT - obj_z):.1f} cm below the current tabletop"
                f" (LAB_TABLE_HEIGHT {LAB_TABLE_HEIGHT:.2f} m). Only the right arm + hand are applied."
            )
            # right arm + hand ONLY: the map's other joints are a stale
            # pre-fixed-base crouch, and writing all 43 would also stomp the
            # PD targets that `hold_inert_joints` just set to the defaults.
            # Clamped like the `dict` branch: the map was authored past at least
            # one joint's limit, and both slider widgets clamp on their first
            # frame -- so an unclamped load silently drifts off the pose it
            # just claimed to have restored.
            for n in editable:
                q[0, idx[n]] = _clamp_into_limits(n, idx[n], gmq[idx[n]])
        elif args_cli.pose == "dict":
            for n, v in PREGRASP_JOINT_POS.items():
                if n in robot.joint_names:
                    i = robot.joint_names.index(n)
                    q[0, i] = _clamp_into_limits(n, i, v)
            print("[pose_lab] START POSE: PREGRASP_JOINT_POS (CURLED claw)")
        else:
            print("[pose_lab] START POSE: env default DEFAULT_ARM_JOINT_POS (neutral hover, fingers open)")

        # write the target ALONGSIDE the state: writing state alone leaves the PD
        # target at the previous pose, so the controller visibly drags the arm
        # into place over the next frames instead of it simply being there.
        robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=eid)
        robot.set_joint_position_target_index(target=q, env_ids=eid)
        robot.write_data_to_sim()

        # ---- shared state ----------------------------------------------------
        # ``vals`` is the SINGLE source of truth for the editable joints. Every
        # control surface writes it and nothing else; the main loop turns it into
        # PD targets once per iteration. UI callbacks additionally queue commands
        # in ``pending`` -- they never touch simulation buffers themselves, which
        # keeps both surfaces byte-identical in behaviour.
        start_q = robot.data.joint_pos.torch[:1].clone()
        vals = {n: start_q[0, idx[n]].item() for n in editable}
        # The AUTHORED pose (``vals``) is what the control surfaces edit; the
        # COMMANDED PD target (``cmd``) chases it at MAX_TARGET_SPEED. Keeping
        # them separate is what stops a large edit from being a force impulse,
        # while leaving every control surface writing plain joint angles.
        cmd = dict(vals)
        pending = {
            "reset_arm": False,
            "reset_spatula": False,
            # frame of the FIRST health-gate trip of ANY kind, latched forever.
            # Unchanged meaning, so the selftest's catch-all still fails a run
            # that recovered -- auto-recovery must never turn a diverging
            # selftest green.
            "diverged": None,
            # frame of the first UNRECOVERABLE trip: non-finite state, or the
            # recovery budget exhausted. Only this one stops the interactive loop.
            "fatal": None,
            "recoveries": 0,
            "mute_until": 0,
            # running max of (joint speed [rad/s], spatula speed [m/s], spatula range [m])
            "peaks": [0.0, 0.0, 0.0],
            "frames": 0,
            # a LIST, not one slot: two resets can be queued in the same frame
            # (both buttons, or a pose file setting both sentinels), and a single
            # slot silently dropped the first one's verdict
            "checks": [],
            "status": "ready",
            "clicks": 0,
            "log": [],
        }

        print("[pose_lab] startup joint values [rad]:")
        for n in editable:
            print(f"    {n:32s} {vals[n]:+.4f}")

        # ---- live pose file --------------------------------------------------
        # Dragging joints in the viewport does not work while the solver is
        # stepping: Newton owns the articulation state and re-asserts it every
        # step, so viewport edits are overwritten instantly. Edit this JSON file
        # instead and the pose updates as soon as you save.
        pose_file = args_cli.pose_file
        # Deliberately NOT ``pose_file``: the loop watches that path's mtime, so
        # writing the authored pose back there would re-apply, on the very next
        # iteration, the pose that just diverged the scene.
        recovery_file = os.path.splitext(pose_file)[0] + "_recovered.json"

        def dump(path):
            """Write the current editable-joint values [rad] to *path* as JSON."""
            with open(path, "w") as fh:
                json.dump({n: round(vals[n], 4) for n in editable}, fh, indent=2)

        # ---- status + command queue ------------------------------------------
        sliders = {}
        status_label = {}
        ui_win = {"win": None, "reason": ""}
        buttons = {}

        def _say(msg):
            """Print a status line and mirror it into every control surface."""
            pending["clicks"] += 1
            line = f"#{pending['clicks']} {msg}"
            pending["status"] = line
            pending["log"].append(line)
            print(f"[pose_lab] {line}", flush=True)
            lab = status_label.get("l")
            if lab is not None:
                lab.text = line

        def _queue(what):
            """Queue a reset for the next :func:`_tick` to drain."""
            pending[what] = True
            _say(f"queued: {what.replace('_', ' ').upper()}")

        # Bound once and shared: the buttons, the pose-file sentinels and the
        # selftest all hold THIS callable, so exercising one exercises all.
        click = {name: (lambda n=name: _queue(n)) for name in ("reset_arm", "reset_spatula")}

        def _resync_sliders():
            """Push ``vals`` back into the omni.ui sliders after a programmatic change."""
            for n, sl in sliders.items():
                sl.model.set_value(vals[n])

        # ---- omni.ui window --------------------------------------------------
        try:
            import omni.ui as ui

            win = ui.Window("Spatula Pose Lab", width=460, height=600)
            ui_win["win"] = win

            def _make_cb(joint_name):
                def _cb(model):
                    vals[joint_name] = model.get_value_as_float()

                return _cb

            with win.frame:
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=3, height=0):
                        with ui.HStack(height=32, spacing=6):
                            buttons["reset_arm"] = ui.Button("RESET ARM", clicked_fn=click["reset_arm"])
                            buttons["reset_spatula"] = ui.Button("RESET SPATULA", clicked_fn=click["reset_spatula"])
                        status_label["l"] = ui.Label(pending["status"], height=22, word_wrap=True)
                        ui.Spacer(height=6)
                        ui.Label("RIGHT ARM", height=22)
                        for n in RIGHT_ARM + ["__SEP__"] + RIGHT_HAND:
                            if n == "__SEP__":
                                ui.Spacer(height=8)
                                ui.Label("RIGHT HAND", height=22)
                                continue
                            lo, hi = lim[n]
                            with ui.HStack(height=24, spacing=6):
                                ui.Label(_short(n), width=150)
                                sl = ui.FloatSlider(min=lo, max=hi, step=0.005)
                                sl.model.set_value(vals[n])
                                sl.model.add_value_changed_fn(_make_cb(n))
                                sliders[n] = sl
        except ImportError as err:
            # omni.ui ships only as a Kit extension payload, so it cannot exist on
            # the kitless Newton path. Anything OTHER than an import failure is a
            # real layout bug and is deliberately left to propagate.
            ui_win["reason"] = f"{err} -- there is no Kit runtime under --visualizer newton; use --visualizer kit"

        # ---- Newton side panel -----------------------------------------------
        panel = {"registered": False, "reason": "no Newton visualizer in this session"}

        def _panel(imgui):
            """Draw the pose-lab controls inside the Newton viewer's side panel."""
            imgui.separator()
            imgui.text("Spatula Pose Lab")
            if imgui.button("RESET ARM##poselab"):
                click["reset_arm"]()
            if imgui.button("RESET SPATULA##poselab"):
                click["reset_spatula"]()
            imgui.text(pending["status"])
            # immediate mode: re-reading ``vals`` every frame IS the slider resync
            for n in editable:
                lo, hi = lim[n]
                changed, v = imgui.slider_float(f"{_short(n)}##poselab", vals[n], lo, hi)
                if changed:
                    vals[n] = v

        def _try_register_newton_panel() -> bool:
            """Attach :func:`_panel` to the Newton viewer; a no-op once attached.

            Called before the loop and again each iteration until it succeeds, so
            it cannot lose a race with viewer creation.

            Returns:
                Whether the panel is registered.
            """
            if panel["registered"]:
                return True
            for viz in env.sim.visualizers:
                if "newton" not in type(viz).__name__.lower():
                    continue
                # ``_viewer`` is private but there is no public accessor for it,
                # and it is the only route to ``register_ui_callback``.
                viewer = getattr(viz, "_viewer", None)
                if viewer is None:
                    continue  # viewer not built yet; retry next iteration
                if not hasattr(viewer, "register_ui_callback"):
                    panel["reason"] = (
                        f"{type(viz).__name__} is present but its viewer ({type(viewer).__name__}) has no"
                        " register_ui_callback(); the Newton API moved"
                    )
                    continue
                if getattr(viewer, "gui", None) is None:
                    # ViewerGL builds its ImGui layer only outside headless mode,
                    # and register_ui_callback() is a SILENT no-op without it --
                    # so this must be checked, or the banner would report a panel
                    # that does not exist. Headless is also inferred from a
                    # missing DISPLAY, so an SSH session lands here.
                    panel["reason"] = (
                        f"{type(viz).__name__} has no ImGui layer (headless viewer: no DISPLAY, or"
                        " NewtonVisualizerCfg.headless); use the pose file below"
                    )
                    continue
                viewer.register_ui_callback(_panel, position="side")
                panel["registered"] = True
                panel["reason"] = 'left panel > "Example Options" > "Spatula Pose Lab"'
                return True
            return False

        _try_register_newton_panel()

        # ---- one lab iteration -----------------------------------------------
        # The interactive loop and the selftest BOTH go through this, so a check
        # that passes here cannot pass against code the buttons do not run.
        def _drain():
            """Apply any queued resets, then schedule their verdicts.

            Runs outside every UI callback, so exactly one writer touches the
            simulation and a failure is reported rather than dying inside the UI.
            """
            if pending["reset_arm"]:
                pending["reset_arm"] = False
                try:
                    reset_arm(env, eid, start_q, editable_ids)
                    for n in editable:
                        vals[n] = start_q[0, idx[n]].item()
                        # snap, do NOT slew: reset_arm teleports the joint STATE,
                        # so a commanded target still walking back from the old
                        # pose would immediately drag the arm off the pose the
                        # reset just restored -- and the verdict would judge that
                        # drag rather than the reset.
                        cmd[n] = vals[n]
                    _resync_sliders()
                    pending["checks"].append(("arm", start_q[:, editable_ids].clone(), RESET_SETTLE_FRAMES))
                except Exception as err:
                    # "ERRORED", never "FAILED": run_selftest._verdicts_since
                    # selects log lines containing " PASS" or " FAIL", so a
                    # "RESET ARM FAILED" line would land inside a verdict window
                    # and break the "exactly one verdict" checks.
                    _say(f"RESET ARM ERRORED: {type(err).__name__}: {err}")
                    traceback.print_exc()
            if pending["reset_spatula"]:
                pending["reset_spatula"] = False
                try:
                    pending["checks"].append(("spatula", reset_spatula(env, eid, dev), RESET_SETTLE_FRAMES))
                except Exception as err:
                    # "ERRORED", never "FAILED" -- see the reset-arm branch above.
                    _say(f"RESET SPATULA ERRORED: {type(err).__name__}: {err}")
                    traceback.print_exc()

        def _report_due_checks():
            """Report every scheduled verdict whose settle window has elapsed."""
            still_waiting = []
            for kind, ref, frames in pending["checks"]:
                if frames > 1:
                    still_waiting.append((kind, ref, frames - 1))
                elif kind == "arm":
                    err_q = (robot.data.joint_pos.torch[:1][:, editable_ids] - ref).abs().max().item()
                    verdict = "PASS" if err_q < ARM_RESET_TOL else "FAIL"
                    _say(f"arm: max right arm/hand error {err_q:.4f} rad (tol {ARM_RESET_TOL})  {verdict}")
                else:
                    now_p = (
                        env.scene["spatula"].data.root_pos_w.torch[eid[0]] - env.scene.env_origins[eid[0]]
                    ).tolist()
                    off = math.dist(now_p, SPATULA_SPAWN_POS)
                    verdict = "PASS" if off < SPATULA_RESET_TOL else "FAIL"
                    _say(
                        f"spatula: moved {math.dist(now_p, ref):.4f} m -> {off:.4f} m from spawn"
                        f" (tol {SPATULA_RESET_TOL})  {verdict}"
                    )
            pending["checks"] = still_waiting

        def _recover(frame, detail):
            """Restore the scene after a FINITE runaway and keep the session alive.

            Full-width joint state, unlike :func:`reset_arm`: a runaway is not
            confined to the editable columns, so the legs / waist / left arm are
            restored and zeroed too. Only the editable PD TARGETS are written --
            the inert ones belong to ``hold_inert_joints``.

            The arm goes back to the lab's START pose, not to the last authored
            one: a large authored jump is the measured cause of a runaway, so
            restoring into it re-diverges on the next frame. The authored pose is
            written to ``recovery_file`` first, so nothing is lost.
            """
            pending["recoveries"] += 1
            try:
                dump(recovery_file)  # BEFORE vals is overwritten
                saved = recovery_file
            except OSError as err:
                saved = f"<could not be written: {err}>"
            robot.write_joint_state_to_sim_index(position=start_q, velocity=torch.zeros_like(start_q), env_ids=eid)
            robot.set_joint_position_target_index(target=start_q[:, editable_ids], joint_ids=editable_ids, env_ids=eid)
            robot.write_data_to_sim()
            reset_spatula(env, eid, dev)
            for n in editable:
                vals[n] = cmd[n] = start_q[0, idx[n]].item()
            _resync_sliders()
            dropped = len(pending["checks"])
            pending["checks"] = []
            pending["mute_until"] = pending["frames"] + RECOVERY_MUTE_FRAMES
            _say(
                f"SIMULATION RUNAWAY at frame {frame}: {detail}. Scene restored to the start pose and"
                f" the spatula to spawn; your authored pose was written to {saved}."
                f" Recovery {pending['recoveries']}/{MAX_RECOVERIES}."
                + (f" {dropped} pending reset verdict(s) dropped." if dropped else "")
            )

        watched = SceneEntityCfg("robot", joint_names=RUNAWAY_JOINT_NAMES)
        watched.resolve(env.scene)
        # list(), not a bare pass-through: if the regexes ever matched every joint
        # SceneEntityCfg would collapse them to slice(None), and list() raises here
        # rather than silently folding the fingers into the gate.
        watched_ids = list(watched.joint_ids)
        spawn_t = torch.tensor(SPATULA_SPAWN_POS, device=dev)

        def _check_health():
            """Classify this frame, recover from a runaway, stop on a non-finite state.

            One device sync per step (inside :func:`health_report`), far cheaper
            than the render this loop already performs every step. Without it a
            divergence is only noticed if some later readout happens to sample
            it, and the purely structural checks report success on a dead
            simulation.
            """
            spatula = env.scene["spatula"]
            kind, detail, peaks = health_report(
                robot.data.joint_pos.torch,
                robot.data.joint_vel.torch,
                watched_ids,
                spatula.data.root_pos_w.torch - env.scene.env_origins,
                spatula.data.root_vel_w.torch,
                spawn_t,
            )
            for i, value in enumerate(peaks):
                if math.isfinite(value) and value > pending["peaks"][i]:
                    pending["peaks"][i] = value
            if pending["fatal"] is not None:
                return
            if kind is None:
                if pending["mute_until"] and pending["frames"] >= pending["mute_until"]:
                    pending["mute_until"] = 0
                    _say(f"recovery {pending['recoveries']} held for {RECOVERY_MUTE_FRAMES} frames; continuing")
                return
            if pending["diverged"] is None:
                pending["diverged"] = pending["frames"]
            if pending["frames"] < pending["mute_until"]:
                return  # the teleport's own transient, not a fresh runaway
            if kind == "runaway" and pending["recoveries"] < MAX_RECOVERIES:
                _recover(pending["frames"], detail)
                return
            pending["fatal"] = pending["frames"]
            why = (
                f"{detail}; a state write does not clear the solver's warm-start and contact buffers,"
                " and the step is CUDA-graph captured, so a restored pose would be re-poisoned on the"
                " next step"
                if kind == "nonfinite"
                else f"{detail}; {MAX_RECOVERIES} recoveries did not hold"
            )
            _say(f"SIMULATION DIVERGED at frame {pending['frames']}: {why}. Every reading from here on is meaningless.")

        def _tick():
            """Drain queued resets, drive the arm from ``vals``, step, then verify."""
            _drain()
            # Walk the COMMANDED target toward the authored pose at a bounded
            # rate. A step change here is a force impulse the contact solve
            # answers with a spatula ejected faster than it was struck; see
            # MAX_TARGET_SPEED_DEFAULT.
            limit = MAX_TARGET_SPEED * sim_dt
            if math.isinf(limit):
                # --max_target_speed 0: no limiter, byte-identical to the
                # pre-limiter behaviour of writing the authored pose straight out
                cmd.update(vals)
            else:
                for n in editable:
                    cmd[n] += max(-limit, min(limit, vals[n] - cmd[n]))
            # Only the editable columns are written, so the leg / waist / left arm
            # targets set by `hold_inert_joints` are never clobbered.
            # set_joint_position_target_index() only stages; write_data_to_sim is
            # the flush this loop needs because it drives sim.step directly.
            tgt = torch.tensor([[cmd[n] for n in editable]], device=dev)
            robot.set_joint_position_target_index(target=tgt, joint_ids=editable_ids, env_ids=eid)
            robot.write_data_to_sim()
            # Render whenever a window exists -- including under --selftest, so a
            # requested viewer is pumped instead of hanging unresponsive.
            env.sim.step(render=bool(env.sim.visualizers))
            env.scene.update(sim_dt)
            pending["frames"] += 1
            # before the verdicts, so a verdict computed on a dead frame is
            # preceded in the log by the reason it is meaningless
            _check_health()
            _report_due_checks()

        # ---- selftest --------------------------------------------------------
        if args_cli.selftest:
            hooks = LabHooks(
                tick=_tick,
                click=click,
                vals=vals,
                pending=pending,
                draw_panel=_panel,
                buttons=buttons,
                editable=editable,
            )
            code = run_selftest(env, eid, dev, start_q, idx, hooks)
            env.close()
            sys.exit(code)

        # ---- banner ----------------------------------------------------------
        dump(pose_file)
        ui_mark = "x" if ui_win["win"] is not None else " "
        ui_note = (
            'omni.ui window "Spatula Pose Lab"' if ui_win["win"] is not None else f"unavailable: {ui_win['reason']}"
        )
        print("\n" + "=" * 78)
        print(f"SPATULA POSE LAB   visualizer: {viz_label}   start pose: {args_cli.pose}")
        print("CONTROL SURFACES")
        print(f"  [{'x' if panel['registered'] else ' '}] Newton side panel  {panel['reason']}")
        print(f"  [{ui_mark}] omni.ui window     {ui_note}")
        print(f"  [x] pose file          edit {pose_file} and save; the pose follows.")
        print('                         "__reset_arm__": 1 or "__reset_spatula__": 1 fires a reset.')
        speed_note = f"{MAX_TARGET_SPEED:.2f} rad/s" if math.isfinite(MAX_TARGET_SPEED) else "OFF (targets snap)"
        print(f"TARGET RATE LIMIT  {speed_note}   (--max_target_speed; 0 disables)")
        print(
            f"HEALTH GATE        joint {RUNAWAY_JOINT_SPEED} rad/s | spatula {RUNAWAY_OBJECT_SPEED} m/s |"
            f" range {RUNAWAY_OBJECT_RANGE} m -> auto-recovery, max {MAX_RECOVERIES}"
        )
        print("NOTE the viewer's \"Pause Simulation\" button freezes THIS script's loop; a reset")
        print('     clicked while paused shows "queued" and applies on resume.')
        print("Goal: fingertip z near 0 (tips on the tabletop) and the thumb's local-y")
        print("opposite in sign to index/middle (the straddle).")
        print("Ctrl+C to exit.  --save writes the pose as a one-stage grasp map.")
        print("=" * 78, flush=True)

        last_mtime = os.path.getmtime(pose_file)
        last_print = 0.0
        # A visualizer that raises is removed from the list, so "the list is now
        # empty" has to end the run too -- otherwise a crashed viewer leaves this
        # loop spinning forever with nothing on screen.
        had_visualizers = bool(env.sim.visualizers)
        # Take SIGINT back before entering the loop. App startup installs its own
        # disposition, which left the banner's "Ctrl+C to exit" simply untrue:
        # measured, an interrupt aimed at the process died at exit 130 with the
        # `except KeyboardInterrupt` below never entered (so `--save` never ran),
        # and one aimed at the process GROUP was swallowed outright -- the run
        # kept going. `default_int_handler` is what a plain Python program has:
        # it raises KeyboardInterrupt in the main thread, which is exactly the
        # exception this loop is already written to catch and save from. Set here
        # rather than at startup so the selftest, which must not be interruptible
        # into a partial PASS, keeps the app's own handling.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            while True:
                _try_register_newton_panel()

                # ---- pose file ------------------------------------------------
                try:
                    m = os.path.getmtime(pose_file)
                except OSError:
                    m = last_mtime
                if m != last_mtime:
                    last_mtime = m
                    try:
                        with open(pose_file) as fh:
                            loaded = json.load(fh)
                    except Exception as err:
                        print(f"[pose_lab] bad JSON, ignoring: {err}", flush=True)
                        loaded = None
                    if loaded:
                        if loaded.pop("__reset_arm__", 0):
                            click["reset_arm"]()
                        if loaded.pop("__reset_spatula__", 0):
                            click["reset_spatula"]()
                        for n, v in loaded.items():
                            if n not in idx:
                                print(f"[pose_lab] ignoring '{n}': only the right arm + hand are editable", flush=True)
                                continue
                            vals[n] = _clamp_into_limits(n, idx[n], v)
                        _resync_sliders()
                        last_print = 0.0  # force a fresh readout

                _tick()

                # ---- stop on an UNRECOVERABLE divergence -----------------------
                # Streaming `nan` fingertip readouts at a user who is trying to
                # author a pose is worse than stopping: nothing that follows is a
                # measurement of anything. A finite runaway does NOT land here --
                # `_check_health` restores the scene and the session continues.
                if pending["fatal"] is not None:
                    print(
                        "[pose_lab] the simulation cannot be recovered in place -- restart the tool."
                        f" Your last authored pose is in {recovery_file}. If this followed a large pose"
                        " jump, make the jump in smaller steps or lower --max_target_speed.",
                        flush=True,
                    )
                    break

                # ---- exit when the last window closes --------------------------
                # `is_rendering` reads a carb setting, not the visualizer list, so
                # without this the loop spins on forever.
                if had_visualizers and not any(v.is_running() and not v.is_closed for v in env.sim.visualizers):
                    print("[pose_lab] viewer closed, exiting", flush=True)
                    break

                now = time.time()
                if now - last_print >= args_cli.print_every:
                    last_print = now
                    d, y = _handle_segment_geometry(env, HANDLE_SEGMENT_P0_B, HANDLE_SEGMENT_P1_B, fc, oc)
                    z = robot.data.body_pos_w.torch[:, fc.body_ids, 2] - env.scene.env_origins[:, 2:3]
                    ti = tip_names.index("right_hand_thumb_2_link")
                    ii = tip_names.index("right_hand_index_1_link")
                    print("\n--- FINGERTIPS ---", flush=True)
                    for k, n in enumerate(tip_names):
                        print(
                            f"    {n:28s} z {100 * (z[0, k].item() - LAB_TABLE_HEIGHT):+6.2f} cm | "
                            f"handle-dist {100 * d[0, k].item():5.2f} cm | local-y {100 * y[0, k].item():+6.2f} cm",
                            flush=True,
                        )
                    print(f"    straddle: {bool((y[0, ti] * y[0, ii] < 0).item())}", flush=True)
        except KeyboardInterrupt:
            pass
        wrote = True  # nothing was asked for, so nothing failed
        if args_cli.save:
            if pending["fatal"] is None and pending["recoveries"]:
                print(
                    f"[pose_lab] WARNING: {pending['recoveries']} runaway recovery/recoveries occurred, so the"
                    f" saved pose is the post-recovery START pose. Your authored pose is in {recovery_file}.",
                    flush=True,
                )
            # A divergence BREAKS the loop above and lands here, so the guards
            # live inside save_pose rather than in this branch: the dangerous
            # path is the default one.
            wrote = save_pose(
                args_cli.save,
                args_cli.stage_name,
                robot.data.joint_pos.torch[0].detach().cpu().clone(),
                (env.scene["spatula"].data.root_pos_w.torch[0] - env.scene.env_origins[0]).detach().cpu().clone(),
                env.scene["spatula"].data.root_quat_w.torch[0].detach().cpu().clone(),
                pending["fatal"],
            )
        env.close()
        # The refusal has to reach the SHELL, not just the terminal. `--save` is
        # this tool's scripted entry point -- `pose_lab.py --save p.pt &&
        # train.py p.pt` -- and exiting 0 after refusing to write hands the next
        # command either a missing file or, worse, a STALE p.pt from an earlier
        # good run, which the refusal message itself promises is still there.
        # That is the same "downstream reads a pose this run did not produce"
        # failure save_pose exists to prevent, so it must not survive the exit.
        if not wrote:
            sys.exit(1)


if __name__ == "__main__":
    main()
