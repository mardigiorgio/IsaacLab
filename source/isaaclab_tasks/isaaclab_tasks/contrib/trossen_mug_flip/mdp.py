# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flip-task MDP terms for the Trossen mug flip.

The generic layer comes from the slide's set (reach shaping, containment
terminations, action hygiene — the flip is a tabletop manipulation on the
same rig); flip-only terms live here. The task: the mug starts upside down
with its handle toward the arm, and the policy must right it BY THE HANDLE
at its own spot, then let go and settle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply

from isaaclab_tasks.contrib.trossen_mug_tree.hang_fsm_core import FsmInputs, HangFsm
from isaaclab_tasks.contrib.trossen_mug_tree.mdp import (  # noqa: F401  (they read env._fsm)
    fsm_metric_ms_grasp,
    fsm_metric_ms_lift,
    fsm_metric_ms_release,
    fsm_metric_ms_retreat,
    fsm_metric_regressions,
    fsm_metric_stage,
    fsm_milestones,
    fsm_shaping,
)
from isaaclab_tasks.contrib.trossen_mug_tree.mdp import fsm_metric_ms_insert as fsm_metric_ms_rotate  # noqa: F401

from isaaclab_tasks.contrib.trossen_mug_slide.mdp import *  # noqa: F401,F403
from isaaclab_tasks.contrib.trossen_mug_slide.mdp import _finite, _object_calm, _sensor_force_mag
from isaaclab_tasks.contrib.trossen_mug_lift.mdp import (  # noqa: F401
    _HANDLE_OFFSET_B,
    reset_arm_reverse_curriculum,
    SUCCESS_POS_THRESHOLD,
    SUCCESS_TILT_THRESHOLD,
    ObjectPoseSuccessCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Upright gate for the flip: the task IS the orientation, so the gate is the
# evaluator's strict cos(30 deg) — no slide-style loosening, a half-righted
# mug is not a righted mug.
UPRIGHT_MIN_COS = 0.87


def _up_cos(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the object's tilt: world-z component of the body z-axis.

    +1 upright, -1 upside down (the spawn), from the rotation-matrix zz
    entry of the root quaternion (x, y, z, w order in the data buffers).
    """
    quat = env.scene[object_cfg.name].data.root_quat_w.torch
    return 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])




def handle_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """``1 - tanh(d / std)`` from the end-effector to the mug's HANDLE (TRI's
    ``handle_middle`` frame carried by the live mug pose), not to the mug root.

    The root origin is the mug's bottom plane, so for the flip's inverted spawn
    ``object_ee_distance`` pointed the reach at the mug's TOP: 1000 iterations
    of hovering over the base, zero handle pinches (run 9uil59fv, 2026-08-28).
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    p = obj.data.root_pos_w.torch
    q = obj.data.root_quat_w.torch
    handle_w = p + quat_apply(q, torch.tensor(_HANDLE_OFFSET_B, device=p.device, dtype=p.dtype).expand_as(p))
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    return _finite(1.0 - torch.tanh(torch.norm(handle_w - ee_pos_w, dim=1) / std))


def handle_contact_count(
    env: ManagerBasedRLEnv,
    left_sensor: str = "pad_left_handle",
    right_sensor: str = "pad_right_handle",
    threshold: float = 0.01,
) -> torch.Tensor:
    """Number of finger pads on the HANDLE pieces (0, 1 or 2): the lift's
    ``pad_contact_count`` restricted to the handle sensors, so a one-pad brush
    of the handle earns partial credit and body contact earns none."""
    left = _sensor_force_mag(env, left_sensor) > threshold
    right = _sensor_force_mag(env, right_sensor) > threshold
    return _finite(left.float() + right.float())


# Finger collision block length along link_6 +x, from the rig USD (carriage joints
# at link_6 x = 0.0865, finger boxes 0..0.0696 beyond them; EE_TCP_OFFSET 0.087 is
# therefore the finger BASE and the fingertips sit FINGER_LEN further along the tool axis).
FINGER_LEN = 0.0696
# Reach/pinch target on the handle bar in the mug body frame: the LOW (rim-end)
# pinch point, 4.2 cm above the table when the mug is inverted (body z = rest
# height 0.119 - 0.042); handle_middle (z 0.058) is the fragile mid pinch.
HANDLE_PINCH_B = (0.062, 0.0, 0.077)


def handle_tip_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["follower_left_link_6"]),
) -> torch.Tensor:
    """``1 - tanh(d / std)`` from the FINGERTIP midpoint to the mug's handle_middle.

    The proven pinch (probe_generate_bank, 2026-08-28: 60-degree downward
    approach, jaws stop at 5.0/5.0 mm on the 11.7 mm bar, 111 N per pad on
    the handle pieces, 0 N on the body, mug raised 143 mm) has the fingertips
    at handle_middle and the TCP 70 mm back along the tool axis. A TCP-to-
    handle reach (run xw2q8ppv) asks for the finger roots on the bar, which
    puts 70 mm of finger through the 8 mm handle-wall gap: unlearnable.
    """
    return _finite(1.0 - torch.tanh(_tip_handle_distance(env, object_cfg, ee_frame_cfg, wrist_cfg) / std))


def _tip_handle_distance(env, object_cfg, ee_frame_cfg, wrist_cfg) -> torch.Tensor:
    """Fingertip-midpoint to handle_middle distance [m] (wrist_cfg must be resolved)."""
    obj = env.scene[object_cfg.name]
    robot = env.scene[wrist_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    p = obj.data.root_pos_w.torch
    q = obj.data.root_quat_w.torch
    handle_w = p + quat_apply(q, torch.tensor(HANDLE_PINCH_B, device=p.device, dtype=p.dtype).expand_as(p))
    tcp_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    wid = wrist_cfg.body_ids
    wrist_w = robot.data.body_pos_w.torch[:, wid[0] if isinstance(wid, list) else wid]
    if wrist_w.dim() == 3:
        wrist_w = wrist_w[:, 0]
    tool = tcp_w - wrist_w
    tool = tool / torch.linalg.vector_norm(tool, dim=-1, keepdim=True).clamp_min(1e-6)
    tip_w = tcp_w + FINGER_LEN * tool
    return torch.linalg.vector_norm(handle_w - tip_w, dim=-1)


def body_contact_without_handle(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    threshold: float = 1.0,
    handle_threshold: float = 0.01,
) -> torch.Tensor:
    """1.0 while the pads press the mug BODY above ``threshold`` and the handle is
    NOT held: a body grab. Body contact during a handle pinch is exempt --
    measured (probe_flip_bank_stages, 2026-08-28): a mug hanging in the jaws by
    its handle presses the fingertips with 5 N at the weakest squeeze and 70 N at
    a hard close (the bar sits 8 mm off the wall), so an unconditional fine made
    dropping the mug the rational policy in every lifted start."""
    body = _sensor_force_mag(env, sensor_name) > threshold
    held = handle_held(env, threshold=handle_threshold) > 0.5
    return _finite((body & ~held).float())


def handle_held(
    env: ManagerBasedRLEnv,
    left_sensor: str = "pad_left_handle",
    right_sensor: str = "pad_right_handle",
    threshold: float = 0.5,
) -> torch.Tensor:
    """1.0 while BOTH finger pads press the mug's handle pieces: the opposed
    handle pinch. One-pad contact is a brush, not a hold."""
    left = _sensor_force_mag(env, left_sensor) > threshold
    right = _sensor_force_mag(env, right_sensor) > threshold
    return _finite((left & right).float())


class upright_progress(ManagerTermBase):
    """Flip ratchet: 1.0 per ``min_improvement`` of NEW episode-best up-cosine,
    credited only while the handle is held and the mug is calm.

    The mug starts at up-cos -1; righting it sweeps to +1. Paying only on
    episode-best progress makes rocking it back and forth worthless, and the
    handle-hold gate makes shoving it over with the arm worthless — flipping
    BY THE HANDLE is the task. Ground given back cannot be re-earned.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best = torch.full((env.num_envs,), -2.0, device=env.device)
        # Per-episode latch: the mug crossed UPRIGHT while the handle was held.
        # upright_at_goal / arm_retreated_after_flip / the by-handle metric read
        # it through env.flip_by_handle: a tumble that lands upright (measured
        # 2026-08-28: upright_at_goal 28/episode at iter 5 from yanked mugs)
        # must not collect the arrival annuity -- the task is a HANDLE flip.
        self.flipped_held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.flip_by_handle = self.flipped_held

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best[env_ids] = -2.0
        self.flipped_held[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_improvement: float = 0.05,
        max_speed: float = float("inf"),
        contact_threshold: float = 0.5,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        up = _up_cos(env, object_cfg).nan_to_num(nan=-2.0, posinf=-2.0, neginf=-2.0)
        gate = (handle_held(env, threshold=contact_threshold) > 0.5) & _object_calm(env, max_speed, object_cfg)
        unseeded = self.best <= -2.0
        self.best[unseeded] = up[unseeded]
        improved = gate & (up > self.best + min_improvement)
        self.best[improved] = up[improved]
        self.flipped_held |= (handle_held(env, threshold=contact_threshold) > 0.5) & (up > UPRIGHT_MIN_COS)
        return improved.float()


class lift_progress(ManagerTermBase):
    """Lift ratchet: 1.0 per ``min_improvement`` [m] of NEW episode-best mug
    height above its rest, credited only while the handle is held, up to
    ``max_lift`` above rest.

    Bridges the flip's dead zone (run zbe5j091, 2026-08-28): the held policy
    tilted the mug against the table for one up-cos ratchet step (~20 deg)
    and plateaued, because a 180-degree flip needs the bar ~11 cm up (the far
    rim sweeps ~10 cm) and nothing paid for lifting. Same shape as
    upright_progress: episode-best only, so bobbing re-earns nothing.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best = torch.full((env.num_envs,), -2.0, device=env.device)
        self.seed_z = torch.full((env.num_envs,), -2.0, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best[env_ids] = -2.0
        self.seed_z[env_ids] = -2.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_improvement: float = 0.01,
        max_lift: float = 0.12,
        contact_threshold: float = 0.01,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        z = (obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).nan_to_num(nan=-2.0)
        gate = handle_held(env, threshold=contact_threshold) > 0.5
        unseeded = self.best <= -2.0
        self.best[unseeded] = z[unseeded]
        self.seed_z[unseeded] = z[unseeded]
        improved = gate & (z > self.best + min_improvement) & (z <= self.seed_z + max_lift)
        self.best[improved] = z[improved]
        return improved.float()


class flip_fsm(ManagerTermBase):
    """The hang's staged economy (hang_fsm_core) on the flip's predicates.

    Stages: 0 APPROACH -> 1 PINCHED (handle held) -> 2 LIFTED (held, root
    lift_height above its inverted rest) -> 3 ROTATED (mug crossed upright
    WHILE HELD; a tumble never counts) -> 4 PLACED (upright on the table, calm,
    released) -> success when the arm parks at ``pose`` (ready pose).
    Publishes outputs on ``env._fsm`` (fsm_milestones / fsm_shaping / metrics
    are the hang's, imported) and the by-handle latch on ``env.flip_by_handle``.
    Runs in the termination manager (before rewards).

    Why (run wt7f4r33, 2026-08-28): per-step pinch income made "hold on the
    table" a local optimum -- handle_pinch 36/episode, lift <1 cm, no flip.
    Here holding pays once (milestone + reach ratchet) and only progress pays.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.fsm = HangFsm(env.num_envs, env.device, ratchet_w=cfg.params.get("ratchet_w"), carry_requires_lifted=True)
        self.flipped_held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._z0 = torch.full((env.num_envs,), float("nan"), device=env.device)
        env.flip_by_handle = self.flipped_held

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fsm.reset(env_ids)
        self.flipped_held[env_ids] = False
        self._z0[env_ids] = float("nan")
        if hasattr(self, "_hold_count"):
            self._hold_count[env_ids] = 0
        if hasattr(self, "_err0"):
            self._err0[env_ids] = float("nan")

    def __call__(
        self,
        env,
        pose: dict[str, float],
        joint_tol: float,
        sensor_name: str = "pad_object_contact",
        contact_threshold: float = 0.01,
        lift_height: float = 0.06,
        z_max: float = 0.06,
        max_speed: float = 0.1,
        reach_std: float = 0.2,
        rotate_min_cos: float = 0.7,
        rest_z: float | None = None,
        ratchet_w: tuple | None = None,
        success_mode: str = "placed",
        hold_frames: int = 30,
        wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["follower_left_link_6"]),
    ) -> torch.Tensor:
        """``rotate_min_cos`` (2026-08-28, probe_flip_rotate_sweep): from an 11 cm
        held lift, sweeping forearm roll x wrist pitch x wrist roll reaches held
        up_cos > 0.87 in 1/180 cells, > 0.7 in 6, > 0.5 in 11, and the pinch is
        lost at the end of every top cell -- ROTATED means 'past 45 degrees from
        upright while held'; PLACED still demands upright (0.87) on the table."""
        obj = env.scene["object"]
        robot = env.scene["robot"]
        p = obj.data.root_pos_w.torch - env.scene.env_origins
        up = _up_cos(env, SceneEntityCfg("object")).nan_to_num(nan=-1.0)
        held = handle_held(env, threshold=contact_threshold) > 0.5
        pad_f = _sensor_force_mag(env, sensor_name)
        released = pad_f < contact_threshold
        # per-episode rest height of the root (inverted spawn), seeded on the first frame
        unseeded = torch.isnan(self._z0)
        # rest_z (2026-08-28): with a LIFTED bank the first frame is not the rest, so the
        # lift reference is the spawn's rest height when given, not the episode's first z.
        self._z0 = torch.where(unseeded, torch.full_like(p[:, 2], rest_z) if rest_z is not None else p[:, 2], self._z0)
        lifted = p[:, 2] > self._z0 + lift_height
        # CARRY validity uses a lower bar (hysteresis): the root of a rotating mug
        # dips as it turns (the base swings below the handle), which must not
        # regress the stage mid-rotation.
        lifted_hold = p[:, 2] > self._z0 + 0.4 * lift_height
        upright = up > UPRIGHT_MIN_COS
        self.flipped_held |= (up > rotate_min_cos) & held  # rotated past the threshold while held: a HANDLE flip
        rotated = (up > rotate_min_cos) & self.flipped_held
        calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
        placed = rotated & (p[:, 2] < z_max) & calm
        ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
        target = torch.tensor([pose[n] for n in pose], device=env.device)
        arm_err = torch.abs(robot.data.joint_pos.torch[:, ids] - target).max(dim=1).values
        arm_ok = arm_err < joint_tol
        # bounded progress scalars
        d_tip = _tip_handle_distance(env, SceneEntityCfg("object"), SceneEntityCfg("ee_frame"), wrist_cfg)
        reach_prog = 1.0 - torch.tanh(d_tip / reach_std)
        lift_prog = ((p[:, 2] - self._z0) / lift_height).clamp(0.0, 1.0)
        rotate_prog = ((up + 1.0) / 2.0).clamp(0.0, 1.0)
        gid, _ = robot.find_joints(["follower_left_left_carriage_joint"], preserve_order=True)
        open_frac = (robot.data.joint_pos.torch[:, gid[0]] / 0.044).clamp(0.0, 1.0)
        release_prog = 0.5 * open_frac + 0.5 * (self.fsm.persist.float() / 12.0).clamp(0.0, 1.0)
        if not hasattr(self, "_err0"):
            self._err0 = torch.full((env.num_envs,), float("nan"), device=env.device)
        just_placed = (self.fsm.stage == 4) & torch.isnan(self._err0)
        self._err0 = torch.where(just_placed, arm_err.clamp(min=1e-3), self._err0)
        self._err0 = torch.where(self.fsm.stage < 4, torch.full_like(self._err0, float("nan")), self._err0)
        retreat_prog = torch.where(
            torch.isnan(self._err0), torch.zeros_like(arm_err), (1.0 - arm_err / self._err0).clamp(0.0, 1.0)
        )
        out = self.fsm.step(FsmInputs(
            held=held, lifted=lifted, threaded=rotated, supported=placed, released=released, arm_ok=arm_ok,
            reach_prog=reach_prog, lift_prog=lift_prog, insert_prog=rotate_prog,
            release_prog=release_prog, retreat_prog=retreat_prog, lifted_hold=lifted_hold,
        ))
        out["regress_total"] = self.fsm.regressions
        if success_mode == "in_hand":
            # Success = flipped BY THE HANDLE and held upright in hand for hold_frames
            # (2026-08-28): setting the mug down upright from a single pinch is
            # kinematically out of reach on this rig (fingers point ~40 deg up when
            # the mug is upright; no IK reaches that at table height; drops land
            # upright 0-2%), so PLACED/retreat cannot pay. Without a terminal success
            # the learned policy flips the mug in 0.7 s, then rolls it back and
            # forth and drops it (probe_flip_policy_rollout, model_1550).
            if not hasattr(self, "_hold_count"):
                self._hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            done_pred = (out["stage"] >= 3) & rotated & held
            self._hold_count = torch.where(done_pred, self._hold_count + 1, torch.zeros_like(self._hold_count))
            out["success"] = self._hold_count >= hold_frames
        env._fsm = out
        return out["success"]


def _flip_by_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Latch from upright_progress: True once the mug crossed upright while held this episode."""
    latch = getattr(env, "flip_by_handle", None)
    if latch is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return latch


def flip_by_handle_metric(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Metric at 1e-3 scale (weight 1.0 in cfg): fraction of steps with the by-handle latch set."""
    return _flip_by_handle(env).float() * 1e-3


def upright_at_goal(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    z_max: float = 0.06,
    max_speed: float = float("inf"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The flip's arrival annuity: pays the planar-arrival kernel only while
    the mug is STRICTLY upright, on the table, and calm.

    The strict upright gate is the task; the planar kernel puts the righted
    mug back on its tape-measured spot instead of wherever the flip threw it.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    distance = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
    z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    gate = (
        (_up_cos(env, object_cfg) > UPRIGHT_MIN_COS)
        & (z_local < z_max)
        & _object_calm(env, max_speed, object_cfg)
        & _flip_by_handle(env)
    )
    return _finite(gate * (1.0 - torch.tanh(distance / std)))


def arm_retreated_after_flip(
    env: ManagerBasedRLEnv,
    std: float,
    vel_std: float,
    command_name: str,
    z_max: float = 0.06,
    max_speed: float = float("inf"),
    contact_sensor: str = "pad_object_contact",
    contact_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
) -> torch.Tensor:
    """Post-flip retreat: the arrival annuity's own gates times arm stillness
    times a NO-CONTACT gate on the pads.

    Income exists only on top of a completed, settled flip, so it cannot fund
    hovering short of one; the no-contact factor makes "let go of the mug"
    itself the paid behavior — a policy that keeps clutching the righted mug
    collects arrival but not retreat.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    distance = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
    z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    gate = (
        (_up_cos(env, object_cfg) > UPRIGHT_MIN_COS)
        & (z_local < z_max)
        & _object_calm(env, max_speed, object_cfg)
        & (_sensor_force_mag(env, contact_sensor) < contact_threshold)
        & _flip_by_handle(env)
    )
    speed = torch.linalg.vector_norm(robot.data.joint_vel.torch[:, arm_cfg.joint_ids], dim=1)
    return _finite(gate * (1.0 - torch.tanh(distance / std)) * (1.0 - torch.tanh(speed / vel_std)))


def object_orientation_in_world(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object root quaternion (x, y, z, w), world frame — the arm base is
    env-aligned so world doubles as the base frame. The flip policy cannot
    act on an orientation it cannot see."""
    return _finite(env.scene[object_cfg.name].data.root_quat_w.torch)
