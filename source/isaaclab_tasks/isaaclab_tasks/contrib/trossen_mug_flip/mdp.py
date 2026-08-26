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
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.contrib.trossen_mug_slide.mdp import *  # noqa: F401,F403
from isaaclab_tasks.contrib.trossen_mug_slide.mdp import _finite, _object_calm, _sensor_force_mag
from isaaclab_tasks.contrib.trossen_mug_lift.mdp import (  # noqa: F401
    SUCCESS_POS_THRESHOLD,
    SUCCESS_TILT_THRESHOLD,
    ObjectPoseSuccessCommand,
    validate_object_spawn,
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

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best[env_ids] = -2.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_improvement: float = 0.05,
        max_speed: float = float("inf"),
        max_ang_speed: float = 6.0,
        contact_threshold: float = 0.5,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        up = _up_cos(env, object_cfg).nan_to_num(nan=-2.0, posinf=-2.0, neginf=-2.0)
        obj = env.scene[object_cfg.name]
        ang_ok = torch.linalg.vector_norm(obj.data.root_ang_vel_w.torch, dim=1) < max_ang_speed
        gate = (
            (handle_held(env, threshold=contact_threshold) > 0.5)
            & _object_calm(env, max_speed, object_cfg)
            & ang_ok
        )
        unseeded = self.best <= -2.0
        self.best[unseeded] = up[unseeded]
        improved = gate & (up > self.best + min_improvement)
        self.best[improved] = up[improved]
        return improved.float()



class flip_hold_and_retreat(ManagerTermBase):
    """The flip's completion economy in one stateful term: a dwell-ramped
    upright-at-goal annuity, a one-time completion bonus when the righted
    mug has HELD for ``dwell_steps``, and a retreat income (arm stillness x
    pads-off) that exists ONLY after that milestone latches — letting go
    before the flip is banked pays nothing.

    Logged through the pose command's metrics: ``dwell`` and ``completed``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._dwell = torch.zeros(n, device=dev)
        self._completed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._dwell[env_ids] = 0.0
        self._completed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        vel_std: float,
        command_name: str,
        z_max: float = 0.06,
        max_speed: float = float("inf"),
        dwell_steps: int = 30,
        contact_sensor: str = "pad_object_contact",
        contact_threshold: float = 0.5,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
    ) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        obj = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        distance = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w.torch[:, :2], dim=1)
        z_local = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        upright_at_spot = (
            (_up_cos(env, object_cfg) > UPRIGHT_MIN_COS)
            & (z_local < z_max)
            & _object_calm(env, max_speed, object_cfg)
        )
        self._dwell = torch.where(upright_at_spot, self._dwell + 1.0, torch.zeros_like(self._dwell))
        just_completed = ~self._completed & (self._dwell >= dwell_steps)
        self._completed |= just_completed

        kernel = 1.0 - torch.tanh(distance / std)
        hold_income = upright_at_spot.float() * (self._dwell / dwell_steps).clamp(max=1.0) * kernel * 2.0
        speed = torch.linalg.vector_norm(robot.data.joint_vel.torch[:, arm_cfg.joint_ids], dim=1)
        pads_off = _sensor_force_mag(env, contact_sensor) < contact_threshold
        retreat = (
            self._completed.float()
            * upright_at_spot.float()
            * pads_off.float()
            * (1.0 - torch.tanh(speed / vel_std))
        )

        cmd_term = env.command_manager.get_term(command_name)
        cmd_term.metrics["dwell"] = (self._dwell / dwell_steps).clamp(max=1.0)
        cmd_term.metrics["completed"] = self._completed.float()
        return _finite(hold_income + retreat + just_completed.float() * 20.0)


def object_orientation_in_world(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object root quaternion (x, y, z, w), world frame — the arm base is
    env-aligned so world doubles as the base frame. The flip policy cannot
    act on an orientation it cannot see."""
    return _finite(env.scene[object_cfg.name].data.root_quat_w.torch)
