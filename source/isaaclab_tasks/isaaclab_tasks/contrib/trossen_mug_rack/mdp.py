# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the mug-rack place: the hang's staged FSM with the rack's own
"placed" predicate -- INVERTED (up-axis within 10 degrees of -Z, TRI's
put_mugs_on_plates tolerance) and RESTING ON THE LATTICE (rack contact,
inside the measured stable band), instead of the branch-through-loop gate."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.trossen_mug_lift.mdp import *  # noqa: F401,F403
from isaaclab_tasks.contrib.trossen_mug_lift.mdp import _pad_force_mags  # noqa: F401
from isaaclab_tasks.contrib.trossen_mug_tree.hang_fsm_core import FsmInputs, HangFsm
from isaaclab_tasks.contrib.trossen_mug_tree.mdp import (  # noqa: F401 -- reuse the hang's reward/metric readers
    _fsm_out,
    fsm_metric_ms_grasp,
    fsm_metric_ms_insert,
    fsm_metric_ms_lift,
    fsm_metric_ms_release,
    fsm_metric_ms_retreat,
    fsm_metric_regressions,
    fsm_metric_stage,
    fsm_milestones,
    fsm_shaping,
)


def inverted_in_bay(
    env,
    bay_center_env: tuple[float, float, float],
    bay_half_xy: float,
    z_range: tuple[float, float],
    min_down_cos: float = 0.985,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool per env: mug up-axis within acos(min_down_cos) of -Z AND its root inside
    the measured stable rest band of the lattice (xy box, z band). The rack place's
    'threaded'."""
    obj = env.scene[object_cfg.name]
    p = obj.data.root_pos_w.torch - env.scene.env_origins
    q = obj.data.root_quat_w.torch
    up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=p.device, dtype=p.dtype).expand_as(p))
    c = torch.tensor(bay_center_env, device=p.device, dtype=p.dtype)
    in_xy = ((p[:, :2] - c[:2]).abs() < bay_half_xy).all(dim=1)
    in_z = (p[:, 2] > z_range[0]) & (p[:, 2] < z_range[1])
    return (up[:, 2] < -min_down_cos) & in_xy & in_z


class rack_fsm(ManagerTermBase):
    """The hang's staged FSM re-targeted: 'inserted' = inverted over/in the bay,
    'supported' = inverted-in-bay + rack contact + calm. Same milestones, ratchets,
    rest-neutral shaping and success bonus (hang_fsm_core)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.fsm = HangFsm(env.num_envs, env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fsm.reset(env_ids)
        if hasattr(self, "_err0"):
            self._err0[env_ids] = float("nan")

    def __call__(
        self,
        env,
        pose: dict[str, float],
        joint_tol: float,
        bay_center_env: tuple[float, float, float],
        bay_half_xy: float,
        z_range: tuple[float, float],
        sensor_name: str,
        rack_sensor_name: str,
        contact_threshold: float = 0.01,
        support_force: float = 0.05,
        lift_z: float = 0.10,
        max_speed: float = 0.1,
    ) -> torch.Tensor:
        obj = env.scene["object"]
        robot = env.scene["robot"]
        p = obj.data.root_pos_w.torch - env.scene.env_origins
        q = obj.data.root_quat_w.torch
        pad_f = _pad_force_mags(env, sensor_name)
        held = (pad_f > contact_threshold).all(dim=1)
        released = pad_f.max(dim=1).values < contact_threshold
        lifted = p[:, 2] > lift_z
        inserted = inverted_in_bay(env, bay_center_env, bay_half_xy, z_range)
        rf = env.scene[rack_sensor_name].data.net_forces_w
        rack_touch = torch.linalg.vector_norm(rf.torch if hasattr(rf, "torch") else rf, dim=-1).max(dim=1).values > support_force
        calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
        supported = inserted & rack_touch & calm
        ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
        target = torch.tensor([pose[n] for n in pose], device=env.device)
        arm_err = torch.abs(robot.data.joint_pos.torch[:, ids] - target).max(dim=1).values
        arm_ok = arm_err < joint_tol
        # progress scalars
        pad_ids, _ = robot.find_bodies("follower_left_gripper_.*")
        d_reach = torch.linalg.vector_norm(robot.data.body_pos_w.torch[:, pad_ids] - obj.data.root_pos_w.torch[:, None, :], dim=-1).max(dim=-1).values
        reach_prog = 1.0 - torch.tanh(d_reach / 0.4)
        lift_prog = ((p[:, 2] - 0.021) / (lift_z - 0.021)).clamp(0.0, 1.0)
        up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=p.device, dtype=p.dtype).expand_as(p))
        c = torch.tensor(bay_center_env, device=p.device, dtype=p.dtype)
        d_bay = torch.linalg.vector_norm(p - c, dim=-1)
        insert_prog = (1.0 - torch.tanh(d_bay / 0.15)) * ((1.0 - up[:, 2]) / 2.0)  # closeness x inversion
        gid, _ = robot.find_joints(["follower_left_left_carriage_joint"], preserve_order=True)
        open_frac = (robot.data.joint_pos.torch[:, gid[0]] / 0.044).clamp(0.0, 1.0)
        release_prog = 0.5 * open_frac + 0.5 * (self.fsm.persist.float() / 12.0).clamp(0.0, 1.0)
        if not hasattr(self, "_err0"):
            self._err0 = torch.full((env.num_envs,), float("nan"), device=env.device)
        just_placed = (self.fsm.stage == 4) & torch.isnan(self._err0)
        self._err0 = torch.where(just_placed, arm_err.clamp(min=1e-3), self._err0)
        self._err0 = torch.where(self.fsm.stage < 4, torch.full_like(self._err0, float("nan")), self._err0)
        retreat_prog = torch.where(torch.isnan(self._err0), torch.zeros_like(arm_err), (1.0 - arm_err / self._err0).clamp(0.0, 1.0))
        out = self.fsm.step(FsmInputs(
            held=held, lifted=lifted, threaded=inserted, supported=supported, released=released, arm_ok=arm_ok,
            reach_prog=reach_prog, lift_prog=lift_prog, insert_prog=insert_prog, release_prog=release_prog, retreat_prog=retreat_prog,
        ))
        out["regress_total"] = self.fsm.regressions
        env._fsm = out
        return out["success"]
