# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the mug hang: the mug lift's set, verbatim; hang-only terms land here."""

from isaaclab_tasks.contrib.trossen_mug_lift.mdp import *  # noqa: F401,F403
from isaaclab_tasks.contrib.trossen_mug_lift.mdp import _pad_force_mags  # noqa: F401 -- underscore names skip the star import


import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv

# Mug handle opening in the BODY frame, measured on TRI's COLLISION tets
# (2026-08-28; the earlier 0.043..0.0645 came from the VISUAL glTF and was
# 5.5 mm too far out -- the authored goal put the branch OUTSIDE the loop and
# every hang probe failed on geometry, not physics). 1 mm inset each side.
_HANDLE_X = (0.0385, 0.0580)
_HANDLE_Z = (0.031, 0.087)


def branch_through_handle(
    env,
    branch_base_env: tuple[float, float, float],
    branch_axis_env: tuple[float, float, float],
    s_range: tuple[float, float] = (-0.005, 0.105),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool per env: the goal branch's AXIS passes through the mug's handle
    opening -- the topological fact of hanging, which no pose kernel can fake.

    The branch is static in the env frame; its axis is expressed in the mug
    body frame and intersected with the handle plane (y = 0). The crossing
    must land inside the measured loop opening AND within the branch's own
    length, so leaning against the trunk or hovering at the right pose with
    the loop clear earns nothing.
    """
    obj = env.scene[object_cfg.name]
    p = obj.data.root_pos_w.torch - env.scene.env_origins
    q = obj.data.root_quat_w.torch
    qi = quat_inv(q)
    b0 = torch.tensor(branch_base_env, device=p.device, dtype=p.dtype).expand_as(p)
    ax = torch.tensor(branch_axis_env, device=p.device, dtype=p.dtype).expand_as(p)
    p0 = quat_apply(qi, b0 - p)
    d = quat_apply(qi, ax)
    dy = torch.where(d[:, 1].abs() < 1e-6, torch.full_like(d[:, 1], 1e-6), d[:, 1])
    s = -p0[:, 1] / dy
    c = p0 + s.unsqueeze(-1) * d
    inside = (
        (c[:, 0] > _HANDLE_X[0]) & (c[:, 0] < _HANDLE_X[1])
        & (c[:, 2] > _HANDLE_Z[0]) & (c[:, 2] < _HANDLE_Z[1])
        & (s > s_range[0]) & (s < s_range[1])
        & (d[:, 1].abs() > 1e-6)
    )
    return inside


def hung_object_pose_match(env, pos_std, command_name, branch_base_env, branch_axis_env):
    """object_pose_match, paid ONLY while the branch threads the handle."""
    gate = branch_through_handle(env, branch_base_env, branch_axis_env)
    return gate.float() * object_pose_match(env, pos_std, command_name)


def hung_success_at_goal(env, command_name, pos_std, sensor_name, branch_base_env, branch_axis_env, contact_threshold=0.01):
    """The held-at-goal bridge, gated on the threading."""
    gate = branch_through_handle(env, branch_base_env, branch_axis_env)
    return gate.float() * success_at_goal(env, command_name, pos_std, sensor_name, contact_threshold)


def hung_arm_finish(env, pose, std, pos_std, command_name, sensor_name, branch_base_env, branch_axis_env, contact_threshold=0.01, max_speed=0.1):
    """arm_finish_pose_match, gated on the threading."""
    gate = branch_through_handle(env, branch_base_env, branch_axis_env)
    return gate.float() * arm_finish_pose_match(
        env, pose, std, pos_std, command_name, sensor_name, contact_threshold=contact_threshold, max_speed=max_speed
    )


def hung_finished_at_pose(
    env, branch_base_env, branch_axis_env, pose, joint_tol, pos_tol, min_up_cos, command_name, sensor_name,
    contact_threshold=0.01, max_speed=0.1,
):
    """The positive termination, gated on the threading."""
    gate = branch_through_handle(env, branch_base_env, branch_axis_env)
    return gate & finished_at_pose(
        env, pose, joint_tol, pos_tol, min_up_cos, command_name, sensor_name,
        contact_threshold=contact_threshold, max_speed=max_speed,
    )


# Handle-loop center in the mug body frame: midpoint of the measured opening
# (x 0.043..0.0645, z 0.025..0.092), on the handle plane.
_LOOP_CENTER_B = (0.0483, 0.0, 0.0590)


def loop_to_branch(
    env,
    std: float,
    branch_base_env: tuple[float, float, float],
    branch_axis_env: tuple[float, float, float],
    sensor_name: str,
    contact_threshold: float = 0.01,
    s_range: tuple[float, float] = (-0.005, 0.105),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The gradient INTO the threading gate: 1 - tanh(d/std) on the distance
    from the goal branch's axis segment to the handle-loop center, paid only
    while the mug is held. This is the maneuver itself -- bring the loop mouth
    to the branch -- shaped continuously where the gate is binary."""
    obj = env.scene[object_cfg.name]
    p = obj.data.root_pos_w.torch - env.scene.env_origins
    q = obj.data.root_quat_w.torch
    loop_w = p + quat_apply(q, torch.tensor(_LOOP_CENTER_B, device=p.device, dtype=p.dtype).expand_as(p))
    b0 = torch.tensor(branch_base_env, device=p.device, dtype=p.dtype).expand_as(p)
    ax = torch.tensor(branch_axis_env, device=p.device, dtype=p.dtype).expand_as(p)
    t = ((loop_w - b0) * ax).sum(dim=-1).clamp(s_range[0], s_range[1])
    closest = b0 + t.unsqueeze(-1) * ax
    d = torch.linalg.vector_norm(loop_w - closest, dim=-1)
    held = opposed_grasp(env, sensor_name, contact_threshold)
    # PRESENTATION factor: the branch threads along the loop's NORMAL (body
    # +Y), so proximity only pays times |axis . y_body| -- approaching the
    # branch face-on earns, hovering beside it does not (proximity alone was
    # farmed: high loop_to_branch with zero threading, run rx4ebi94).
    y_w = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=p.device, dtype=p.dtype).expand_as(p))
    face_on = (y_w * ax).sum(dim=-1).abs()
    return held.float() * (1.0 - torch.tanh(d / std)) * face_on


def carry_orientation(
    env,
    command_name: str,
    sensor_name: str,
    contact_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense up-axis alignment with the COMMANDED pose during the carry, held
    only: makes wrist rotation profitable on the way, not just at the goal."""
    from isaaclab.utils.math import combine_frame_transforms  # noqa: PLC0415

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, des_quat_w = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3], command[:, 3:7]
    )
    up = torch.tensor([0.0, 0.0, 1.0], device=obj.data.root_pos_w.torch.device).expand(env.num_envs, 3)
    cos = (quat_apply(obj.data.root_quat_w.torch, up) * quat_apply(des_quat_w, up)).sum(dim=-1)
    held = opposed_grasp(env, sensor_name, contact_threshold)
    return held.float() * (1.0 + cos) / 2.0


# ============================================================== latched state machine
from isaaclab.managers import ManagerTermBase


def _hang_state(env, key):
    st = getattr(env, "_hang_state", None)
    if st is None or key not in st:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return st[key]


class hang_state_machine(ManagerTermBase):
    """The task's phase logic, latched per env, doubling as the termination.

    Runs in the TERMINATION manager (which computes before rewards each step,
    manager_based_rl_env.py:250 vs :263), so every reward term reads this
    step's state from ``env._hang_state``.

    Phases: carry -> (supported for ``support_frames``) -> PLACED latch ->
    release + retreat -> (placed & released & supported & arm at finish for
    ``finish_frames`` consecutive frames) -> episode ends as a success.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        z = lambda dt: torch.zeros(env.num_envs, dtype=dt, device=env.device)  # noqa: E731
        self.support_count = z(torch.long)
        self.finish_count = z(torch.long)
        self.placed = z(torch.bool)
        self.was_released = z(torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.support_count[env_ids] = 0
        self.finish_count[env_ids] = 0
        self.placed[env_ids] = False
        self.was_released[env_ids] = False

    def __call__(
        self,
        env,
        pose: dict[str, float],
        joint_tol: float,
        branch_base_env: tuple[float, float, float],
        branch_axis_env: tuple[float, float, float],
        sensor_name: str,
        tree_sensor_name: str,
        contact_threshold: float = 0.01,
        support_force: float = 0.05,
        support_frames: int = 12,
        finish_frames: int = 6,
        max_speed: float = 0.1,
    ) -> torch.Tensor:
        obj = env.scene["object"]
        robot = env.scene["robot"]
        threaded = branch_through_handle(env, branch_base_env, branch_axis_env)
        tree_f = env.scene[tree_sensor_name].data.net_forces_w
        tree_touch = torch.linalg.vector_norm(tree_f.torch if hasattr(tree_f, "torch") else tree_f, dim=-1).max(dim=1).values > support_force
        calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
        supported = threaded & tree_touch & calm
        self.support_count = torch.where(supported, self.support_count + 1, torch.zeros_like(self.support_count))
        self.placed |= self.support_count >= support_frames

        pad_f = _pad_force_mags(env, sensor_name)
        released = pad_f.max(dim=1).values < contact_threshold
        self.was_released |= self.placed & released
        recontact = self.was_released & ~released

        ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
        target = torch.tensor([pose[n] for n in pose], device=env.device)
        arm_ok = torch.abs(robot.data.joint_pos.torch[:, ids] - target).max(dim=1).values < joint_tol

        finish_ok = self.placed & released & supported & arm_ok
        self.finish_count = torch.where(finish_ok, self.finish_count + 1, torch.zeros_like(self.finish_count))

        env._hang_state = {
            "placed": self.placed, "released": released, "supported": supported,
            "arm_ok": arm_ok, "recontact": recontact,
        }
        return self.finish_count >= finish_frames


# ------------------------------------------------ pre-place gated carry rewards
def fingers_to_object_pp(env, std, sensor_name, contact_threshold, asset_cfg):
    """fingers_to_object, zeroed once PLACED: reaching for a placed mug is regression."""
    return (~_hang_state(env, "placed")).float() * fingers_to_object(env, std, sensor_name, contact_threshold, asset_cfg)


def mug_grasped_pp(env, sensor_name, threshold):
    return (~_hang_state(env, "placed")).float() * mug_grasped(env, sensor_name, threshold)


def pad_contact_count_pp(env, sensor_name, threshold):
    return (~_hang_state(env, "placed")).float() * pad_contact_count(env, sensor_name, threshold)


def loop_to_branch_pp(env, std, branch_base_env, branch_axis_env, sensor_name, contact_threshold=0.01):
    return (~_hang_state(env, "placed")).float() * loop_to_branch(env, std, branch_base_env, branch_axis_env, sensor_name, contact_threshold)


def carry_orientation_pp(env, command_name, sensor_name, contact_threshold=0.01):
    return (~_hang_state(env, "placed")).float() * carry_orientation(env, command_name, sensor_name, contact_threshold)


def hung_success_at_goal_pp(env, command_name, pos_std, sensor_name, branch_base_env, branch_axis_env, contact_threshold=0.01):
    return (~_hang_state(env, "placed")).float() * hung_success_at_goal(env, command_name, pos_std, sensor_name, branch_base_env, branch_axis_env, contact_threshold)


# ------------------------------------------------ post-place income and penalty
def release_and_retreat(env, pose, std):
    """PLACED only: gripper opening plus arm progress toward the finish pose."""
    placed = _hang_state(env, "placed").float()
    robot = env.scene["robot"]
    gid, _ = robot.find_joints(["follower_left_left_carriage_joint"], preserve_order=True)
    open_frac = (robot.data.joint_pos.torch[:, gid[0]] / 0.044).clamp(0.0, 1.0)
    ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
    target = torch.tensor([pose[n] for n in pose], device=env.device)
    err = torch.linalg.vector_norm(robot.data.joint_pos.torch[:, ids] - target, dim=1)
    return placed * (0.5 * open_frac + (1.0 - torch.tanh(err / std)))


def recontact_penalty(env):
    """PLACED-and-once-released only: pads back on the mug costs."""
    return _hang_state(env, "recontact").float()


# ------------------------------------------------ metrics (weight 1e-9: log-only)
def metric_placed(env):
    return _hang_state(env, "placed").float()


def metric_released(env):
    return (_hang_state(env, "placed") & _hang_state(env, "released")).float()


def metric_arm_at_finish(env):
    return _hang_state(env, "arm_ok").float()


def metric_recontact(env):
    return _hang_state(env, "recontact").float()


def early_release_penalty(
    env,
    sensor_name: str,
    contact_threshold: float = 0.01,
    z_air: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dropping the mug BEFORE the place latch costs: pre-place, mug airborne
    (above table height), and no pad on it. Prices the observed failure --
    release at 4.0-4.7 s with the loop unseated, mug falling at the tree base."""
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    airborne = z > z_air
    released = _pad_force_mags(env, sensor_name).max(dim=1).values < contact_threshold
    return ((~_hang_state(env, "placed")) & airborne & released).float()


# ============================================================ staged FSM (2026-08-26)
from .hang_fsm_core import FsmInputs, HangFsm


class hang_fsm(ManagerTermBase):
    """Sim glue for :mod:`hang_fsm_core`: computes the physical predicates and
    bounded progress scalars, delegates staging/milestones/PBRS to the core,
    publishes the outputs on ``env._fsm``, and serves as the success
    termination. Runs in the termination manager (before rewards)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.fsm = HangFsm(env.num_envs, env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fsm.reset(env_ids)

    def __call__(
        self,
        env,
        pose: dict[str, float],
        joint_tol: float,
        branch_base_env: tuple[float, float, float],
        branch_axis_env: tuple[float, float, float],
        sensor_name: str,
        tree_sensor_name: str,
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
        threaded = branch_through_handle(env, branch_base_env, branch_axis_env)
        tf = env.scene[tree_sensor_name].data.net_forces_w
        tree_touch = torch.linalg.vector_norm(tf.torch if hasattr(tf, "torch") else tf, dim=-1).max(dim=1).values > support_force
        calm = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < max_speed
        supported = threaded & tree_touch & calm
        ids, _ = robot.find_joints(list(pose.keys()), preserve_order=True)
        target = torch.tensor([pose[n] for n in pose], device=env.device)
        arm_err = torch.abs(robot.data.joint_pos.torch[:, ids] - target).max(dim=1).values
        arm_ok = arm_err < joint_tol

        # bounded progress scalars
        pad_ids, _ = robot.find_bodies("follower_left_gripper_.*")
        pad_pos = robot.data.body_pos_w.torch[:, pad_ids]
        d_reach = torch.linalg.vector_norm(pad_pos - obj.data.root_pos_w.torch[:, None, :], dim=-1).max(dim=-1).values
        reach_prog = 1.0 - torch.tanh(d_reach / 0.4)
        lift_prog = ((p[:, 2] - 0.021) / (lift_z - 0.021)).clamp(0.0, 1.0)
        loop_w = p + quat_apply(q, torch.tensor(_LOOP_CENTER_B, device=p.device, dtype=p.dtype).expand_as(p))
        b0 = torch.tensor(branch_base_env, device=p.device, dtype=p.dtype).expand_as(p)
        ax = torch.tensor(branch_axis_env, device=p.device, dtype=p.dtype).expand_as(p)
        t = ((loop_w - b0) * ax).sum(dim=-1).clamp(-0.005, 0.105)
        d_loop = torch.linalg.vector_norm(loop_w - (b0 + t.unsqueeze(-1) * ax), dim=-1)
        y_w = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=p.device, dtype=p.dtype).expand_as(p))
        insert_prog = (1.0 - torch.tanh(d_loop / 0.1)) * (y_w * ax).sum(dim=-1).abs()
        gid, _ = robot.find_joints(["follower_left_left_carriage_joint"], preserve_order=True)
        open_frac = (robot.data.joint_pos.torch[:, gid[0]] / 0.044).clamp(0.0, 1.0)
        release_prog = 0.5 * open_frac + 0.5 * (self.fsm.persist.float() / 12.0).clamp(0.0, 1.0)
        # sigma 3.0 (was 1.0, 2026-08-28): at the placed pose the arm is ~1.5 rad
        # from the finish pose and tanh(1.5) left retreat_prog at 0.09 -- no slope
        # to climb, so the policy parked at PLACED forever (86% placed, 0%
        # complete, arm never at finish). At 3.0 the placed pose reads 0.45 and
        # the finish 1.0: half a unit of ratchet to earn by swinging away.
        retreat_prog = 1.0 - torch.tanh(arm_err / 3.0)

        out = self.fsm.step(FsmInputs(
            held=held, lifted=lifted, threaded=threaded, supported=supported, released=released, arm_ok=arm_ok,
            reach_prog=reach_prog, lift_prog=lift_prog, insert_prog=insert_prog,
            release_prog=release_prog, retreat_prog=retreat_prog,
        ))
        out["regress_total"] = self.fsm.regressions
        env._fsm = out
        return out["success"]


_METRIC_SCALE = 1e-3  # metrics carry 1e-3 x the true value at weight 1.0 (float32-safe; W&B x1000)


def _fsm_out(env, key, dim0=None):
    st = getattr(env, "_fsm", None)
    if st is None or key not in st:
        return torch.zeros(env.num_envs, device=env.device)
    v = st[key]
    return v.float() if v.dtype != torch.float32 else v


def fsm_milestones(env):
    """One-shot milestone bonuses, paid on the advancement step only."""
    return _fsm_out(env, "milestone_reward")


def fsm_shaping(env):
    """Strict PBRS within the active stage: gamma*Phi(s') - Phi(s)."""
    return _fsm_out(env, "shaping")


def fsm_metric_stage(env):
    return _METRIC_SCALE * _fsm_out(env, "stage")


def fsm_metric_regressions(env):
    return _METRIC_SCALE * _fsm_out(env, "regressed")


def fsm_metric_ms_grasp(env):
    return _METRIC_SCALE * (_fsm_out(env, "new_milestones")[:, 0] if getattr(env, "_fsm", None) else torch.zeros(env.num_envs, device=env.device))


def fsm_metric_ms_lift(env):
    return _METRIC_SCALE * (_fsm_out(env, "new_milestones")[:, 1] if getattr(env, "_fsm", None) else torch.zeros(env.num_envs, device=env.device))


def fsm_metric_ms_insert(env):
    return _METRIC_SCALE * (_fsm_out(env, "new_milestones")[:, 2] if getattr(env, "_fsm", None) else torch.zeros(env.num_envs, device=env.device))


def fsm_metric_ms_release(env):
    return _METRIC_SCALE * (_fsm_out(env, "new_milestones")[:, 3] if getattr(env, "_fsm", None) else torch.zeros(env.num_envs, device=env.device))


def fsm_metric_ms_retreat(env):
    return _METRIC_SCALE * (_fsm_out(env, "new_milestones")[:, 4] if getattr(env, "_fsm", None) else torch.zeros(env.num_envs, device=env.device))
