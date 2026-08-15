# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classic Franka-cube-lift MDP terms for the Trossen spatula task.

The four functions are the reference lift task's reach / lift / goal-track terms and
the object-position observation, implemented against the stable generic layer only
(``isaaclab.envs.mdp`` plus the ``.torch`` views of the warp-backed data buffers), so
this task does not depend on the evolving ``core.lift`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finite(t: torch.Tensor) -> torch.Tensor:
    """Zero non-finite entries.

    Rewards are computed BEFORE terminations in the manager step, so the step
    in which a world diverges would hand PPO a non-finite reward even though
    the speed/non-finite termination resets that world in the same step. A
    diverging world therefore contributes a finite zero for its single dying
    step; the event itself stays measured via the termination counters.
    """
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NaN-safe action-rate penalty (see :func:`_finite`)."""
    return _finite(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1))


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """NaN-safe joint-velocity penalty (see :func:`_finite`)."""
    asset = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1))


def _sensor_force_mag(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Total filtered contact-force magnitude per env for a shape-filtered sensor.

    Order-invariant reduction over sensor bodies and filtered shapes (the
    resolved shape order is not stable across builds), NaN-safe like every
    reward input in this task.
    """
    forces = env.scene.sensors[sensor_name].data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    net = forces.torch.sum(dim=2)
    mag = torch.linalg.vector_norm(net, dim=-1)
    return mag.reshape(env.num_envs, -1).sum(dim=-1).nan_to_num(0.0)


def handle_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while the finger pads press the mug's HANDLE pieces."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def body_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while any arm link presses the mug's BODY (wall/base) pieces."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def pad_contact_bipolar(
    env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1, off_scale: float = 0.025
) -> torch.Tensor:
    """+1 while the pads press the object, ``-off_scale`` while they do not.

    The asymmetry is deliberate: under strict terminations a symmetric
    penalty is a suicide pump — bleeding -w for a full episode costs more
    than the -10 terminal price of batting the mug away, so the policy ends
    the bleed by killing the object. ``off_scale`` is sized so a full episode
    of never-touching (150 steps) still costs LESS than one failure
    termination.
    """
    c = (_sensor_force_mag(env, sensor_name) > threshold).float()
    return _finite(c - off_scale * (1.0 - c))


def object_off_table(
    env: ManagerBasedRLEnv,
    x_bound: float,
    y_bound: float,
    z_bound: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose object left the workspace (env-local xy footprint or height).

    xy bounds catch objects knocked sideways onto the floor. The height bound
    catches near-vertical contact ejections, which keep the object inside the
    xy footprint for the whole flight and would otherwise coast until timeout
    while feeding the physics error norm ever-larger coordinates.
    """
    obj = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w.torch[:, :3] - env.scene.env_origins[:, :3]
    return (pos_local[:, 0].abs() > x_bound) | (pos_local[:, 1].abs() > y_bound) | (pos_local[:, 2] > z_bound)


def object_speed_exceeds(
    env: ManagerBasedRLEnv,
    max_linear_speed: float,
    max_angular_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate envs whose object exceeds any physically reachable speed.

    MJWarp does not consume the authored body velocity clamps (Newton drops
    ``velocity_limit_sim`` when constructing the solver model), so the required
    speed bound is enforced here, per the Newton asset-migration guide: a
    diverging contact event is terminated at the first control step it exceeds
    speeds no legitimate trajectory reaches, before the state can escalate to
    non-finite. Non-finite velocities terminate unconditionally (NaN compares
    false, so the isfinite clause cannot be folded into the thresholds).
    """
    obj = env.scene[object_cfg.name]
    lin = torch.norm(obj.data.root_lin_vel_w.torch, dim=1)
    ang = torch.norm(obj.data.root_ang_vel_w.torch, dim=1)
    finite = torch.isfinite(lin) & torch.isfinite(ang)
    return (lin > max_linear_speed) | (ang > max_angular_speed) | ~finite


def robot_state_abnormal(
    env: ManagerBasedRLEnv,
    max_joint_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate envs whose ROBOT state is non-finite or implausibly fast.

    The object-side speed valve cannot see an arm-side blowup: a constraint
    explosion through a grasped/hooked object drives the low-inertia wrist
    joints non-finite, and joint state feeds the policy observations directly.
    Same containment contract as :func:`object_speed_exceeds`, robot side.
    """
    asset = env.scene[asset_cfg.name]
    qpos = asset.data.joint_pos.torch
    qvel = asset.data.joint_vel.torch
    finite = torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1)
    return (qvel.abs().amax(dim=1) > max_joint_vel) | ~finite


def physics_diverged(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate envs whose world latched the adaptive solver's divergence flag.

    A latched world (NaN/divergent state, or an inner solve still failing at
    the dt floor under SAP containment) held its last committed FINITE state
    while its clock skipped to the boundary: the solver refused to commit a
    step it could not converge. Every state-space valve above therefore sees a
    plausible frozen scene and stays silent, while actions no longer influence
    the state -- the episode is broken by construction. The manager preserves
    the transient solver latch in a per-env mask that survives until read
    here; terminating hands the env to the reset machinery, which clears the
    mask for the reset envs. All-False when no adaptive Newton solver is
    active (nothing can latch).
    """
    try:
        from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
    except ImportError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mask = NewtonMJWarpManager.get_diverged_env_mask()
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask[: env.num_envs] != 0


# Handle center in the mug body frame (SDF handle_middle frame).
_HANDLE_OFFSET_B = (0.062, 0.0, 0.058)


def handle_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """World position of the mug's handle center."""
    obj = env.scene[object_cfg.name]
    offset = torch.tensor(_HANDLE_OFFSET_B, device=env.device).expand(env.num_envs, 3)
    return obj.data.root_pos_w.torch + quat_apply(obj.data.root_quat_w.torch, offset)


def handle_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    ee_body_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_name: str | None = None,
) -> torch.Tensor:
    """Reach shaping toward the HANDLE, alignment-weighted.

    Distance alone is a point condition any wrist drape can satisfy, so the
    shaping is multiplied by how well the gripper's approach axis (the EE
    link's +x, through the fingers) points at the handle: parked poses with
    the TCP near the handle but the fingers aimed elsewhere earn nothing."""
    robot = env.scene[robot_cfg.name]
    idx = robot.body_names.index(ee_body_name)
    link_pos = robot.data.body_pos_w.torch[:, idx]
    link_quat = robot.data.body_quat_w.torch[:, idx]
    axis = quat_apply(link_quat, torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(env.num_envs, 3))
    handle = handle_pos_w(env, object_cfg)
    to_handle = handle - link_pos
    align = torch.clamp(
        (axis * to_handle).sum(dim=1) / to_handle.norm(dim=1).clamp_min(1e-6), min=0.0
    )
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(handle - ee_pos_w, dim=1)
    shaped = (1.0 - torch.tanh(distance / std)) * align
    if contact_sensor_name is not None:
        # Contact counts as distance zero: while the pads press the handle the
        # term pays its maximum, so attempting and holding contact strictly
        # dominates hovering at the shaping optimum — without this, the dense
        # reach income competes against the sparse touch it should lead into
        # (measured: touch rate collapsed once the hover optimum was found).
        touching = handle_grasped(env, contact_sensor_name)
        shaped = torch.maximum(shaped, touching)
    return _finite(shaped)


def handle_grasped(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 0.1) -> torch.Tensor:
    """1.0 while the finger pads carry force on the HANDLE pieces — the strict
    by-the-handle possession predicate (proximity alone does not qualify)."""
    return _finite((_sensor_force_mag(env, sensor_name) > threshold).float())


def lift_progress_by_handle(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    sensor_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense lift progress, payable only while the handle is gripped.

    The lift indicator is a step at ``target_height``: the first centimeters of
    a lift pay nothing, so no gradient leads a holding policy upward. This term
    pays the PATH — height above rest, normalized to the target — under the
    same handle-grip gate as the lift income itself.
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w.torch[:, 2]
    progress = torch.clamp((z - rest_height) / (target_height - rest_height), min=0.0, max=1.0)
    return _finite(progress * handle_grasped(env, sensor_name))


def object_lifted_by_handle(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    sensor_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Lift income payable only while the handle is grasped."""
    return object_is_lifted(env, minimal_height, object_cfg) * handle_grasped(env, sensor_name)


def object_goal_distance_by_handle(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    sensor_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal income payable only while the handle is grasped."""
    return object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg) * handle_grasped(
        env, sensor_name
    )


class dropped_after_lift(ManagerTermBase):
    """Terminate envs where a LIFTED object has come back down.

    A per-env latch records that the object cleared ``lift_height`` at any
    point in the episode; once latched, the object returning below
    ``drop_height`` ends the episode. Losing a lift is terminal, while never
    lifting is judged by the other terms — so the gate never punishes
    pre-grasp exploration.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._was_lifted[:] = False
        else:
            self._was_lifted[env_ids] = False
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height: float,
        drop_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        z = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2]
        finite = torch.isfinite(z)
        self._was_lifted |= finite & (z > lift_height)
        return self._was_lifted & finite & (z < drop_height)


def object_tipped(
    env: ManagerBasedRLEnv,
    min_up_cos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object has tipped past the spill threshold.

    The mug's body frame is Z-up; the tilt is the angle between its body +Z
    and world +Z, read as the z-component of the rotated up axis:
    ``1 - 2(x^2 + y^2)``. The Newton data API returns quaternions in
    (x, y, z, w) order (Warp's layout), so x, y are indices 0, 1 — indexing
    this wxyz-style false-fired on the mug's authored 90-degree yaw and
    insta-terminated every episode. ``min_up_cos`` 0.5 = leaning past 60
    degrees — knocked onto its side or carried spilling, either way the
    episode is over. Fires only on finite state so a diverged world is judged
    by the containment valves, not here.
    """
    quat = env.scene[object_cfg.name].data.root_quat_w.torch
    up_z = 1.0 - 2.0 * (quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1])
    return torch.isfinite(up_z) & (up_z < min_up_cos)


def object_knocked_from_spot(
    env: ManagerBasedRLEnv,
    xy_tol: float,
    z_max: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object sits at table height OUTSIDE its spawn box.

    A mug that is at table height (env-local z below ``z_max``) but displaced
    more than ``xy_tol`` from its authored spawn was batted, dragged, or
    tipped-and-rolled there — a clean carry leaves the table vertically from
    the spawn spot. Complements ``object_tipped``: this fires on knocked-AWAY
    even when the mug stays upright. Finite-only, like every valve.
    """
    obj = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w.torch[:, :3] - env.scene.env_origins[:, :3]
    disp = pos_local[:, :2] - obj.data.default_root_pose.torch[:, :2]
    finite = torch.isfinite(pos_local).all(dim=-1)
    return finite & (pos_local[:, 2] < z_max) & (disp.abs().amax(dim=-1) > xy_tol)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """1.0 when the object root is above ``minimal_height`` [m] (world z)."""
    obj = env.scene[object_cfg.name]
    return _finite(torch.where(obj.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0))


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reach shaping ``1 - tanh(|object - ee| / std)`` using the ee_frame sensor's first target."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    return _finite(1.0 - torch.tanh(distance / std))


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal-tracking shaping, gated on the object being lifted."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
    return _finite((obj.data.root_pos_w.torch[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std)))


def object_held(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1.0 when the object rides within ``threshold`` [m] of the TCP.

    Possession predicate for a parallel-jaw gripper: a grasped object moves
    with the TCP, a batted one separates within a step, so gating income on
    this indicator makes ballistic strategies unpayable.
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w.torch - ee_pos_w, dim=1)
    return torch.where(distance < threshold, 1.0, 0.0)


def object_lifted_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Lift indicator payable only while the object is possessed."""
    return object_is_lifted(env, minimal_height, object_cfg) * object_held(env, threshold, object_cfg, ee_frame_cfg)


def object_goal_distance_held(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Goal-tracking income payable only while the object is possessed."""
    return object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg) * object_held(
        env, threshold, object_cfg, ee_frame_cfg
    )


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root position expressed in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return object_pos_b
