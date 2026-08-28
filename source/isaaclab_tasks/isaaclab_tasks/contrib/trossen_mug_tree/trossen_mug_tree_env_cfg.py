# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI MUG HANG: pick the mug off the table and hang it on the
TRI mug tree (dorhors_wood_mug_holder), by the rim, so its handle hooks a branch.

TRI's ``bimanual_hang_mugs_on_mug_holder_from_table`` scenario, on the single-arm
rig. Everything transfers from the mug lift; the deltas are the tree in the scene
and the GOAL POSE the mug has to match: TRI's own hung-mug rest pose on a branch,
composed here from TRI's SDF branch frame and TRI's scenario weld.

EVERY PLACEMENT CONSTANT BELOW IS USER-SET. The values are starting points: TRI's
where TRI authored one (the mug-on-branch weld), labeled placeholders where TRI's
station-relative ranges do not transfer to this rig (the tree's spot on the table).
Author them in the pose lab (``pose_lab.py --task IsaacContrib-MugHang-Trossen-v0``;
``--object-at-goal`` drops the mug at GOAL_POSE so the hang can be judged).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab_tasks.contrib.trossen_mug_lift.bedrock import apply_reverse_curriculum
from isaaclab_tasks.contrib.trossen_mug_lift.trossen_mug_lift_env_cfg import (
    _SPAWN_X,
    _SPAWN_Y,
    BANK_POSE_XY_JACOBIAN,
    GRASP_BANK_POSE,
    TerminationsCfg,
    TrossenMugLiftEnvCfg,
    TrossenMugLiftSceneCfg,
)

from . import mdp
from .assets import BRANCH_BASE_FRAMES, MUG_TREE_USD_PATH, TRI_FRAMES

# ============================================================================ USER-SET
# Tree base on the table (env frame, meters). PLACEHOLDER: TRI welds the holder
# at table-relative ranges that map to x 0.28-0.48 here -- beyond this arm's
# reach -- so only TRI's DIRECTION from the mug spot (toward +X, slightly toward
# the arm) is kept; the radius is unvalidated. Set it in the lab.
TREE_POS = (0.20, -0.2, 0.02)
# Tree yaw (degrees). TRI welds the holder at identity in a frame whose +X faces
# the robots; that is env +Y here, hence +90: the bottom/top branch pairs point
# toward and away from the arm, the mid pair sideways.
TREE_YAW_DEG = 90.0
# Which branch the mug hangs on. TRI's scenarios hang mug_0 on bottom_right /
# top_right / mid_front / mid_back (its robot-facing set); bottom_right is the
# lowest of those.
GOAL_BRANCH = "bottom_right_branch_base"
# ============================================================================

# TRI's hung-mug pose: mug origin relative to a ``*_branch_base`` frame, from
# riverway_drying_rack_multitask_scenarios.yaml (MugOnMugHolder_0 / WeldedMugs_1).
TRI_MUG_ON_BRANCH_POS = (0.09434705, 0.01579505, 0.00668992)
TRI_MUG_ON_BRANCH_RPY_DEG = (-73.14290783, 6.61464528, 133.98682999)

assert GOAL_BRANCH in BRANCH_BASE_FRAMES, f"{GOAL_BRANCH!r} not in TRI's frames {BRANCH_BASE_FRAMES}"


# ------------------------------------------------------------ frame arithmetic
def _rot_rpy(r: float, p: float, y: float) -> list[list[float]]:
    """SDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat_vec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _compose(R1, t1, R2, t2):
    """(R1, t1) o (R2, t2)."""
    return _mat_mul(R1, R2), [a + b for a, b in zip(t1, _mat_vec(R1, t2))]


def _rpy_from_rot(R) -> tuple[float, float, float]:
    """Inverse of :func:`_rot_rpy` (the pose command's own convention)."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2][0])))
    roll = math.atan2(R[2][1], R[2][2])
    yaw = math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def _quat_xyzw_from_rot(R) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + R[0][0] + R[1][1] + R[2][2])) / 2.0
    if w > 1e-6:
        return ((R[2][1] - R[1][2]) / (4 * w), (R[0][2] - R[2][0]) / (4 * w), (R[1][0] - R[0][1]) / (4 * w), w)
    # 180-degree case: largest diagonal element.
    i = max(range(3), key=lambda k: R[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + R[i][i] - R[j][j] - R[k][k])) * 2.0
    q = [0.0, 0.0, 0.0, (R[k][j] - R[j][k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j][i] + R[i][j]) / s
    q[k] = (R[k][i] + R[i][k]) / s
    return tuple(q)


def hung_mug_pose_env(
    tree_pos=TREE_POS, tree_yaw_deg=TREE_YAW_DEG, branch=GOAL_BRANCH
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Mug origin pose in the env frame when hung on ``branch`` per TRI's weld.

    Returns ``(pos, rpy)``; ``rpy`` in the pose command's roll-pitch-yaw.
    """
    R_tree = _rot_rpy(0.0, 0.0, math.radians(tree_yaw_deg))
    b_pos, b_rpy = TRI_FRAMES[branch]
    R_tb, t_tb = _compose(R_tree, list(tree_pos), _rot_rpy(*b_rpy), list(b_pos))
    R_bm = _rot_rpy(*(math.radians(a) for a in TRI_MUG_ON_BRANCH_RPY_DEG))
    R_m, t_m = _compose(R_tb, t_tb, R_bm, list(TRI_MUG_ON_BRANCH_POS))
    return tuple(t_m), _rpy_from_rot(R_m)


# AUTHORED in the pose lab 2026-08-25 (mug released onto bottom_right, settled,
# PRINT): the pose the task rewards. hung_mug_pose_env() remains the TRI-weld
# seed for re-authoring after any tree move.
GOAL_POSE_ENV = ((0.1965, -0.1639, 0.0879), (-0.2131, -0.6705, -2.7193))  # 2026-08-28: branch through the COLLISION loop (x 0.048, z 0.059 body)
# USER-SET: the arm's FINISHED pose -- the ready posture swung 1.2 rad away
# from the tree (tree at +X; TCP settles at env (-0.39, +0.18, 0.42), FK-probed
# 2026-08-25). Re-author in the lab (SAVE) any time.
ARM_FINISH_POSE = {
    "follower_left_joint_0": -1.2,
    "follower_left_joint_1": 1.570796,
    "follower_left_joint_2": 1.570796,
    "follower_left_joint_3": 0.0,
    "follower_left_joint_4": 0.0,
    "follower_left_joint_5": 0.0,
}

_R_GOAL = _mat_mul(
    _rot_rpy(0.0, 0.0, math.radians(TREE_YAW_DEG)),
    _mat_mul(_rot_rpy(*TRI_FRAMES[GOAL_BRANCH][1]), _rot_rpy(*(math.radians(a) for a in TRI_MUG_ON_BRANCH_RPY_DEG))),
)
GOAL_QUAT_XYZW = _quat_xyzw_from_rot(_R_GOAL)


# The goal branch's axis in the ENV frame, for the threading gate: base point
# and unit direction, composed from TREE_POS/TREE_YAW and TRI's branch frame.
_R_TREE = _rot_rpy(0.0, 0.0, math.radians(TREE_YAW_DEG))
_BP, _BRPY = TRI_FRAMES[GOAL_BRANCH]
GOAL_BRANCH_BASE_ENV = tuple(a + b for a, b in zip(TREE_POS, _mat_vec(_R_TREE, list(_BP))))
GOAL_BRANCH_AXIS_ENV = tuple(_mat_vec(_mat_mul(_R_TREE, _rot_rpy(*_BRPY)), [0.0, 0.0, 1.0]))
_GATE_PARAMS = {"branch_base_env": GOAL_BRANCH_BASE_ENV, "branch_axis_env": GOAL_BRANCH_AXIS_ENV}


def _yaw_quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    h = math.radians(yaw_deg) / 2.0
    return (0.0, 0.0, math.sin(h), math.cos(h))


# ---------------------------------------------------------------------- scene
@configclass
class MugTreeSceneCfg(TrossenMugLiftSceneCfg):
    """The rig scene plus the TRI mug tree; the mug at its tape spot (inherited)."""

    tree: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MugTree",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TREE_POS, rot=_yaw_quat_xyzw(TREE_YAW_DEG)),
        spawn=sim_utils.UsdFileCfg(usd_path=MUG_TREE_USD_PATH),
    )

    # Mug pieces touching the tree (TRI's ``bodies_in_contact`` for the hang).
    # Sensed on the mug BODY, filtered to the tree's collision shapes (the
    # family's sensor form: bodies sense, shapes filter).
    object_tree_contact = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_shape_prim_expr=["{ENV_REGEX_NS}/MugTree/collisions_.*/.*"],
    )


@configclass
class HangEventCfg:
    """Fixed starts: exact mug spawn at the tape spot, arm at home/ready, no bank."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # Zeroed home scatter, declared BEFORE the bank event (bedrock's ordering
    # guard): home starts stay exact; the bank overwrites its subset after.
    randomize_arm_start = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="follower_left_joint_[0-5]"),
        },
    )


@configclass
class HangTerminationsCfg(TerminationsCfg):
    """The lift's terminations plus the POSITIVE one: job done, arm parked.

    Fires only when the mug rests in the goal pose (released, calm) AND the arm
    is at ARM_FINISH_POSE -- the episode ends as a success and the completion
    bonus (rewards.completion) pays on that step."""

    task_complete = DoneTerm(
        func=mdp.hang_fsm,
        params={
            "branch_base_env": None,  # filled in __post_init__ from _GATE_PARAMS
            "branch_axis_env": None,
            "pose": ARM_FINISH_POSE,
            "joint_tol": 0.3,
            "sensor_name": "pad_object_contact",
            "tree_sensor_name": "object_tree_contact",
        },
    )


@configclass
class HangCurriculumCfg:
    """Empty: bootstrap mode from home starts."""


@configclass
class HangRewardsCfg:
    """The STAGED economy, whole and self-contained (2026-08-26 refactor): no
    per-step proximity or contact income anywhere. One-shot milestones + strict
    PBRS shaping (hang_fsm_core) + one success bonus + the family's action tax
    and blowup fine. Max pre-completion return 35 + 10 < success 100 -- the
    inequality the reward-economy tests assert."""

    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)
    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-50, params={"term_keys": ["robot_abnormal"]})
    milestones = RewTerm(func=mdp.fsm_milestones, params={}, weight=1.0)
    shaping = RewTerm(func=mdp.fsm_shaping, params={}, weight=1.0)
    success_bonus = RewTerm(func=mdp.is_terminated_term, weight=100.0, params={"term_keys": ["task_complete"]})
    # diagnostics: values pre-scaled by 1e-3 (mdp._METRIC_SCALE), weight 1.0 -- x1000 in W&B.
    # (weight 1e-9 underflowed float32 and read as 0.0000.)
    metric_stage = RewTerm(func=mdp.fsm_metric_stage, params={}, weight=1.0)
    metric_regressions = RewTerm(func=mdp.fsm_metric_regressions, params={}, weight=1.0)
    metric_ms_grasp = RewTerm(func=mdp.fsm_metric_ms_grasp, params={}, weight=1.0)
    metric_ms_lift = RewTerm(func=mdp.fsm_metric_ms_lift, params={}, weight=1.0)
    metric_ms_insert = RewTerm(func=mdp.fsm_metric_ms_insert, params={}, weight=1.0)
    metric_ms_release = RewTerm(func=mdp.fsm_metric_ms_release, params={}, weight=1.0)


@configclass
class TrossenMugHangEnvCfg(TrossenMugLiftEnvCfg):
    # 2000, not the family's 8192: one fixed spawn, one fixed goal pose, no
    # placement DR -- there is no generalization axis to feed with envs.
    scene: MugTreeSceneCfg = MugTreeSceneCfg(num_envs=2000, env_spacing=2.5)
    rewards: HangRewardsCfg = HangRewardsCfg()
    events: HangEventCfg = HangEventCfg()
    terminations: HangTerminationsCfg = HangTerminationsCfg()
    curriculum: HangCurriculumCfg = HangCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # FIVE uniform subdivisions of the family's 1/90 boundary at the SAME
        # 30 Hz control (ordered 2026-08-25): dt 1/450 (2.22 ms), decimation 15.
        self.sim.dt = 1 / 450
        self.decimation = 15
        self.sim.render_interval = self.decimation
        # THE GOAL POSE: the mug hung on GOAL_BRANCH per TRI's weld, as the
        # command (position AND orientation). The marker draws it in the viewer;
        # the lab's --object-at-goal drops the mug there.
        (gx, gy, gz), (gr, gp, gyaw) = GOAL_POSE_ENV
        rg = self.commands.object_pose.ranges
        rg.pos_x, rg.pos_y, rg.pos_z = (gx, gx), (gy, gy), (gz, gz)
        rg.roll, rg.pitch, rg.yaw = (gr, gr), (gp, gp), (gyaw, gyaw)
        # Rewards are HangRewardsCfg, whole: nothing inherited, nothing appended.
        self.terminations.task_complete.params.update(_GATE_PARAMS)
        # BANK STARTS (2026-08-28): with the goal now physically reachable, the
        # bottleneck measured at iter 1000 is discovery -- <1% of home-start
        # episodes get past the grasp. The mug's spawn, rig and pinch are the
        # lift's exactly, so the lift's VALIDATED pre-grasp (teleop-authored,
        # passed the dynamic ladder) banks here unchanged; Florensa anneal back
        # toward home over the first 100 iterations (2400 env-steps).
        apply_reverse_curriculum(
            self,
            bank_pose=GRASP_BANK_POSE,
            bank_xy_jacobian=BANK_POSE_XY_JACOBIAN,
            nominal_object_xy=(_SPAWN_X, _SPAWN_Y),
            bank_fraction=0.5,
            end_step=2_400,
        )
        # (Rewards inherited from the lift for the scaffold: rim reach, grasp-gated
        # position ratchet, held-at-goal success. The hang's own success shaping is
        # the next change, once the goal pose is authored.)


@configclass
class TrossenMugHangEnvCfg_PLAY(TrossenMugHangEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False


def ledger() -> str:
    """Every authored constant with its source, for TROSSEN_SCENES.md / the console."""
    (gx, gy, gz), (gr, gp, gyaw) = GOAL_POSE_ENV
    lines = [
        "MugHang (IsaacContrib-MugHang-Trossen-v0) -- trossen_mug_tree/trossen_mug_tree_env_cfg.py",
        f"  TREE_POS      = {TREE_POS}   [USER-SET placeholder: TRI direction, radius unvalidated]",
        f"  TREE_YAW_DEG  = {TREE_YAW_DEG}   [TRI relative orientation to the robot]",
        f"  GOAL_BRANCH   = {GOAL_BRANCH}   [USER-SET; TRI's set: bottom_right, top_right, mid_front, mid_back]",
        f"  mug spawn     = ({_SPAWN_X:.3f}, {_SPAWN_Y:.3f}, 0.021)   [inherited tape spot, trossen_mug_lift]",
        f"  GOAL_POSE_ENV = pos ({gx:.4f}, {gy:.4f}, {gz:.4f}) rpy ({gr:.4f}, {gp:.4f}, {gyaw:.4f}) rad   [computed: TRI weld on {GOAL_BRANCH}]",
        f"  GOAL_QUAT     = {tuple(round(v, 4) for v in GOAL_QUAT_XYZW)} (xyzw)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(ledger())
