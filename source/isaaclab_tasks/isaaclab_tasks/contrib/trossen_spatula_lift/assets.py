# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI rig articulation for the spatula-lift task.

Fully self-contained: the rig USD (from Trossen's official
https://github.com/TrossenRobotics/trossen_ai_isaac, zero external references) and its
no-rails override are vendored under ``assets/usd/`` -- no clone, no env setup. The
no-rails variant is the default: the rig's rail frame is a collision body a lift policy
exploits by jamming the object against it instead of grasping. ``TROSSEN_RAILS=1``
selects the full rig as a contact-rich ablation.

Actuator gains follow the Isaac Lab manipulation reference (arm stiffness=80,
damping=4, as Franka lift): the USD-baked gains are ~500x stiffer and reproduce the
policy's command chatter as visible hold jitter. Each gripper actuates only its LEFT
carriage; the right carriage is a USD ``physxMimicJoint`` mirroring it (the benign
"14 != 16 actuators" warning at load).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_USD_DIR = os.path.join(os.path.dirname(__file__), "assets", "usd")

STATIONARY_AI_USD = os.path.join(_USD_DIR, "stationary_ai.usd")
STATIONARY_AI_NORAILS_USD = os.path.join(_USD_DIR, "stationary_ai_norails.usda")


def rig_usd_path() -> str:
    """The rig USD to load: no-rails default, full rig via ``TROSSEN_RAILS=1``."""
    if os.environ.get("TROSSEN_RAILS") == "1":
        return STATIONARY_AI_USD
    return STATIONARY_AI_NORAILS_USD


STATIONARY_AI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=rig_usd_path(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "follower_left_joint_[0-5]": 0.0,
            "follower_left_left_carriage_joint": 0.0,
            "follower_right_joint_[0-5]": 0.0,
            "follower_right_left_carriage_joint": 0.0,
        },
    ),
    actuators={
        "left_arm": ImplicitActuatorCfg(joint_names_expr=["follower_left_joint_[0-5]"], stiffness=80.0, damping=4.0),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_left_carriage_joint"], stiffness=None, damping=None
        ),
        "right_arm": ImplicitActuatorCfg(joint_names_expr=["follower_right_joint_[0-5]"], stiffness=None, damping=None),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_left_carriage_joint"], stiffness=None, damping=None
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
