# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trossen Stationary AI rig articulation for the mug-lift task.

Fully self-contained: the rig USD (from Trossen's official
https://github.com/TrossenRobotics/trossen_ai_isaac, zero external references) and its
no-rails override are vendored under ``assets/usd/`` -- no clone, no env setup. The
no-rails variant is the default: the rig's rail frame is a collision body a lift policy
exploits by jamming the object against it instead of grasping. ``TROSSEN_RAILS=1``
selects the full rig as a contact-rich ablation.

Actuator gains are the vendor's, verbatim from Trossen's official MuJoCo model
(``trossen_arm_mujoco`` ``stationary_ai.xml``): per-joint kp/kv (stiff shoulders,
soft wrists), vendor armature and joint friction, effort caps at the vendor
forcerange. Both carriage joints of each gripper are actuated explicitly; the USD
``physxMimicJoint`` on the right carriages is PhysX-specific and not honored by
other backends.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

_USD_DIR = os.path.join(os.path.dirname(__file__), "assets", "usd")

STATIONARY_AI_USD = os.path.join(_USD_DIR, "stationary_ai.usd")
STATIONARY_AI_NORAILS_USD = os.path.join(_USD_DIR, "stationary_ai_norails.usda")
# Task overrides: zero the physxMimicJoint gearing on the passive right carriages (the
# mimic is PhysX-specific; Newton maps it loosely — both carriages are explicitly
# actuated instead, so the gripper is identical under every backend) and DEACTIVATE
# the entire follower_right arm: the task is single-arm, and the idle second arm
# only adds bodies, contacts, and observation width.
STATIONARY_AI_TASK_USD = os.path.join(_USD_DIR, "stationary_ai_task.usda")
STATIONARY_AI_TASK_RAILS_USD = os.path.join(_USD_DIR, "stationary_ai_task_rails.usda")


def rig_usd_path() -> str:
    """The rig USD to load: no-rails default, full rig via ``TROSSEN_RAILS=1``."""
    if os.environ.get("TROSSEN_RAILS") == "1":
        return STATIONARY_AI_TASK_RAILS_USD
    return STATIONARY_AI_TASK_USD


STATIONARY_AI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=rig_usd_path(),
        activate_contact_sensors=True,
        # The rig USD authors no friction on any collider, so every link falls
        # back to the shared shape cfg. State the rubber pads' coefficient here
        # instead, matching what the vendor-curated MuJoCo model of this arm
        # authors on its colliders (mujoco_menagerie trossen_wxai,
        # friction="1 5e-3 5e-4"); MuJoCo takes the element-wise maximum of a
        # pair's coefficients at equal priority, so this governs the grasp
        # rather than being averaged away against the mug's ceramic value.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # Torsional and rolling friction rows on every contact this rig makes.
        # At the default condim=3 a pinched object is free to spin about the
        # fingertip contacts and works its way out of the grasp, no matter how
        # high the sliding coefficient is. The vendor model scopes condim=6 to
        # its gripper class; applied rig-wide here because the spawn addresses
        # the articulation as a whole, which is a superset -- the arm links
        # carry no other contacts once self-collisions are off. PhysX ignores
        # the mjc: namespace, so this is inert on that backend.
        #
        # solref is the contact RESPONSE TIME, and it is authored rather than
        # left to the ke/kd conversion: that conversion turns this scene's
        # contact stiffness into a time constant far below what a step of
        # sim.dt can represent, which a fixed step absorbs silently and an
        # error-controlled step pays for by subdividing without bound. 0.02 s
        # is MuJoCo's own default and equals the 2*dt floor at this timestep.
        # Only the mjc: namespace is touched, so the SAP backend still reads
        # the authored ke.
        collision_props=MujocoCollisionCfg(condim=6, solref=(0.02, 1.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collisions OFF: under MuJoCo contacts every robot mesh is a convex
            # hull, and the two finger assemblies' hulls overlap the grasp channel, so
            # a free-air close stalls at a finger gap wider than the blade -- the
            # fingers can never reach it. Finger-object and object-table contact are
            # robot-external pairs and unaffected. Cost: the arm no longer collides
            # with the rig tabletop (same articulation); restore via fitted pad
            # collision boxes if that ever matters.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The Trossen CONTROLLER's home position, verbatim from the official
        # demo scripts (trossen_arm demos move.py: home_positions = zeros with
        # joints 1 and 2 at pi/2) — the pose the driver brings the arm to from
        # sleep before any operation. Carriages start OPEN at 0.044 per the
        # vendor's own sim episode convention (trossen_arm_mujoco
        # START_ARM_POSE); the driver's home zeroes the gripper, but episodes
        # on the rig begin with the gripper opened for the task.
        joint_pos={
            "follower_left_joint_0": 0.0,
            "follower_left_joint_1": 1.5707963267948966,
            "follower_left_joint_2": 1.5707963267948966,
            "follower_left_joint_[3-5]": 0.0,
            "follower_left_left_carriage_joint": 0.044,
            "follower_left_right_carriage_joint": 0.044,
        },
    ),
    actuators={
        # Vendor gains, verbatim from Trossen's official MuJoCo model of this
        # rig (trossen_arm_mujoco stationary_ai.xml): per-joint kp/kv with
        # stiff shoulders and soft wrists, vendor armature and joint friction,
        # effort caps equal to the vendor forcerange.
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_joint_[0-2]"],
            stiffness=200.0,
            damping=10.0,
            armature=0.032,
            friction=0.1,
            effort_limit_sim=27.0,
            # Vendor velocity limit (URDF, authored in the rig USD only as
            # PhysX-namespaced attrs Newton never reads: 193.37 deg/s).
            # Without it the arm explores 20-30 rad/s whips no hardware can
            # reach, and the adaptive march pays for integrating them.
            velocity_limit_sim=3.375,
        ),
        "left_wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_joint_3"],
            stiffness=100.0,
            damping=5.0,
            armature=0.0018,
            friction=0.1,
            effort_limit_sim=7.0,
            # Vendor 401.07 deg/s (see shoulder note).
            velocity_limit_sim=7.0,
        ),
        "left_wrist_distal": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_joint_[4-5]"],
            stiffness=50.0,
            damping=5.0,
            armature=0.0018,
            friction=0.1,
            effort_limit_sim=7.0,
            velocity_limit_sim=7.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
            stiffness=1000.0,
            damping=50.0,
            armature=0.1,
            friction=0.1,
            effort_limit_sim=400.0,
            # Vendor carriage limit (USD: 0.0875 m/s, PhysX-namespaced).
            velocity_limit_sim=0.0875,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
