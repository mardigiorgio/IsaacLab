"""Grasp pose authored with pose_grasp.py.

Shaped as the ``pose`` argument of ``mdp.reset_arm_reverse_curriculum``.
"""

# Gripper when authored: 0.0440 m per carriage (100% open,
# finger gap ~136.3 mm). NOT part of the dict:
# reset_arm_reverse_curriculum leaves the carriages at their default open state
# so closing the grasp stays the policy's own first action.

GRASP_STRADDLE_POSE = {
    "follower_left_joint_0": 0.000000,
    "follower_left_joint_1": 1.570796,
    "follower_left_joint_2": 1.570796,
    "follower_left_joint_3": 0.000000,
    "follower_left_joint_4": 0.000000,
    "follower_left_joint_5": 0.000000,
}
