# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the plate tasks: the mug lift's set, verbatim.

The plate tasks are deliberately the same MDP on different geometry — the
rim-circle reach kernel, the grasp-gated ratchet, the success gate and the
containment terminations all read the scene through entity names and live
poses, none of them mug-specific. Anything plate-only added later lives here.
"""

from isaaclab_tasks.contrib.trossen_mug_lift.mdp import *  # noqa: F401,F403
