# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TRI dorhors_wood_mug_holder (the mug tree): USD path and the SDF's named frames.

The frames are parsed from TRI's own SDF (``lbm_src/``) at import so the task
cfg composes poses against TRI's frame definitions, never against a copy.
"""

import os
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
LBM_SRC_DIR = os.path.join(_HERE, "lbm_src")
MUG_TREE_SDF_PATH = os.path.join(LBM_SRC_DIR, "dorhors_wood_mug_holder.sdf")
MUG_TREE_GLTF_PATH = os.path.join(LBM_SRC_DIR, "dorhors_wood_mug_holder.gltf")
MUG_TREE_USD_PATH = os.path.join(_HERE, "usd", "mug_tree.usd")


def read_sdf_frames(sdf_path: str = MUG_TREE_SDF_PATH) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Named frames of a single-link TRI SDF: name -> ((x, y, z), (roll, pitch, yaw)) in the body frame.

    Every frame in the holder SDF is posed relative to the link or to ``origin``
    (itself the link), so no chaining is needed; a frame relative to anything else
    raises rather than silently mis-posing.
    """
    root = ET.parse(sdf_path).getroot()
    model = root.find("model")
    link_name = model.find("link").get("name")
    frames: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for fr in model.findall("frame"):
        pose = fr.find("pose")
        rel = pose.get("relative_to", link_name)
        if rel not in (link_name, "origin"):
            raise ValueError(f"frame {fr.get('name')!r} is relative to {rel!r}; only body-relative frames are supported")
        v = [float(t) for t in pose.text.split()]
        frames[fr.get("name")] = (tuple(v[:3]), tuple(v[3:]))
    return frames


def read_sdf_cylinders(sdf_path: str = MUG_TREE_SDF_PATH) -> dict[str, tuple[tuple[float, ...], tuple[float, ...], float, float]]:
    """TRI's <collision> cylinders: name -> ((x, y, z), (roll, pitch, yaw), radius, length), body frame."""
    root = ET.parse(sdf_path).getroot()
    link = root.find("model").find("link")
    out = {}
    for col in link.findall("collision"):
        cyl = col.find("geometry").find("cylinder")
        if cyl is None:
            continue
        v = [float(t) for t in col.find("pose").text.split()]
        out[col.get("name")] = (tuple(v[:3]), tuple(v[3:]), float(cyl.find("radius").text), float(cyl.find("length").text))
    return out


TRI_FRAMES = read_sdf_frames()
TRI_COLLISION_CYLINDERS = read_sdf_cylinders()
BRANCH_BASE_FRAMES = tuple(n for n in TRI_FRAMES if n.endswith("_branch_base"))
