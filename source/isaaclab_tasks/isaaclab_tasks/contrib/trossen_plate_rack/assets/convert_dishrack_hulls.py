# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convex variants of TRI's wire dish rack, cut from ``dishrack_mesh.usd`` (run convert_plate_rack.py first).

The raw 41,758-face wire costs 105 s/iter at 512 envs against the raw mug mesh, so
the mug-rack scene picks a hulled variant via ``RACK_COLLISION``. Every piece is the
convex hull of TRI's OWN wire faces in a region -- nothing is authored by hand::

    dishrack_hull.usd  one hull of the whole wire   (a wedge: arches at one end -> the mug slides off)
    dishrack_bay.usd   floor-lattice slab + tray    (walls are ghosts)
    dishrack_cage.usd  floor slab + tray + 2 side panels + 2 end panels  (6 pieces; plate-slot uprights are ghosts)

Usage: python convert_dishrack_hulls.py
"""

import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

USD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
FLOOR_Z = 0.052  # rack-local height of the floor lattice (densest wire band, from a z histogram of the bay)
FLOOR_BAND = 0.012


def _read_parts(path):
    stage = Usd.Stage.Open(path)
    parts = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh = UsdGeom.Mesh(prim)
            v = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            f = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            parts[prim.GetParent().GetName().split("_")[-1]] = (v, f)
    return parts


def _write(path, visuals, hulls):
    vec = lambda a: [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in a]  # noqa: E731
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Dishrack")
    stage.SetDefaultPrim(root.GetPrim())
    for name, (v, f) in visuals.items():
        vis = UsdGeom.Mesh.Define(stage, f"/Dishrack/visuals_{name}/mesh")
        vis.GetPointsAttr().Set(vec(v))
        vis.GetFaceVertexCountsAttr().Set([3] * len(f))
        vis.GetFaceVertexIndicesAttr().Set([int(i) for t in f for i in t])
        vis.GetDisplayColorAttr().Set([Gf.Vec3f(0.75, 0.76, 0.78)])
    for name, mesh in hulls.items():
        col = UsdGeom.Mesh.Define(stage, f"/Dishrack/collisions_{name}/mesh")
        col.GetPointsAttr().Set(vec(mesh.vertices))
        col.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        col.GetFaceVertexIndicesAttr().Set([int(i) for t in mesh.faces for i in t])
        col.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).GetApproximationAttr().Set("convexHull")
    stage.GetRootLayer().Save()
    print(f"wrote {path}: " + ", ".join(f"{k} {len(m.faces)}f" for k, m in hulls.items()))


def main():
    parts = _read_parts(os.path.join(USD_DIR, "dishrack_mesh.usd"))
    wv, wf = parts["wire"]
    bv, bf = parts["base"]
    hull = lambda sel: trimesh.Trimesh(wv, wf[sel], process=True).convex_hull  # noqa: E731
    tray = trimesh.Trimesh(bv, bf, process=True).convex_hull
    tc = wv[wf].mean(1)
    fz = wv[wf][:, :, 2]
    floor = (fz.max(1) < FLOOR_Z + FLOOR_BAND) & (fz.min(1) > FLOOR_Z - FLOOR_BAND)
    above = ~floor & (tc[:, 2] > FLOOR_Z)
    bay = floor & (tc[:, 0] > 0.0) & (tc[:, 0] < 0.17)  # the mug bay is the +x half (after the rack's Rz(-90))

    _write(os.path.join(USD_DIR, "dishrack_hull.usd"), parts, {"wire": hull(slice(None)), "base": tray})
    _write(os.path.join(USD_DIR, "dishrack_bay.usd"), parts, {"bayfloor": hull(bay), "tray": tray})
    _write(
        os.path.join(USD_DIR, "dishrack_cage.usd"),
        parts,
        {
            "floor": hull(floor),
            "side_pos": hull(above & (tc[:, 1] > 0.118)),
            "side_neg": hull(above & (tc[:, 1] < -0.118)),
            "end_pos": hull(above & (tc[:, 0] > 0.165) & (np.abs(tc[:, 1]) <= 0.118)),
            "end_neg": hull(above & (tc[:, 0] < -0.170) & (np.abs(tc[:, 1]) <= 0.118)),
            "tray": tray,
        },
    )


if __name__ == "__main__":
    main()
