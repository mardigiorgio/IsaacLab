# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the plate and dishrack USDs for the Trossen plate tasks — kit-free.

Both assets are PROCEDURAL: no vendor mesh of either exists in the LBM bank,
so the geometry is authored here from dimensioned constants (a validated
source, when one appears, replaces this file's output byte-for-byte at the
same paths). Follows ``trossen_mug_lift/assets/convert_mug.py``:

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_plate_rack/assets/convert_plate_rack.py

Writes ``usd/plate.usd`` and ``usd/dishrack.usd`` next to this script.

PLATE — a revolved 4 mm ceramic shell, body frame at the bottom center, Z up.
Collision is authored as PRE-SPLIT pieces because the Newton pipeline hulls
every collision prim independently; a single-hull plate is a solid puck with
no rim to pinch. The rim carries its own finer sector band so the grip zone's
hulls stay within a chord sagitta of ~0.7 mm:

    /Plate/collisions_base/mesh          center disc
    /Plate/collisions_wall_[0-11]/mesh   30-degree slope sectors
    /Plate/collisions_rim_[0-11]/mesh    30-degree rim-band sectors

DISHRACK — a wire rack from BOX primitives only (exact primitive narrow
phase, no hulling, no decomposition): two base rails carrying N_TINE_PAIRS
vertical tine pairs at SLOT_PITCH, kinematic root. A plate seated in a slot
rests on the table between two tine pairs and exposes its top rim arc above
the tines for a parallel-jaw pinch.
"""

import os

import numpy as np
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# ------------------------------------------------------------------- plate
PLATE_RADIUS = 0.100  # [m] rim outer radius
PLATE_HEIGHT = 0.025  # [m] rim tip above the resting plane
PLATE_SHELL_T = 0.004  # [m] shell thickness
PLATE_BASE_R = 0.060  # [m] flat center disc radius
PLATE_MASS = 0.350  # [kg] ceramic salad plate
PLATE_COLOR = (0.93, 0.93, 0.91)
N_WALL_SECTORS = 12
N_RIM_SECTORS = 12
RIM_BAND_R = 0.088  # [m] faces beyond this radius belong to the rim band
N_THETA = 48  # revolve resolution

# Same contact impedance rationale as the mug (see convert_mug.py): stiff
# saturated region, 2 mm full-hardness width, priority wins pair mixing.
PLATE_SOLIMP = (0.9, 0.999, 0.002, 0.5, 2.0)
PLATE_CONTACT_PRIORITY = 1

# ---------------------------------------------------------------- dishrack
RACK_LEN_X = 0.300  # [m] rail direction
RACK_WID_Y = 0.200  # [m]
RAIL_SQ = 0.008  # [m] base rail square section
TINE_SQ = 0.006  # [m] tine square section
TINE_H = 0.100  # [m] tine height above the rail top
N_TINE_PAIRS = 7
SLOT_PITCH = 0.040  # [m] tine-pair spacing along X -> 6 slots of ~34 mm clearance
# Slot fit: the plate's standing depth is PLATE_HEIGHT (25 mm), so a 34 mm
# slot leaves ~9 mm of lean/reset play -- a 32 mm pitch left 1 mm and jammed.
TINE_PAIR_GAP_Y = 0.120  # [m] lateral gap between a pair's two tines
RACK_COLOR = (0.75, 0.76, 0.78)


def _revolve_profile() -> trimesh.Trimesh:
    """Closed shell cross-section (r, z), lofted around Z."""
    t = PLATE_SHELL_T
    outer = [
        (1e-4, 0.0),
        (PLATE_BASE_R, 0.0),
        (0.096, 0.019),
        (PLATE_RADIUS, PLATE_HEIGHT),
    ]
    inner = [
        (0.0965, PLATE_HEIGHT + 0.0018),
        (0.0925, 0.0225),
        (PLATE_BASE_R - 0.002, t),
        (1e-4, t),
    ]
    prof = np.array(outer + inner, dtype=np.float64)
    n_p = len(prof)
    verts = []
    for k in range(N_THETA):
        a = 2.0 * np.pi * k / N_THETA
        c, s = np.cos(a), np.sin(a)
        for r, z in prof:
            verts.append((r * c, r * s, z))
    faces = []
    for k in range(N_THETA):
        k2 = (k + 1) % N_THETA
        for i in range(n_p):
            i2 = (i + 1) % n_p
            a0, b0 = k * n_p + i, k * n_p + i2
            a1, b1 = k2 * n_p + i, k2 * n_p + i2
            faces.append((a0, b0, b1))
            faces.append((a0, b1, a1))
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)


def _partition_plate(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    tri_c = mesh.triangles_center
    r = np.linalg.norm(tri_c[:, :2], axis=1)
    ang = np.arctan2(tri_c[:, 1], tri_c[:, 0])
    groups: dict[str, np.ndarray] = {"collisions_base": np.flatnonzero(r <= PLATE_BASE_R)}
    wall = (r > PLATE_BASE_R) & (r <= RIM_BAND_R)
    rim = r > RIM_BAND_R
    ws = ((ang + np.pi) / (2 * np.pi) * N_WALL_SECTORS).astype(int) % N_WALL_SECTORS
    rs = ((ang + np.pi) / (2 * np.pi) * N_RIM_SECTORS).astype(int) % N_RIM_SECTORS
    for s in range(N_WALL_SECTORS):
        idx = np.flatnonzero(wall & (ws == s))
        if len(idx):
            groups[f"collisions_wall_{s}"] = idx
    for s in range(N_RIM_SECTORS):
        idx = np.flatnonzero(rim & (rs == s))
        if len(idx):
            groups[f"collisions_rim_{s}"] = idx
    return groups


def _author_mesh(stage, path, mesh: trimesh.Trimesh, collide: bool, color):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in mesh.vertices])
    prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    prim.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if collide:
        prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
        api.GetApproximationAttr().Set("convexHull")
        p = prim.GetPrim()
        p.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
            Vt.FloatArray([float(v) for v in PLATE_SOLIMP])
        )
        p.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int, custom=True).Set(PLATE_CONTACT_PRIORITY)
    return prim


def _hull_piece(mesh: trimesh.Trimesh, face_idx: np.ndarray, max_verts: int = 24) -> trimesh.Trimesh:
    raw = mesh.submesh([face_idx], append=True)
    verts = raw.vertices
    if len(verts) > max_verts:
        sel = [int(np.argmax(np.linalg.norm(verts - verts.mean(0), axis=1)))]
        d = np.linalg.norm(verts - verts[sel[0]], axis=1)
        for _ in range(max_verts - 1):
            sel.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(verts - verts[sel[-1]], axis=1))
        verts = verts[sel]
    return trimesh.convex.convex_hull(verts)


def _write_plate(out_dir: str) -> None:
    mesh = _revolve_profile()
    groups = _partition_plate(mesh)
    out_path = os.path.join(out_dir, "plate.usd")
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Plate")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.GetMassAttr().Set(PLATE_MASS)
    # Thin shallow shell: COM slightly above the base plane; disc inertia.
    com_z = 0.010
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, com_z))
    ixx = 0.25 * PLATE_MASS * PLATE_RADIUS**2
    izz = 0.5 * PLATE_MASS * PLATE_RADIUS**2
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, ixx, izz))
    _author_mesh(stage, "/Plate/visuals/visuals", mesh, collide=False, color=PLATE_COLOR)
    for name, face_idx in groups.items():
        piece = _hull_piece(mesh, face_idx)
        _author_mesh(stage, f"/Plate/{name}/mesh", piece, collide=True, color=PLATE_COLOR)
    stage.GetRootLayer().Save()
    print(f"[convert_plate_rack] wrote {out_path}: {len(groups)} collision pieces, mass {PLATE_MASS} kg")


def _author_box(stage, path, center, half, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _write_rack(out_dir: str) -> None:
    out_path = os.path.join(out_dir, "dishrack.usd")
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Dishrack")
    stage.SetDefaultPrim(root.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.GetKinematicEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).GetMassAttr().Set(1.0)

    rail_z = RAIL_SQ / 2.0
    half_span = (N_TINE_PAIRS - 1) * SLOT_PITCH / 2.0
    for side, y in (("l", TINE_PAIR_GAP_Y / 2.0), ("r", -TINE_PAIR_GAP_Y / 2.0)):
        _author_box(
            stage,
            f"/Dishrack/rail_{side}",
            (0.0, y, rail_z),
            (RACK_LEN_X / 2.0, RAIL_SQ / 2.0, RAIL_SQ / 2.0),
            RACK_COLOR,
        )
        for k in range(N_TINE_PAIRS):
            x = -half_span + k * SLOT_PITCH
            _author_box(
                stage,
                f"/Dishrack/tine_{side}_{k}",
                (x, y, RAIL_SQ + TINE_H / 2.0),
                (TINE_SQ / 2.0, TINE_SQ / 2.0, TINE_H / 2.0),
                RACK_COLOR,
            )
    # Cross rails close the frame at the X ends.
    for end, x in (("a", -RACK_LEN_X / 2.0 + RAIL_SQ), ("b", RACK_LEN_X / 2.0 - RAIL_SQ)):
        _author_box(
            stage,
            f"/Dishrack/cross_{end}",
            (x, 0.0, rail_z),
            (RAIL_SQ / 2.0, RACK_WID_Y / 2.0, RAIL_SQ / 2.0),
            RACK_COLOR,
        )
    stage.GetRootLayer().Save()
    n_boxes = 2 + 2 * N_TINE_PAIRS + 2
    print(f"[convert_plate_rack] wrote {out_path}: {n_boxes} box colliders, kinematic")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
    os.makedirs(out_dir, exist_ok=True)
    _write_plate(out_dir)
    _write_rack(out_dir)


if __name__ == "__main__":
    main()
