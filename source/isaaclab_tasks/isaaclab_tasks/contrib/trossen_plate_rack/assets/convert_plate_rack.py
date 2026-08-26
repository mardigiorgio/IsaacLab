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
        api.GetApproximationAttr().Set("none")
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


def _author_box(stage, path, center, half, color, guide: bool = False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if guide:
        # Hidden collision stand-in: the render geometry is the real mesh.
        cube.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
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


# ------------------------------------------------------- TRI lbm_eval assets
# Source: ToyotaResearchInstitute/lbm_eval release 1.1.0, staged in lbm_src/.
# The IKEA Dinera 8" plate and the sweet_home drying rack are the validated
# geometry the LBM benchmark itself runs; mass/COM/inertia below are the
# plate SDF's authored values.
_LBM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbm_src")
TRI_PLATE_GLTF = "ikea_dinera_plate_8in_blue.gltf"
TRI_PLATE_COLLISION_VTK = "ikea_dinera_plate_8in_low_8faces.vtk"  # the SDF's <collision> mesh
TRI_RACK_PARTS = ("sweet_home_dish_drying_rack_wireframe", "sweet_home_dish_drying_rack_base")
TRI_PLATE_MASS = 0.375  # [kg]
TRI_PLATE_COM = (0.0, 0.0, 0.00915)  # [m]
TRI_PLATE_INERTIA = (8.6e-4, 8.6e-4, 1.71e-3)  # [kg m^2]


def _load_collision_vtk(path: str) -> trimesh.Trimesh:
    """TRI's COLLISION mesh: the boundary surface of the SDF's ``_low*.vtk`` tet
    mesh, faces oriented outward. This is the geometry TRI's own simulator
    collides with (the SDF's <collision> element); the glTF is the <visual>."""
    import meshio  # noqa: PLC0415

    m = meshio.read(path)
    tets = np.vstack([c.data for c in m.cells if c.type == "tetra"])
    pts = np.asarray(m.points, dtype=np.float64)
    faces = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    opposite = np.concatenate([tets[:, 3], tets[:, 2], tets[:, 1], tets[:, 0]])
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    bidx = first[counts == 1]
    bfaces = faces[bidx].copy()
    opp = pts[opposite[bidx]]
    a, b, c = pts[bfaces[:, 0]], pts[bfaces[:, 1]], pts[bfaces[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), opp - a) > 0
    bfaces[inward] = bfaces[inward][:, [0, 2, 1]]
    surf = trimesh.Trimesh(vertices=pts, faces=bfaces, process=False)
    surf.remove_unreferenced_vertices()
    return surf


def _load_zup(name: str) -> trimesh.Trimesh:
    """Load a Y-up glTF, map its length axis onto X, floor to z = 0.

    glTF (x, y, z) -> env (-z, -x, y): Y-up to Z-up plus a -90 degree yaw so
    the rack's slot axis (glTF z) runs along env X, the direction the plate
    task's tine-bracketing assumes. The plate is a surface of revolution, so
    the yaw is inert for it and the shared mapping keeps one code path.
    """
    mesh = trimesh.load(os.path.join(_LBM_SRC, name), force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    # trimesh bakes each glTF's scene transform, so files arrive in mixed
    # frames (the plate lands Z-up, the rack Y-up). Both assets' true up axis
    # is their MIN-extent axis (plate: shell thickness; rack: its height is
    # smaller than either footprint side), so detect rather than assume,
    # exactly as convert_mug.py does.
    if int(np.argmin(np.ptp(v, axis=0))) == 1:
        v = np.stack([-v[:, 2], -v[:, 0], v[:, 1]], axis=1)
    v[:, :2] -= (v[:, :2].max(0) + v[:, :2].min(0)) / 2.0
    v[:, 2] -= v[:, 2].min()
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)


def _write_plate_tri(out_dir: str) -> None:
    mesh = _load_zup(TRI_PLATE_GLTF)  # the <visual>
    col = _load_collision_vtk(os.path.join(_LBM_SRC, TRI_PLATE_COLLISION_VTK))  # the <collision>
    col.vertices[:, :2] -= (mesh.vertices[:, :2].max(0) + mesh.vertices[:, :2].min(0)) / 2.0 * 0.0  # VTK is already in the SDF body frame
    r_max = float(np.linalg.norm(mesh.vertices[:, :2], axis=1).max())
    z_max = float(mesh.vertices[:, 2].max())
    global PLATE_BASE_R, RIM_BAND_R  # noqa: PLW0603 -- partition thresholds scale with the asset
    PLATE_BASE_R = 0.58 * r_max
    RIM_BAND_R = 0.88 * r_max
    groups = _partition_plate(col)
    out_path = os.path.join(out_dir, "plate.usd")
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Plate")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.GetMassAttr().Set(TRI_PLATE_MASS)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*TRI_PLATE_COM))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*TRI_PLATE_INERTIA))
    _author_mesh(stage, "/Plate/visuals/visuals", mesh, collide=False, color=PLATE_COLOR)
    # RAW TRI sub-mesh per piece (representation ruling, 2026-08-24): the
    # sector split is kept for the piece-named contact sensors only.
    for name, face_idx in groups.items():
        piece = col.submesh([face_idx], append=True)
        _author_mesh(stage, f"/Plate/{name}/mesh", piece, collide=True, color=PLATE_COLOR)
    n_col = sum(len(g) for g in groups.values())
    assert n_col == len(col.faces), f"collision pieces cover {n_col} of {len(col.faces)} collision-mesh faces"
    stage.GetRootLayer().Save()
    print(
        f"[convert_plate_rack] wrote {out_path} (TRI ikea_dinera): {len(groups)} TRI-collision-mesh pieces, {n_col} faces (VTK boundary), "
        f"R={r_max:.4f} rim_z={z_max:.4f} -> cfg: PLATE_RIM_RADIUS~{0.97 * r_max:.3f} "
        f"PLATE_RIM_HEIGHT~{z_max - 0.002:.3f} PLATE_STAND_Z~{0.02 + r_max:.3f}"
    )


def _write_rack_tri(out_dir: str) -> None:
    out_path = os.path.join(out_dir, "dishrack.usd")
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Dishrack")
    stage.SetDefaultPrim(root.GetPrim())
    # STATIC collider, deliberately no RigidBodyAPI (TableGuard pattern): the
    # rack never moves, and OpenUSD 25.11's physics parser segfaults with
    # probability rising in its rigid-body descriptor count -- at 80 hull
    # prims under one rigid body the import died 5/5. Static colliders parse
    # through a different descriptor class and dodge the bug entirely.
    # The wireframe rests ON the base tray: stack it by the tray's height.
    z_off = {"sweet_home_dish_drying_rack_wireframe": 0.032, "sweet_home_dish_drying_rack_base": 0.0}
    n_col = 0
    for part in TRI_RACK_PARTS:
        mesh = _load_zup(part + ".gltf")
        mesh.vertices[:, 2] += z_off[part]
        short = "wire" if "wireframe" in part else "base"
        vis = UsdGeom.Mesh.Define(stage, f"/Dishrack/visuals_{short}/mesh")
        vis.GetPointsAttr().Set([Gf.Vec3f(*p) for p in mesh.vertices])
        vis.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        vis.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
        vis.GetDisplayColorAttr().Set([Gf.Vec3f(*RACK_COLOR)])
    # Collision: primitives FITTED to the wireframe's measured lattice --
    # the approach TRI's own SDF takes (its contact model is a separate
    # low-res VTK, not the render mesh). Per-component hulls fail because
    # components are welded subassemblies whose hulls wall off the slots
    # (measured: the plate ends up lying flat on top). The measured facts,
    # from the baked-frame vertices: 12 tine-loop planes perpendicular to Y
    # at 23.53 mm pitch spanning y = -0.145..+0.139, loops rising to
    # z = 0.114 over x = -0.19..+0.165; the base tray top at z = 0.032.
    loop_ys = [-0.145, -0.107, -0.084, -0.060, -0.037, -0.013, 0.010, 0.034, 0.057, 0.081, 0.105, 0.139]
    for k, y in enumerate(loop_ys):
        _author_box(
            stage,
            f"/Dishrack/collisions_loop_{k}",
            (-0.012, y, 0.030 + 0.084 / 2.0),
            (0.18, 0.002, 0.084 / 2.0),
            RACK_COLOR,
            guide=True,
        )
        n_col += 1
    _author_box(
        stage, "/Dishrack/collisions_tray", (0.0, 0.0, 0.032 / 2.0), (0.21, 0.15, 0.032 / 2.0), RACK_COLOR, guide=True
    )
    n_col += 1
    stage.GetRootLayer().Save()
    print(f"[convert_plate_rack] wrote {out_path} (TRI sweet_home visuals): {n_col} fitted slab colliders, static")


def _write_rack_mesh(out_dir: str) -> None:
    """MESH-contact rack: TRI's raw wireframe+base as static collision meshes.

    The mesh narrow phase is triangle-faithful, so no decomposition is needed
    and the wire lattice is exact by construction -- this is the asset the
    make-the-solver-fast-with-meshes program runs against. Static for the
    same OpenUSD parser reason as the slab build."""
    out_path = os.path.join(out_dir, "dishrack_mesh.usd")
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Dishrack")
    stage.SetDefaultPrim(root.GetPrim())
    z_off = {"sweet_home_dish_drying_rack_wireframe": 0.032, "sweet_home_dish_drying_rack_base": 0.0}
    for part in TRI_RACK_PARTS:
        mesh = _load_zup(part + ".gltf")
        mesh.vertices[:, 2] += z_off[part]
        short = "wire" if "wireframe" in part else "base"
        prim = UsdGeom.Mesh.Define(stage, f"/Dishrack/collisions_{short}/mesh")
        prim.GetPointsAttr().Set([Gf.Vec3f(*p) for p in mesh.vertices])
        prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        prim.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
        prim.GetDisplayColorAttr().Set([Gf.Vec3f(*RACK_COLOR)])
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
        api.GetApproximationAttr().Set("none")
    stage.GetRootLayer().Save()
    print(f"[convert_plate_rack] wrote {out_path} (TRI raw-mesh contact, static)")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(_LBM_SRC, TRI_PLATE_GLTF)):
        _write_plate_tri(out_dir)
        _write_rack_tri(out_dir)
        _write_rack_mesh(out_dir)
    else:
        # No procedural fallback: every TRI model is TRI's own mesh, or nothing.
        raise FileNotFoundError(f"TRI source {TRI_PLATE_GLTF!r} missing from {_LBM_SRC}; stage lbm_src first.")


if __name__ == "__main__":
    main()
