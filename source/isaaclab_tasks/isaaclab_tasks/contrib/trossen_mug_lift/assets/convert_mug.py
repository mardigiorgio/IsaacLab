# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the LBM Inomata white mug USD for the Trossen lift task — kit-free.

The USD is authored directly with ``pxr`` from the glTF geometry (loaded via
trimesh), following ``g1_spatula_lift/assets/convert_assets.py``. Run once:

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_mug_lift/assets/convert_mug.py

Writes ``usd/mug_inomata_white.usd`` next to this script.

Body frame is the Drake SDF frame: origin at the mug's bottom center, Z up,
handle along +X. Mass/COM/inertia are the SDF's values.

The collision geometry is authored as PRE-SPLIT pieces because the Newton
pipeline convex-hulls every collision prim independently: a single-hull mug
is a solid blob with no rim to pinch and no handle gap, so each piece must
already be near-convex for its hull to preserve grasp geometry:

    /Mug                              Xform + RigidBodyAPI + MassAPI
    /Mug/visuals/visuals              full visual mesh (no physics APIs)
    /Mug/collisions_base/mesh         bottom disk faces
    /Mug/collisions_wall_[0-7]/mesh   45-degree angular sectors of the wall
    /Mug/collisions_handle_[0-2]/mesh handle thirds by height

Sector hulls thicken the wall by the chord sagitta (~3 mm at 45 degrees) —
acceptable at pinch scale; the cavity and the handle opening survive.
Collision prims are hidden via ``purpose=guide``, NOT ``MakeInvisible()`` —
invisible prims are dropped by the clone pipeline.
"""

import os

import numpy as np
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# The banked TRI source next to this script first; the historical stash paths
# stay as fallbacks for machines that still carry them.
_LBM_MUG_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbm_src")] + [
    os.path.expanduser(f"~/Documents/{d}")
    for d in (
        "research/newton-adaptive/scripts/assets/lbm/mugs/assets",
        "code/newton-adaptive/scripts/assets/lbm/mugs/assets",
    )
]

MUG_GLTF_NAME = "mug_inomata_white_mesh_collision.gltf"
# TRI's <collision> mesh for this mug (the SDF references it; the glTF is the <visual>).
MUG_COLLISION_VTK = "mug_inomata_white_low_16faces.vtk"
# from mug_inomata_white_mesh_collision.sdf
MUG_MASS = 0.0181  # [kg]
MUG_COM = (0.0017863, 0.0, 0.045564)  # [m], body frame
MUG_INERTIA_DIAG = (2.3341e-05, 2.5136e-05, 2.1762e-05)  # [kg m^2] about COM
MUG_HEIGHT = 0.097173  # [m], SDF mug_top_center frame
DISPLAY_COLOR = (0.92, 0.92, 0.90)

N_SECTORS = 8
BASE_Z_MAX = 0.008  # [m] faces below this are the base disk
HANDLE_X_MIN = 0.043  # [m] face centroids beyond this radius-x are handle
HANDLE_Z_SPLITS = (0.033, 0.058, 0.083)  # SDF handle corner/middle frames


def _find_gltf() -> str:
    for d in _LBM_MUG_DIRS:
        p = os.path.join(d, MUG_GLTF_NAME)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{MUG_GLTF_NAME} not found in {_LBM_MUG_DIRS}")


def _load_mesh() -> trimesh.Trimesh:
    mesh = trimesh.load(_find_gltf(), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # glTF is Y-up; the SDF body frame is Z-up. Detect and rotate if needed.
    if np.ptp(verts[:, 1]) > np.ptp(verts[:, 2]):
        verts = verts[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    # sanity: bottom at z~0, top at ~MUG_HEIGHT
    verts[:, 2] -= verts[:, 2].min()
    height = verts[:, 2].max()
    assert abs(height - MUG_HEIGHT) < 0.01, f"height {height:.4f} vs SDF {MUG_HEIGHT}"
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


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


def _load_collision_tets(path: str):
    """TRI's <collision> tet mesh: (points, tets)."""
    import meshio  # noqa: PLC0415

    m = meshio.read(path)
    return np.asarray(m.points, dtype=np.float64), np.vstack([c.data for c in m.cells if c.type == "tetra"])


def _closed_boundary(pts: np.ndarray, tets: np.ndarray) -> trimesh.Trimesh:
    """Boundary surface of a tet subset, outward-oriented: CLOSED by construction
    (cut faces between groups are included), which mesh-mesh contact needs --
    an open patch has no inside, and the narrow phase manufactures phantom
    penetrations against it (measured 2026-08-25: a mug in free air 20 mm from
    the trunk was 'supported' by a bogus contact and pulled in)."""
    faces = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    opposite = np.concatenate([tets[:, 3], tets[:, 2], tets[:, 1], tets[:, 0]])
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    bidx = first[counts == 1]
    bfaces = faces[bidx].copy()
    a, b, c = pts[bfaces[:, 0]], pts[bfaces[:, 1]], pts[bfaces[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), pts[opposite[bidx]] - a) > 0
    bfaces[inward] = bfaces[inward][:, [0, 2, 1]]
    surf = trimesh.Trimesh(vertices=pts, faces=bfaces, process=True)  # merges the VTK's duplicate points
    # Closed = every edge is used an even number of times (a tet group may touch
    # itself along an edge, which is non-manifold but still encloses a volume).
    edges = np.sort(surf.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    assert (cnt % 2 == 0).all(), f"piece boundary is open: {(cnt % 2).sum()} odd edges"
    assert surf.volume > 0, "piece boundary must be outward-wound"
    return surf


def _partition_tets(pts: np.ndarray, tets: np.ndarray) -> dict[str, np.ndarray]:
    """The mug's piece rules (base / 8 wall sectors / 3 handle bands) applied to TET centroids."""
    cen = pts[tets].mean(axis=1)
    handle = cen[:, 0] > HANDLE_X_MIN
    base = (~handle) & (cen[:, 2] < BASE_Z_MAX)
    wall = ~handle & ~base
    groups: dict[str, np.ndarray] = {"collisions_base": np.flatnonzero(base)}
    ang = np.arctan2(cen[:, 1], cen[:, 0])
    sector = ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    for s in range(N_SECTORS):
        idx = np.flatnonzero(wall & (sector == s))
        if len(idx):
            groups[f"collisions_wall_{s}"] = idx
    lo, mid, hi = HANDLE_Z_SPLITS
    z = cen[:, 2]
    bands = [z < (lo + mid) / 2, (z >= (lo + mid) / 2) & (z < (mid + hi) / 2), z >= (mid + hi) / 2]
    for k, band in enumerate(bands):
        idx = np.flatnonzero(handle & band)
        if len(idx):
            groups[f"collisions_handle_{k}"] = idx
    return groups


def _partition(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    """Face-index groups: base disk, 8 wall sectors, 3 handle segments."""
    tri_c = mesh.triangles_center
    handle = tri_c[:, 0] > HANDLE_X_MIN
    base = (~handle) & (tri_c[:, 2] < BASE_Z_MAX)
    wall = ~handle & ~base
    groups: dict[str, np.ndarray] = {"collisions_base": np.flatnonzero(base)}
    ang = np.arctan2(tri_c[:, 1], tri_c[:, 0])
    sector = ((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    for s in range(N_SECTORS):
        idx = np.flatnonzero(wall & (sector == s))
        if len(idx):
            groups[f"collisions_wall_{s}"] = idx
    z = tri_c[:, 2]
    lo, mid, hi = HANDLE_Z_SPLITS
    bands = [z < (lo + mid) / 2, (z >= (lo + mid) / 2) & (z < (mid + hi) / 2), z >= (mid + hi) / 2]
    for k, band in enumerate(bands):
        idx = np.flatnonzero(handle & band)
        if len(idx):
            groups[f"collisions_handle_{k}"] = idx
    return groups


# Contact impedance: dmax 0.999 stiffens the SATURATED region ~52x over the
# 0.95 default (MuJoCo stiffness scales as d/(1-d)), collapsing the pinch
# embed; entry d0 stays at the default 0.9 so first-detection forces keep the
# ramp the stable runs already survived. Width 2 mm = full hardness at the
# wall-thickness scale. priority 1 makes the mug's params WIN pair mixing
# against default-priority fingers/table instead of being solmix-halved.
MUG_SOLIMP = (0.9, 0.999, 0.002, 0.5, 2.0)
MUG_CONTACT_PRIORITY = 1


def _author_mesh(stage, path, mesh: trimesh.Trimesh, purpose_guide: bool):  # noqa: D103
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in mesh.vertices])
    prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    prim.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*DISPLAY_COLOR)])
    if purpose_guide:
        prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
        # MUG_HANDLE_APPROX (2026-08-27 probe): handle pieces may take a hull so the
        # branch<->handle pair runs hull-hull GJK instead of mesh-vs-hull midphase.
        _approx = os.environ.get("MUG_HANDLE_APPROX", "none") if "handle" in path else "none"
        api.GetApproximationAttr().Set(_approx)
        p = prim.GetPrim()
        if os.environ.get("MUG_NO_MJC", "0") != "1":
            p.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
                Vt.FloatArray([float(v) for v in MUG_SOLIMP])
            )
            p.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int, custom=True).Set(MUG_CONTACT_PRIORITY)
    return prim


def main():
    mesh = _load_mesh()  # the <visual> glTF
    pts, tets = _load_collision_tets(os.path.join(os.path.dirname(_find_gltf()), MUG_COLLISION_VTK))  # the <collision> VTK
    pts[:, 2] -= pts[:, 2].min()
    assert abs(pts[:, 2].max() - MUG_HEIGHT) < 0.01, f"collision height {pts[:, 2].max():.4f} vs SDF {MUG_HEIGHT}"
    groups = _partition_tets(pts, tets)
    assert sum(len(g) for g in groups.values()) == len(tets), "every tet must land in exactly one piece"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mug_inomata_white.usd")

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Mug")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.GetMassAttr().Set(MUG_MASS)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*MUG_COM))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*MUG_INERTIA_DIAG))

    _author_mesh(stage, "/Mug/visuals/visuals", mesh, purpose_guide=False)
    for name, face_idx in groups.items():
        # RAW TRI sub-mesh per piece, by the representation ruling (2026-08-24):
        # every TRI model collides as its authored triangles. The pre-split
        # survives only because the contact sensors and the handle-grasp gate
        # filter by piece name; the geometry inside each piece is TRI's own.
        piece = _closed_boundary(pts, tets[face_idx])  # CLOSED piece from its own tets
        _author_mesh(stage, f"/Mug/{name}/mesh", piece, purpose_guide=True)
        ext = piece.vertices.max(0) - piece.vertices.min(0)
        print(f"[convert_mug] {name}: {len(face_idx)} faces, extent {np.round(ext, 4)}")

    n_col = sum(len(g) for g in groups.values())
    stage.GetRootLayer().Save()
    print(f"[convert_mug] wrote {out_path}: {len(groups)} CLOSED TRI-collision pieces from {n_col} tets, mass {MUG_MASS} kg")


if __name__ == "__main__":
    main()
