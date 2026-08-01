# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the LBM Thimma spatula USD for the spatula-lift task — kit-free.

The USD is authored directly with ``pxr`` from the glTF geometry (loaded via
trimesh), no ``omni.kit.asset_converter`` dependency. Run once with::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/contrib/g1_spatula_lift/assets/convert_assets.py

Writes ``usd/thimma_wood_natural_flat_spatula.usd`` next to this script.

Vertices are converted glTF Y-up -> Z-up so the authored body frame matches
the Drake SDF frame (X along the length, blade at low X, handle at high X,
handle centerline rising from z=0.025 at x=0.15 to z=0.041 at x=0.23). Mass,
center of mass, and the SDF frame constants in the env cfg all assume this.

The collision geometry is split into TWO prims at a mesh-derived X plane so
shape-level contact filters can tell handle from blade contact (this is what
makes "pick it up by the handle" measurable):

    /Spatula                          Xform + RigidBodyAPI + MassAPI
    /Spatula/visuals/visuals          full visual mesh (no physics APIs)
    /Spatula/collisions_blade/mesh    x <  split plane, purpose=guide
    /Spatula/collisions_handle/mesh   x >= split plane, purpose=guide

Collision prims are hidden via ``purpose=guide``, NOT ``MakeInvisible()`` —
invisible prims are dropped by the clone pipeline. The split partitions the
FACES of the original mesh by triangle-centroid X (``slice_mesh_plane`` needs
shapely, not a repo dependency): no new vertices, the two open halves exactly
tile the original closed surface, and the boundary is jagged by at most one
triangle (~mm) — irrelevant for contact classification. Consequently the
prims carry NO NewtonSDFCollisionAPI tag — SDF needs closed geometry, and
this scene has no mesh-vs-mesh contact pair that would want one (the spatula
halves are the only raw meshes; table parts are boxes and robot links get
hulled by ``simplify_meshes``).
"""

import os

import numpy as np
import trimesh

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# the newton-adaptive checkout lives under ~/Documents/code on some machines
# and ~/Documents/research on others
_LBM_SPATULA_DIRS = [
    os.path.expanduser(f"~/Documents/{d}/newton-adaptive/scripts/assets/lbm/spatulas/assets")
    for d in ("code", "research")
]


def _lbm_asset(filename: str) -> str:
    for asset_dir in _LBM_SPATULA_DIRS:
        path = os.path.join(asset_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found in any of: {_LBM_SPATULA_DIRS}")


SPATULA_GLTF = _lbm_asset("thimma_wood_natural_flat_spatula.gltf")
SPATULA_VTK = _lbm_asset("thimma_wood_flat_spatula_low.vtk")
# from thimma_wood_natural_flat_spatula.sdf
SPATULA_MASS = 0.0664  # [kg]
SPATULA_COM = (0.065326, 2.3357e-05, 0.010802)  # [m], body frame
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")
OUT_PATH = os.path.join(OUT_DIR, "thimma_wood_natural_flat_spatula.usd")
DISPLAY_COLOR = (0.72, 0.55, 0.35)  # natural wood

# handle/blade split scan parameters. The mesh origin sits mid-spatula: the
# blade spans x ~[-0.053, 0.055], the handle/neck shaft ~[0.055, 0.247]
# (consistent with the SDF frames: handle center 0.15, hanging hole 0.23).
BIN_WIDTH = 0.005  # [m]
HANDLE_MAX_WIDTH = 0.045  # [m] — Y-width below this (past the blade) = handle
SPLIT_X_RANGE = (0.03, 0.10)  # [m] — sanity bounds for the found split plane


def _split_plane_x(verts: np.ndarray) -> float:
    """Find the handle/blade split X by scanning Y-width along the length.

    Walks +X in ``BIN_WIDTH`` bins from the widest (blade) region and returns
    the first bin start where the width has dropped below
    ``HANDLE_MAX_WIDTH``. Empty bins (possible over the blade slots) inherit
    the previous width so they cannot fake a narrow section.
    """
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    edges = np.arange(x_min, x_max + BIN_WIDTH, BIN_WIDTH)
    widths = []
    prev = 0.0
    for x0 in edges[:-1]:
        sel = (verts[:, 0] >= x0) & (verts[:, 0] < x0 + BIN_WIDTH)
        width = float(np.ptp(verts[sel, 1])) if sel.any() else prev
        widths.append(width)
        prev = width
    widths = np.asarray(widths)
    profile = ", ".join(f"{e:.3f}:{w:.3f}" for e, w in zip(edges[::4], widths[::4]))
    print(f"[convert_assets] width profile (x:width every 2 cm): {profile}")
    widest = int(np.argmax(widths))
    for i in range(widest, len(widths)):
        if widths[i] < HANDLE_MAX_WIDTH:
            split = float(edges[i])
            print(
                f"[convert_assets] width scan: max {widths.max():.3f} m at x={edges[widest]:.3f}, split at {split:.3f}"
            )
            return split
    raise RuntimeError("no handle section found in width scan")


def _vtk_tet_surface() -> trimesh.Trimesh:
    """Surface triangles of the LBM low-res tet mesh — the asset's OWN collision geometry.

    Drake uses ``*_low.vtk`` (846 pts / 3597 tets) for contact; its boundary
    surface (~5-6x fewer triangles than the visual glTF) is the right
    collision mesh here too. Legacy ASCII VTK is parsed directly (no meshio
    dependency): boundary faces are the tet faces that appear exactly once,
    wound outward via the owning tet's centroid. The VTK is already in the
    Drake/SDF frame — no up-axis conversion.
    """
    with open(SPATULA_VTK) as f:
        tokens = f.read().split()
    i = tokens.index("POINTS")
    n_pts = int(tokens[i + 1])
    pts = np.array(tokens[i + 3 : i + 3 + 3 * n_pts], dtype=np.float64).reshape(n_pts, 3)
    j = tokens.index("CELLS")
    n_cells, cell_size = int(tokens[j + 1]), int(tokens[j + 2])
    cells = np.array(tokens[j + 3 : j + 3 + cell_size], dtype=np.int64).reshape(n_cells, 5)
    assert (cells[:, 0] == 4).all(), "non-tet cells in VTK"
    tets = cells[:, 1:]

    tet_faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    owner = np.tile(np.arange(len(tets)), 4)
    keys = np.sort(tet_faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inverse] == 1
    faces, owners = tet_faces[boundary], owner[boundary]

    # wind outward: flip faces whose normal points toward the owning tet's centroid
    tri = pts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    to_centroid = pts[tets[owners]].mean(axis=1) - tri.mean(axis=1)
    flip = (normals * to_centroid).sum(axis=1) > 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    mesh = _submesh(pts, faces)
    print(f"[convert_assets] vtk collision surface: {len(mesh.faces)} tris, watertight={mesh.is_watertight}")
    return mesh


def _submesh(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Build a compact submesh from a face subset, dropping unreferenced vertices."""
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices=verts[used], faces=remap[faces], process=False)


def _author_collision_mesh(stage: Usd.Stage, path: str, mesh: trimesh.Trimesh):
    """Author one collision Mesh prim: CollisionAPI + purpose=guide (no SDF tag, see module docstring)."""
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    lo, hi = v.min(axis=0), v.max(axis=0)
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.GetPointsAttr().Set([Gf.Vec3f(*p) for p in v.tolist()])
    geom.GetFaceVertexCountsAttr().Set([3] * len(f))
    geom.GetFaceVertexIndicesAttr().Set(f.flatten().tolist())
    geom.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    geom.GetExtentAttr().Set([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    geom.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return len(f)


def main():
    mesh = trimesh.load(SPATULA_GLTF, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)  # glTF Y-up -> Z-up (SDF body frame)
    zmesh = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    lo, hi = v.min(axis=0), v.max(axis=0)
    ext = hi - lo
    print(f"[convert_assets] gltf verts={len(v)} faces={len(zmesh.faces)}")
    print(f"[convert_assets] bbox min={np.round(lo, 4).tolist()} max={np.round(hi, 4).tolist()}")
    # body frame == SDF frame checks: length along X, thin in Z, CoM inside bbox
    assert 0.26 <= ext[0] <= 0.34, f"X extent {ext[0]:.3f} not spatula-length — up-axis conversion wrong?"
    assert ext[2] < 0.08, f"Z extent {ext[2]:.3f} too tall — up-axis conversion wrong?"
    assert lo[0] - 0.02 <= SPATULA_COM[0] <= hi[0], "SDF CoM outside mesh bbox"

    split_x = _split_plane_x(v)
    assert SPLIT_X_RANGE[0] <= split_x <= SPLIT_X_RANGE[1], f"split x {split_x:.3f} outside {SPLIT_X_RANGE}"

    # collision geometry: the asset's own low-res tet-mesh surface, split at
    # the same plane (visual stays the full glTF)
    col = _vtk_tet_surface()
    cv = np.asarray(col.vertices, dtype=np.float64)
    col_ext = cv.max(axis=0) - cv.min(axis=0)
    assert abs(col_ext[0] - ext[0]) < 0.02 and abs(col_ext[2] - ext[2]) < 0.02, (
        f"VTK/glTF frame mismatch: extents {col_ext} vs {ext}"
    )
    cfaces = np.asarray(col.faces, dtype=np.int64)
    centroid_x = cv[cfaces].mean(axis=1)[:, 0]
    handle = _submesh(cv, cfaces[centroid_x >= split_x])
    blade = _submesh(cv, cfaces[centroid_x < split_x])
    for name, part in (("handle", handle), ("blade", blade)):
        # open at the cut plane by design (see module docstring); watertight
        # is informational here, not a requirement
        print(f"[convert_assets] {name}: {len(part.faces)} tris, watertight={part.is_watertight}")
        assert len(part.faces) > 50, f"{name} half suspiciously small — split plane wrong?"

    # handle centerline z at the grasp x (env-cfg HANDLE_GRASP_OFFSET_B sanity)
    grasp_sel = (v[:, 0] >= 0.195) & (v[:, 0] <= 0.205)
    if grasp_sel.any():
        zs = v[grasp_sel, 2]
        print(
            f"[convert_assets] handle section at x=0.20: z in [{zs.min():.4f}, {zs.max():.4f}]"
            f" center {0.5 * (zs.min() + zs.max()):.4f}"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    stage = Usd.Stage.CreateNew(OUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Spatula")
    stage.SetDefaultPrim(root.GetPrim())
    xf_root = UsdGeom.XformCommonAPI(root.GetPrim())
    xf_root.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
    xf_root.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0))
    xf_root.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))

    UsdGeom.Xform.Define(stage, "/Spatula/visuals")
    geom = UsdGeom.Mesh.Define(stage, "/Spatula/visuals/visuals")
    vis_v = np.asarray(zmesh.vertices, dtype=np.float32)
    vis_f = np.asarray(zmesh.faces, dtype=np.int32)
    geom.GetPointsAttr().Set([Gf.Vec3f(*p) for p in vis_v.tolist()])
    geom.GetFaceVertexCountsAttr().Set([3] * len(vis_f))
    geom.GetFaceVertexIndicesAttr().Set(vis_f.flatten().tolist())
    geom.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    geom.GetDisplayColorAttr().Set([Gf.Vec3f(*DISPLAY_COLOR)])
    geom.GetExtentAttr().Set([Gf.Vec3f(*lo.astype(np.float32).tolist()), Gf.Vec3f(*hi.astype(np.float32).tolist())])

    UsdGeom.Xform.Define(stage, "/Spatula/collisions_blade")
    n_blade = _author_collision_mesh(stage, "/Spatula/collisions_blade/mesh", blade)
    UsdGeom.Xform.Define(stage, "/Spatula/collisions_handle")
    n_handle = _author_collision_mesh(stage, "/Spatula/collisions_handle/mesh", handle)

    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.GetRigidBodyEnabledAttr().Set(True)
    rb.GetKinematicEnabledAttr().Set(False)
    rb.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.GetMassAttr().Set(SPATULA_MASS)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*SPATULA_COM))

    stage.GetRootLayer().Save()
    print(f"[convert_assets] split_x={split_x:.3f}, blade {n_blade} tris + handle {n_handle} tris")
    print(f"[convert_assets] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
