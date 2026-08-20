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

_LBM_MUG_DIRS = [
    os.path.expanduser(f"~/Documents/{d}")
    for d in (
        "code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_mug_lift/assets/lbm_src",
        "code",
    )
] + [
    "/tmp/claude-1002/-home-mdigiorgio-Documents-code/fe8a844e-d1b0-4d64-833c-48934ee6d700/scratchpad/lbm_extract/lbm_eval_models/tableware/mugs/assets"
]

MUG_GLTF_NAME = "mug_inomata_white_mesh_collision.gltf"
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


def _author_mesh(stage, path, mesh: trimesh.Trimesh, purpose_guide: bool):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in mesh.vertices])
    prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    prim.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*DISPLAY_COLOR)])
    if purpose_guide:
        prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
        api.GetApproximationAttr().Set("convexHull")
        p = prim.GetPrim()
        p.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
            Vt.FloatArray([float(v) for v in MUG_SOLIMP])
        )
        p.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int, custom=True).Set(MUG_CONTACT_PRIORITY)
    return prim


def main():
    mesh = _load_mesh()
    groups = _partition(mesh)
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
        raw = mesh.submesh([face_idx], append=True)
        # Author the piece as a LOW-VERTEX convex hull, not the raw submesh:
        # the pipeline hulls each prim anyway, and pair-broadphase demand
        # scales with the triangle product of overlapping shapes — the mug is
        # one side of every grasp-phase pair, so its triangle budget bounds
        # the whole scene's pair demand. Farthest-point subsample to <= 24
        # vertices before hulling.
        verts = raw.vertices
        if len(verts) > 24:
            import numpy as _np

            sel = [int(_np.argmax(_np.linalg.norm(verts - verts.mean(0), axis=1)))]
            d = _np.linalg.norm(verts - verts[sel[0]], axis=1)
            for _ in range(23):
                sel.append(int(_np.argmax(d)))
                d = _np.minimum(d, _np.linalg.norm(verts - verts[sel[-1]], axis=1))
            verts = verts[sel]
        piece = trimesh.convex.convex_hull(verts)
        _author_mesh(stage, f"/Mug/{name}/mesh", piece, purpose_guide=True)
        ext = piece.vertices.max(0) - piece.vertices.min(0)
        print(f"[convert_mug] {name}: {len(face_idx)} faces, extent {np.round(ext, 4)}")

    stage.GetRootLayer().Save()
    print(f"[convert_mug] wrote {out_path}: {len(groups)} collision pieces, mass {MUG_MASS} kg")


if __name__ == "__main__":
    main()
