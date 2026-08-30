# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the TRI mug tree USD (dorhors_wood_mug_holder) -- TRI geometry, raw.

    uv run --no-sync --project ~/Documents/research/icra2027 isaaclab -p \
      source/isaaclab_tasks/isaaclab_tasks/contrib/trossen_mug_tree/assets/convert_mug_tree.py

Writes ``usd/mug_tree.usd`` next to this script:

    /MugTree                        Xform, STATIC collider (no RigidBodyAPI -- the rack pattern)
    /MugTree/visuals/mesh           TRI's glTF, TRI's body frame
    /MugTree/collisions_tree/mesh   the SAME triangles, approximation "none" (the meshes ruling)
    /MugTree/frames/<name>          every <frame> of TRI's SDF, as an Xform

The mesh is authored in TRI's body frame: glTF Y-up -> Z-up (Drake's own mapping,
(x, y, z) -> (x, -z, y)) and nothing else -- no recenter, no floor -- so the SDF's
frames (branch bases, at_rest) land where TRI put them. Geometry is asserted
against the SDF's own cylinders: base disc at z=0, bottom branch pair along +/-X.
"""

import math
import os

import numpy as np
import trimesh

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.trossen_mug_tree.assets import (
    MUG_TREE_GLTF_PATH,
    MUG_TREE_USD_PATH,
    TRI_COLLISION_CYLINDERS,
    TRI_FRAMES,
)

DISPLAY_COLOR = (0.55, 0.40, 0.25)
# Collision source. "sdf" (default): TRI's <collision> cylinders, tessellated into
# closed, outward-wound triangle meshes -- TRI's own contact model for this
# object, as a mesh. "gltf": the render mesh as the collider (an 894-piece soup
# with ~20% inward-wound faces on the trunk and branches; measured 2026-08-25).
COLLISION_SOURCE = os.environ.get("MUG_TREE_COLLISION", "sdf")
CYLINDER_SECTIONS = 32
# Convex SEGMENTATION (final ruling 2026-08-26): every piece convex -- but a
# hull PAIR emits one small contact manifold, and a mug hanging on one long
# branch cylinder was retained by too few points (ejected 4/4; census: 11 vs
# 35 retention contacts). Splitting each cylinder into short segments multiplies
# the pairs, so the loop always spans several manifolds. Narrow phase stays
# entirely on the convex path.
BRANCH_SEGMENTS = 1  # single piece per cylinder (segmentation and sphere/capsule variants stay as probes)
TRUNK_SEGMENTS = 1


def _cylinder_mesh(pos, rpy, radius, length) -> trimesh.Trimesh:
    """Closed cylinder (axis z, centered) posed by the SDF collision pose."""
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=CYLINDER_SECTIONS)
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    cyl.apply_transform(T)
    assert cyl.is_watertight and cyl.volume > 0, "cylinder tessellation must be closed and outward-wound"
    return cyl


def _load_body_frame() -> trimesh.Trimesh:
    mesh = trimesh.load(MUG_TREE_GLTF_PATH, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if not os.path.exists(MUG_TREE_GLTF_PATH):
        raise FileNotFoundError(MUG_TREE_GLTF_PATH)
    # Y-up -> Z-up, Drake's mapping for glTF.
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    # Assertions against TRI's SDF collision cylinders.
    assert abs(v[:, 2].min()) < 0.002, f"base bottom not at z=0: {v[:, 2].min():.4f}"
    band = v[(v[:, 2] > 0.14) & (v[:, 2] < 0.18) & (np.linalg.norm(v[:, :2], axis=1) > 0.03)]
    assert len(band), "no bottom-branch geometry in z 0.14-0.18"
    ex, ey = np.abs(band[:, 0]).max(), np.abs(band[:, 1]).max()
    assert ex > 0.08 and ey < 0.03, f"bottom branch pair not along X: |x|max={ex:.3f} |y|max={ey:.3f}"
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)


def _author_mesh(stage, path, mesh, collide, approximation="none"):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.GetPointsAttr().Set([Gf.Vec3f(*p) for p in mesh.vertices])
    prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    prim.GetFaceVertexIndicesAttr().Set([int(i) for f in mesh.faces for i in f])
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*DISPLAY_COLOR)])
    if collide:
        prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).GetApproximationAttr().Set(approximation)
    return prim


def _quat_from_rpy(r, p, y) -> Gf.Quatf:
    """SDF fixed-axis roll-pitch-yaw (R = Rz Ry Rx) -> Gf quaternion."""
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(w, Gf.Vec3f(x, yy, z))


def main():
    mesh = _load_body_frame()
    os.makedirs(os.path.dirname(MUG_TREE_USD_PATH), exist_ok=True)
    stage = Usd.Stage.CreateNew(MUG_TREE_USD_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/MugTree")
    stage.SetDefaultPrim(root.GetPrim())
    _author_mesh(stage, "/MugTree/visuals/mesh", mesh, collide=False)
    if COLLISION_SOURCE == "gltf":
        _author_mesh(stage, "/MugTree/collisions_tree/mesh", mesh, collide=True)
        n_col = 1
    else:
        n_col = 0
        # SPHERE-CHAIN branches (MUG_TREE_COLLISION=spheres, 2026-08-27 probe):
        # sphere-vs-mesh is an ANALYTIC primitive path -- no GJK -- so it
        # catches the loop's thin wall that hull-vs-mesh returns "separated" on
        # (measured: 0 branch<->handle contacts at the hung pose with hulls, 3
        # upward-normal ones raw). Trunk and base stay hulls.
        if COLLISION_SOURCE == "capsules":
            # NATIVE CAPSULES for the branches (2026-08-28): capsule-vs-mesh is
            # the analytic primitive path -- correct radial normals with NO
            # surface ripple (unlike a sphere chain) and no GJK (unlike a hull,
            # whose side-face normals came out horizontal on every pair and
            # could not carry the loop's weight). Trunk + base remain hulls.
            for name, (pos, rpy, radius, length) in TRI_COLLISION_CYLINDERS.items():
                if "branch" in name:
                    cap = UsdGeom.Capsule.Define(stage, f"/MugTree/collisions_{name}/capsule")
                    cap.GetRadiusAttr().Set(float(radius))
                    cap.GetHeightAttr().Set(float(max(length - 2 * radius, 1e-4)))  # USD capsule height excludes the caps
                    cap.GetAxisAttr().Set("Z")
                    xf = UsdGeom.Xformable(cap.GetPrim())
                    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                    xf.AddOrientOp().Set(_quat_from_rpy(*rpy))
                    cap.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
                    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
                else:
                    _author_mesh(stage, f"/MugTree/collisions_{name}/mesh", _cylinder_mesh(pos, rpy, radius, length), collide=True, approximation="convexHull")
                n_col += 1
            print(f"[convert_mug_tree] capsule branches: {n_col} colliders")
        if COLLISION_SOURCE == "spheres":
            for name, (pos, rpy, radius, length) in TRI_COLLISION_CYLINDERS.items():
                r, pch, y = rpy
                cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pch), math.sin(pch), math.cos(y), math.sin(y)
                R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],[sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],[-sp, cp*sr, cp*cr]])
                z_axis = R @ np.array([0.0, 0.0, 1.0])
                if "branch" in name:
                    _sp = float(os.environ.get('SPHERE_SPACING_R', '0.5'))
                    n_sph = max(2, int(round(length / (radius * _sp))))  # centers every _sp*r: surface ripple ~ 1 - sqrt(1-(_sp/2)^2)
                    for j in range(n_sph):
                        off = (-length / 2.0) + (j + 0.5) * (length / n_sph)
                        cpt = np.array(pos) + off * z_axis
                        sph = UsdGeom.Sphere.Define(stage, f"/MugTree/collisions_{name}_p{j}/sphere")
                        sph.GetRadiusAttr().Set(float(radius))
                        UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*cpt))
                        sph.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
                        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
                        n_col += 1
                else:
                    _author_mesh(stage, f"/MugTree/collisions_{name}/mesh", _cylinder_mesh(pos, rpy, radius, length), collide=True, approximation="convexHull")
                    n_col += 1
            print(f"[convert_mug_tree] sphere-chain branches: {n_col} colliders")
        # convexHull, FINAL (2026-08-28). Three days of "the hull tree ejects
        # the mug" were a POSE error, not physics: the authored goal put the
        # branch 5.5 mm OUTSIDE the handle loop (gate measured on the visual
        # glTF; TRI's collision handle is narrower), so every solver correctly
        # reported nothing to hold. With the branch through the collision loop
        # the hull tree hangs the mug on MJWarp, ICF and ICF-adaptive (drop
        # probes 2/2). A plain torus hung on hull/raw/capsule alike -- the
        # control that exposed it. Hulls keep the pair on the convex path.
        approx = os.environ.get("MUG_TREE_APPROX", "convexHull")
        for name, (pos, rpy, radius, length) in (TRI_COLLISION_CYLINDERS.items() if COLLISION_SOURCE not in ("spheres", "capsules") else []):
            n_seg = 1 if name == "base_col" else (TRUNK_SEGMENTS if "trunk" in name else BRANCH_SEGMENTS)
            r, pch, y = rpy
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pch), math.sin(pch), math.cos(y), math.sin(y)
            R = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])
            z_axis = R @ np.array([0.0, 0.0, 1.0])
            seg_len = length / n_seg
            for j in range(n_seg):
                # segment center along the cylinder's own axis
                off = (-length / 2.0) + (j + 0.5) * seg_len
                cpos = tuple(np.array(pos) + off * z_axis)
                _author_mesh(
                    stage, f"/MugTree/collisions_{name}_s{j}/mesh", _cylinder_mesh(cpos, rpy, radius, seg_len),
                    collide=True, approximation=approx,
                )
                n_col += 1
    for name, (pos, rpy) in TRI_FRAMES.items():
        xf = UsdGeom.Xform.Define(stage, f"/MugTree/frames/{name}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddOrientOp().Set(_quat_from_rpy(*rpy))
    stage.GetRootLayer().Save()
    print(f"[convert_mug_tree] wrote {MUG_TREE_USD_PATH}: visual {len(mesh.faces)} faces; collision={COLLISION_SOURCE} ({n_col} mesh prims); {len(TRI_FRAMES)} TRI frames")
    for name, (pos, rpy) in TRI_FRAMES.items():
        print(f"[convert_mug_tree]   frame {name:28s} pos={np.round(pos, 4)} rpy={np.round(rpy, 4)}")


if __name__ == "__main__":
    main()
