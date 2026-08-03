# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Make the spatula KINEMATIC in a saved scene so it can be placed by hand.

A dynamic spatula falls and reacts to contact the moment physics starts, so it
cannot be dragged into position with the sim running. Kinematic keeps it in the
physics scene (it still collides against the hand) but gravity and contacts no
longer move it — the transform gizmo owns its pose.

Run::

    ./isaaclab.sh -p .../assets/pin_spatula.py --in /tmp/spatula_ok2.usd --out /tmp/spatula_pin.usd
    ./isaaclab.sh -p .../assets/pin_spatula.py --in <saved> --out <new> --dynamic   # undo
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="src", required=True, help="input .usd")
parser.add_argument("--out", dest="dst", required=True, help="output .usd")
parser.add_argument("--dynamic", action="store_true", help="make it dynamic again instead")
args = parser.parse_args()

from pxr import Usd, UsdPhysics


def main():
    stage = Usd.Stage.Open(args.src)
    kinematic = not args.dynamic
    hits = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.endswith("/Spatula"):
            continue
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body:
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateKinematicEnabledAttr().Set(kinematic)
        hits += 1
        print(f"[pin] {path}: kinematicEnabled = {kinematic}")
    if not hits:
        raise SystemExit("[pin] no /Spatula prim found — check the input file")
    stage.Export(args.dst)
    print(f"[pin] wrote {args.dst}")

    check = Usd.Stage.Open(args.dst)
    for prim in check.Traverse():
        if str(prim.GetPath()).endswith("/Spatula"):
            attr = prim.GetAttribute("physics:kinematicEnabled")
            xf = prim.GetAttribute("xformOp:translate")
            print(f"[pin] VERIFY kinematicEnabled={attr.Get()}  translate={xf.Get() if xf else None}")


if __name__ == "__main__":
    main()
