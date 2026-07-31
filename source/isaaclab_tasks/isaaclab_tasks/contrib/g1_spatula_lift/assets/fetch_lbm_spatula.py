# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch the LBM Thimma wooden spatula from the lbm_eval models wheel.

Downloads the ``lbm_eval_models`` wheel (a zip, ~823 MB, MIT/Apache-2.0) from
the lbm_eval GitHub release and extracts only the Thimma natural spatula files
into the local LBM asset stash used by the other LBM-based tasks
(``~/Documents/code/newton-adaptive/scripts/assets/lbm/``), mirroring the
existing ``plates/`` / ``forks/`` layout. Run once with::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/contrib/g1_spatula_lift/assets/fetch_lbm_spatula.py

The download is cached: an already-downloaded wheel next to the stash is
reused. (For constrained links the zip central directory can be range-read to
extract single members without the full download; not implemented here.)
"""

import os
import urllib.request
import zipfile

WHEEL_URL = (
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/"
    "lbm_eval_models-1.1.0-py3-none-any.whl"
)
STASH_DIR = os.path.expanduser("~/Documents/code/newton-adaptive/scripts/assets/lbm")
WHEEL_PATH = os.path.join(STASH_DIR, "lbm_eval_models-1.1.0-py3-none-any.whl")

# wheel-internal members to extract -> stash-relative destinations
_PREFIX = "lbm_eval_models/tableware/spatulas"
MEMBERS = {
    f"{_PREFIX}/thimma_wood_natural_flat_spatula.sdf": "spatulas/thimma_wood_natural_flat_spatula.sdf",
    f"{_PREFIX}/thimma_wood_natural_flat_spatula.README.md": "spatulas/thimma_wood_natural_flat_spatula.README.md",
    f"{_PREFIX}/assets/thimma_wood_natural_flat_spatula.gltf": "spatulas/assets/thimma_wood_natural_flat_spatula.gltf",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula.bin": "spatulas/assets/thimma_wood_flat_spatula.bin",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_low.vtk": "spatulas/assets/thimma_wood_flat_spatula_low.vtk",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_color.png": "spatulas/assets/thimma_wood_flat_spatula_color.png",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_color.ktx2": "spatulas/assets/thimma_wood_flat_spatula_color.ktx2",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_normal.png": "spatulas/assets/thimma_wood_flat_spatula_normal.png",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_normal.ktx2": "spatulas/assets/thimma_wood_flat_spatula_normal.ktx2",
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_occlusion_roughness_metallic.png": (
        "spatulas/assets/thimma_wood_flat_spatula_occlusion_roughness_metallic.png"
    ),
    f"{_PREFIX}/assets/thimma_wood_flat_spatula_occlusion_roughness_metallic.ktx2": (
        "spatulas/assets/thimma_wood_flat_spatula_occlusion_roughness_metallic.ktx2"
    ),
}


def main():
    if not os.path.exists(WHEEL_PATH):
        print(f"[fetch_lbm_spatula] downloading {WHEEL_URL}")
        os.makedirs(STASH_DIR, exist_ok=True)
        urllib.request.urlretrieve(WHEEL_URL, WHEEL_PATH)
    print(f"[fetch_lbm_spatula] wheel: {WHEEL_PATH} ({os.path.getsize(WHEEL_PATH)} bytes)")

    with zipfile.ZipFile(WHEEL_PATH) as zf:
        for member, rel_dest in MEMBERS.items():
            dest = os.path.join(STASH_DIR, rel_dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            print(f"[fetch_lbm_spatula] {rel_dest} ({os.path.getsize(dest)} bytes)")
    print("[fetch_lbm_spatula] done")


if __name__ == "__main__":
    main()
