# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Pad an rsl_rl checkpoint for observation terms APPENDED to the policy obs group.

Widens the actor/critic first Linear (zero columns), the obs normalizers (mean 0,
var/std 1) and the matching Adam moments, so a policy trained on N obs dims can
resume on N+K without losing what it learned.

USAGE: python probe_pad_checkpoint.py <in.pt> <out.pt> <K>
"""
import sys

import torch

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(src, map_location="cpu")
touched = {}
for part in ("actor_state_dict", "critic_state_dict"):
    sd = ck[part]
    first = next(kk for kk, v in sd.items() if v.dim() == 2 and kk.endswith("weight"))
    w = sd[first]
    sd[first] = torch.cat([w, torch.zeros(w.shape[0], k, dtype=w.dtype)], dim=1)
    touched[(part, first)] = (tuple(w.shape), tuple(sd[first].shape))
    for kk in list(sd):
        if "obs_normalizer" in kk and sd[kk].dim() == 2:
            fill = 0.0 if kk.endswith("_mean") else 1.0
            sd[kk] = torch.cat([sd[kk], torch.full((1, k), fill, dtype=sd[kk].dtype)], dim=1)
opt = ck.get("optimizer_state_dict")
if opt and "state" in opt:
    for pid, st in opt["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            v = st.get(name)
            if torch.is_tensor(v) and v.dim() == 2 and any(v.shape == torch.Size(old) for (old, _) in touched.values()):
                st[name] = torch.cat([v, torch.zeros(v.shape[0], k, dtype=v.dtype)], dim=1)
torch.save(ck, dst)
print(f"padded {touched} -> {dst}")
