"""Depth Anything V2 estimates for the NYU training rooms, seen through fake water (review 2, DAv2-depth run).

On real photos the network gets DAv2's depth guess, not true depth. To train it on the same kind of input, each
room gets V fixed fake waters; the underwater image is rendered with the TRUE depth (so the physics is right), and
DAv2 is run on that underwater-looking image (so the guess is fooled by fog the same way as on a real photo).
Stored per (room, version): the water, exposure factor k, depth scale s and the DAv2 map (120x160, float16).
Validation rooms (last 288) get one version: the fixed validation water of train.py (seed 123), no augmentation.
train.py --stored-water uses this file; --depth-input dav2 feeds the DAv2 map, true otherwise (the control).

  python prep_dav2_train.py      # -> data/nyu_train_big_dav2.npz (~0.9 GB), ~20 min on the laptop GPU
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import depth as dav2
from physics import add_water, linear_to_srgb, sample_water, srgb_to_linear
from train import load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nyu_train_big_dav2.npz"
V, NVAL, MURKY_KEEP, DEV = 3, 288, 0.4, "cuda"


def main():
    rgb, depth = load_split("train_big")
    n = len(rgb)
    gen = torch.Generator().manual_seed(7)
    keys = ("beta_D", "beta_B", "B_inf", "E")
    water = {k: np.zeros((n, V, 3), np.float32) for k in keys}
    k_exp, s_depth = np.ones((n, V), np.float32), np.ones((n, V), np.float32)
    zd = np.zeros((n, V, 120, 160), np.float16)
    val_water = sample_water(NVAL, DEV, torch.Generator().manual_seed(123))  # identical to train.py's
    for i in range(n):
        is_val = i >= n - NVAL
        nv = 1 if is_val else V
        if is_val:
            w = {k: val_water[k][i - (n - NVAL)][None].cpu() for k in keys}
            kk, ss = torch.ones(1), torch.ones(1)
        else:
            w = sample_water(nv, "cpu", gen, murky_keep=MURKY_KEEP)
            kk = torch.empty(nv).uniform_(0.7, 1.3, generator=gen)
            ss = torch.exp(torch.empty(nv).uniform_(math.log(0.7), math.log(2.0), generator=gen))
            w = {**w, "B_inf": w["B_inf"] * kk[:, None]}
        J = srgb_to_linear(rgb[i].permute(2, 0, 1)[None].float().div(255)).repeat(nv, 1, 1, 1) * kk[:, None, None, None]
        z = depth[i].float()[None, None].repeat(nv, 1, 1, 1) * ss[:, None, None, None]
        I, _, _ = add_water(J, z, w)
        sigma = torch.empty(nv, 1, 1, 1).uniform_(0.002, 0.01, generator=gen)
        I = (I + sigma * torch.randn(I.shape, generator=gen)).clamp(0, 1)
        ims = [(linear_to_srgb(x).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8) for x in I]
        for v, m in enumerate(dav2.predict(ims, "metric-indoor", bs=nv)):
            zd[i, v] = F.interpolate(torch.from_numpy(m)[None, None], (120, 160), mode="area")[0, 0].numpy()
            for k in keys:
                water[k][i, v] = w[k][v].numpy()
            k_exp[i, v], s_depth[i, v] = kk[v], ss[v]
        if i % 500 == 0:
            print(f"{i}/{n}", flush=True)
    nver = np.where(np.arange(n) >= n - NVAL, 1, V).astype(np.int8)
    np.savez(OUT, zd=zd, k=k_exp, s=s_depth, nver=nver, **water)
    print(OUT, OUT.stat().st_size / 1e9, "GB")


if __name__ == "__main__":
    main()
