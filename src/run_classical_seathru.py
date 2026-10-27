"""Classical baseline: Sea-thru (Akkaynak & Treibitz 2019) as re-implemented in the De-water repo
(external/De-water/src/pipeline/seathru_pipeline.py), given the TRUE range map.

Runs in its own process: that module monkey-patches np.tanh globally on import.
  synth : first 112 NYU test scenes, identical inputs to eval_synth.py (same water + noise seeds)
  real  : the 24 Sea-thru frames (linear RAW, SfM range)
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from metrics import full_reference
from physics import add_water, linear_to_srgb, sample_water, srgb_to_linear
from train import load_split

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "seathru_pipeline", ROOT / "external/De-water/src/pipeline/seathru_pipeline.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def synth(n=112, bs=16):
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), "cpu", torch.Generator().manual_seed(2026))
    gen = torch.Generator().manual_seed(7)
    out, times = {"psnr": [], "ssim": [], "de2000": []}, []
    for b in range(0, n, bs):
        sl = slice(b, b + bs)
        J = srgb_to_linear(rgb[sl].permute(0, 3, 1, 2).float() / 255)
        z = depth[sl].float()[:, None]
        I, _, _ = add_water(J, z, {k: v[sl] for k, v in water.items()})
        I = (I + 0.005 * torch.randn(I.shape, generator=gen)).clamp(0, 1)
        Jt = linear_to_srgb(J).permute(0, 2, 3, 1).numpy()
        for i in range(len(I)):
            t = time.time()
            res, *_ = st.run_seathru_pipeline(I[i].permute(1, 2, 0).numpy(), depth_map=z[i, 0].numpy(), device="cpu")
            times.append(time.time() - t)
            Jh = linear_to_srgb(torch.from_numpy(res).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            for k, v in full_reference(Jh, Jt[i]).items():
                out[k].append(float(v))
        print(b, np.mean(out["psnr"]), flush=True)
    return {**{k: float(np.mean(v)) for k, v in out.items()}, "n": n, "sec_per_image": float(np.mean(times)),
            "per_image": out}


def real():
    d = np.load(ROOT / "data" / "seathru_small.npz")
    outs, times = {}, []
    for i, name in enumerate(d["names"]):
        lin, gt = d[f"lin_{i}"], d[f"gt_{i}"].copy()
        lin = lin / np.percentile(lin, 99.5)
        gt[~(gt > 0.1)] = np.nanmax(gt)  # holes -> far
        t = time.time()
        res, *_ = st.run_seathru_pipeline(np.clip(lin, 0, 1), depth_map=gt, device="cpu")
        times.append(time.time() - t)
        outs[f"st_{i}"] = res
        print(name, f"{times[-1]:.1f}s", flush=True)
    np.savez_compressed(ROOT / "results" / "seathru_classical_real.npz", **outs)
    return float(np.mean(times))


if __name__ == "__main__":
    which = sys.argv[1:] or ["synth", "real"]
    res = {}
    if "real" in which:
        res["real_sec_per_image"] = real()
    if "synth" in which:
        res["synth"] = synth()
    json.dump(res, open(ROOT / "results" / f"classical_seathru_{'_'.join(which)}.json", "w"), indent=1)
