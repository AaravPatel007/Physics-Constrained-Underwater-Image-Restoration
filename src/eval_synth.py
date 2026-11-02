"""Evaluate on the held-out NYU test rooms (654 scenes) under synthetic water.

Two protocols:
  random : each test scene gets one random water (fixed seed)            -> main table
  jerlov : every test scene under each of the 10 exact Jerlov water types -> per-water-type breakdown
Methods: raw, gray-world, oracle (true coefficients), every checkpoint in results/<name>/best.pt,
and BetaNet fed Depth-Anything-V2 depth estimated *from the underwater image* (realistic setting).
Writes results/synth_eval.json (per-method means) and results/synth_per_image.npz (for plots).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import depth as dav2
from metrics import full_reference
from model import BetaNet, UNetDirect
from physics import JERLOV_N, add_water, gray_world, linear_to_srgb, restore, sample_water, srgb_to_linear
from train import load_split

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda"


def load(name):
    ck = torch.load(ROOT / "results" / name / "best.pt", map_location=DEV)
    a = ck["args"]
    net = (BetaNet(not a["no_depth"], pretrained=False) if a["model"] == "beta" else UNetDirect(not a["no_depth"]))
    net.load_state_dict(ck["state"])
    return net.to(DEV).eval(), a["model"]


def to_np(x):
    return linear_to_srgb(x).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def run(rgb, depth, water, nets, use_dav2, bs=16, seed=7):
    gen = torch.Generator().manual_seed(seed)
    res = {}
    coefs = {k: [] for k in nets if nets[k][1] == "beta"}
    coefs["dav2"] = []
    true = {"beta_D": [], "beta_B": [], "B_inf": [], "E": []}
    for b in range(0, len(rgb), bs):
        sl = slice(b, b + bs)
        J = srgb_to_linear(rgb[sl].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[sl].to(DEV).float()[:, None]
        w = {k: v[sl] for k, v in water.items()}
        I, _, _ = add_water(J, z, w)
        I = (I + 0.005 * torch.randn(I.shape, generator=gen).to(DEV)).clamp(0, 1)
        for k in true:
            true[k].append(w[k].cpu())
        Jt = to_np(J)
        outs = {"raw": I, "gray-world": gray_world(I), "oracle (true coefs)": restore(I, z, w)}
        for name, (net, kind) in nets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            if kind == "beta":
                coefs[name].append({k: v.cpu() for k, v in p.items()})
                outs[name] = restore(I, z, p)
            else:
                outs[name] = p
        if use_dav2:
            Is = (to_np(I) * 255).round().astype(np.uint8)
            zd = torch.from_numpy(np.stack(dav2.predict(list(Is), "metric-indoor"))).to(DEV)[:, None].clamp(0.1, 20)
            net = nets["beta_rgbd"][0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, zd)
            coefs["dav2"].append({k: v.cpu() for k, v in p.items()})
            outs["beta_rgbd + DAv2 depth"] = restore(I, zd, p)
        for name, Jh in outs.items():
            Jh = to_np(Jh)
            for i in range(len(Jh)):
                for m, v in full_reference(Jh[i], Jt[i]).items():
                    res.setdefault(name, {}).setdefault(m, []).append(v)
        print(b, flush=True)
    cat = lambda L: {k: torch.cat([d[k] for d in L]).numpy() for k in L[0]} if L else None
    return res, {k: cat(v) for k, v in coefs.items() if v}, {k: torch.cat(v).numpy() for k, v in true.items()}


def main():
    rgb, depth = load_split("test")
    names = [p.parent.name for p in sorted((ROOT / "results").glob("*/best.pt")) if not p.parent.is_symlink()]  # skip old-name links
    nets = {n: load(n) for n in names}
    print("models:", names)

    # timing (batch of 1, 240x320)
    I = torch.rand(1, 3, 240, 320, device=DEV)
    z = torch.rand(1, 1, 240, 320, device=DEV) * 5
    timing = {}
    for n, (net, _) in nets.items():
        for _ in range(5):
            net(I, z)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                net(I, z)
        torch.cuda.synchronize()
        timing[n] = (time.time() - t) / 50 * 1000

    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    res, coefs, true = run(rgb, depth, water, nets, use_dav2="beta_rgbd" in nets)
    summary = {"random": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()},
               "timing_ms": timing}
    np.savez(ROOT / "results" / "synth_per_image.npz",
             **{f"metric|{m}|{k}": np.array(v) for m, d in res.items() for k, v in d.items()},
             **{f"pred|{m}|{k}": v for m, d in coefs.items() for k, v in d.items()},
             **{f"true|{k}": v for k, v in true.items()})

    # per Jerlov type (BetaNet RGB-D only, plus raw / oracle) on 200 test scenes
    sub = slice(0, 200)
    summary["jerlov"] = {}
    for t in JERLOV_N:
        w = sample_water(200, DEV, torch.Generator().manual_seed(1), water_type=t)
        r, c, tr = run(rgb[sub], depth[sub], w, {k: v for k, v in nets.items() if k in ("beta_rgbd", "unet_rgbd")}, False)
        summary["jerlov"][t] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in r.items()}
        if "beta_rgbd" in c:
            summary["jerlov"][t]["beta_D_pred_mean"] = c["beta_rgbd"]["beta_D"].mean(0).tolist()
            summary["jerlov"][t]["beta_D_true"] = tr["beta_D"][0].tolist()
    json.dump(summary, open(ROOT / "results" / "synth_eval.json", "w"), indent=1)
    print(json.dumps(summary["random"], indent=1), json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
