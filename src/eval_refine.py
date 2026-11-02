"""RefineNet vs BetaNet: synthetic test-room gallery + real-photo (UIEB) check.

  synthetic : first N NYU test rooms under one fixed water (eval_synth 'random' protocol). Gallery
              raw | BetaNet | RefineNet | ground truth. Numbers are in results/refine_v1/test_eval.json.
  uieb      : BetaNet vs RefineNet on real UIEB photos, scored against the UIEB *reference* (NOT ground
              truth - no real photo has true coefficients). PSNR/SSIM/dE00 + paired bootstrap CI, gallery.
              RefineNet is fed the plain restore() output it was trained on (not the Day-1 safe_restore).

  python src/eval_refine.py --base baseline_v1_60ep_bigdata --name refine_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_day1 import boot_ci
from eval_real import predict, t
from eval_synth import load, to_np
from metrics import full_reference
from model import RefineNet
from physics import add_water, restore, sample_water, srgb_to_linear
from train import ROOT, load_split, make_batch

FIG = ROOT / "figures" / "8_refinenet"


def refined(ref, Jb, I, z):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return to_np(ref(Jb.clamp(0, 1.5), I, z))


def gallery(rows, titles, path, dpi=110):
    from vizstyle import no_axes, plt
    n = len(rows)
    fig, axs = plt.subplots(n, len(titles), figsize=(3 * len(titles), 3 * n))
    axs = np.atleast_2d(axs)
    for r, imgs in enumerate(rows):
        for c, im in enumerate(imgs):
            no_axes(axs[r, c]), axs[r, c].imshow(np.clip(im, 0, 1))
            if r == 0:
                axs[r, c].set_title(titles[c], fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


@torch.no_grad()
def synthetic(base, ref, n=8):
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), "cuda", torch.Generator().manual_seed(2026))
    gen = torch.Generator().manual_seed(7)
    rows = []
    for i in range(n):
        J = srgb_to_linear(rgb[i:i + 1].cuda().permute(0, 3, 1, 2).float() / 255)
        z = depth[i:i + 1].cuda().float()[:, None]
        w = {k: v[i:i + 1] for k, v in water.items()}
        I = (add_water(J, z, w)[0] + 0.005 * torch.randn(J.shape, generator=gen).cuda()).clamp(0, 1)
        Jb = restore(I, z, base(I, z))
        rows.append([to_np(I)[0], to_np(Jb)[0], refined(ref, Jb, I, z)[0], to_np(J)[0]])
    gallery(rows, ["raw (underwater)", "BetaNet", "BetaNet + RefineNet", "ground truth"], FIG / "synthetic_gallery.png")


@torch.no_grad()
def uieb(base, ref, n_metric=200, n_gallery=8):
    d = np.load(ROOT / "data" / "uieb_train.npz")
    n_metric = min(n_metric, int(d["n"]))
    res = {"BetaNet": {}, "BetaNet + RefineNet": {}}
    rows = []
    for i in range(n_metric):
        I = srgb_to_linear(t(d[f"raw_{i}"] / 255.0))
        z = t(np.clip(d[f"z_{i}"].astype(np.float32), 0.1, 30)[..., None])
        refimg = d[f"ref_{i}"] / 255.0
        _, Jb = predict(base, I, z)
        outs = {"BetaNet": to_np(Jb)[0], "BetaNet + RefineNet": refined(ref, Jb, I, z)[0]}
        for name, o in outs.items():
            for m, v in full_reference(o, refimg).items():
                res[name].setdefault(m, []).append(float(v))
        if i < n_gallery:
            rows.append([d[f"raw_{i}"] / 255.0, outs["BetaNet"], outs["BetaNet + RefineNet"], refimg])
    gallery(rows, ["raw (real photo)", "BetaNet", "BetaNet + RefineNet", "UIEB reference"], FIG / "uieb_gallery.png")
    summary = {name: {m: float(np.mean(v)) for m, v in dd.items()} for name, dd in res.items()}
    summary["ci_refine_minus_betanet"] = {m: boot_ci(res["BetaNet"][m], res["BetaNet + RefineNet"][m])
                                          for m in ("psnr", "ssim", "de2000")}
    summary["n_images"] = n_metric
    summary["note"] = "scored vs UIEB reference images, which are enhanced photos, NOT ground truth"
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    base = load(a.base)[0].requires_grad_(False).eval()
    ref = RefineNet().cuda().eval()
    ref.load_state_dict(torch.load(ROOT / "results" / a.name / "best.pt", map_location="cuda")["state"])

    synthetic(base, ref)
    u = uieb(base, ref)
    json.dump(u, open(ROOT / "results" / a.name / "uieb_eval.json", "w"), indent=1)
    print(f"\nUIEB real photos (N={u['n_images']}, vs reference — not ground truth):")
    print(f"{'':22s} {'PSNR':>6s} {'SSIM':>6s} {'dE00':>6s}")
    for k in ("BetaNet", "BetaNet + RefineNet"):
        print(f"{k:22s} {u[k]['psnr']:6.2f} {u[k]['ssim']:6.3f} {u[k]['de2000']:6.2f}")
    print("refine - betanet (95% CI):", u["ci_refine_minus_betanet"])


if __name__ == "__main__":
    main()
