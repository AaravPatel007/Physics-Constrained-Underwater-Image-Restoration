"""Gallery: the FIRST 25 UIEB val photos (fixed in advance, not picked), raw vs the current model
(baseline_v1_60ep_bigdata + Day 1 fixes) vs the UIEB reference. 5 pages of 5 photos.
Output: figures/5_review2_day1_safe_restore/gallery/page_<n>.png  (+ per-photo ΔE00 / red-speck tags)

  python fig_day1_gallery.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from eval_day1 import to_np, uieb_val
from eval_real import predict, t
from eval_synth import load
from metrics import full_reference
from physics import srgb_to_linear
from safe_restore import safe_restore, speck_score
from vizstyle import no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
MODEL = "baseline_v1_60ep_bigdata"
N, PER = 25, 5


@torch.no_grad()
def main():
    out = ROOT / "figures" / "5_review2_day1_safe_restore" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    net = load(MODEL)[0]
    tuned = json.load(open(ROOT / "results" / "review2_day1" / "eval.json"))["tuning"]["beta_rgbd_big"]["chosen"]
    raws, refs, depths = uieb_val(90)
    rows, stats = [], []
    for i in range(N):
        I = srgb_to_linear(t(raws[i] / 255.0))
        z = t(np.clip(depths[i], 0.1, 30)[..., None])
        p, _ = predict(net, I, z)
        o = to_np(safe_restore(I, z, p, **tuned)[0])
        ref = refs[i] / 255.0
        r = full_reference(o, ref)
        base = full_reference(raws[i] / 255.0, ref)
        rows.append((raws[i] / 255.0, o, ref, r["de2000"], base["de2000"], speck_score(o)))
        stats.append((r["de2000"], base["de2000"], r["psnr"], base["psnr"]))
    for pg in range(N // PER):
        fig, axs = plt.subplots(PER, 3, figsize=(13, 2.75 * PER))
        for k in range(PER):
            i = pg * PER + k
            raw, o, ref, de, de_raw, sp = rows[i]
            for c, (im, ti) in enumerate([(raw, "Raw photo"), (o, "Current model (v1 + Day 1 fixes)"),
                                          (ref, "UIEB reference")]):
                ax = axs[k, c]
                no_axes(ax), ax.imshow(np.clip(im, 0, 1))
                if k == 0:
                    ax.set_title(ti, fontsize=11)
            axs[k, 0].text(0.02, 0.03, f"val #{i}  ΔE vs reference {de_raw:.1f}", transform=axs[k, 0].transAxes,
                           fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
            axs[k, 1].text(0.02, 0.03, f"ΔE {de:.1f} · specks {sp:.1f}", transform=axs[k, 1].transAxes,
                           fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
        fig.suptitle(f"UIEB val photos {pg * PER}–{pg * PER + PER - 1} (first 25, fixed in advance). "
                     "ΔE = colour distance to the reference, lower = closer", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"page_{pg + 1}.png")
        plt.close(fig)
    s = np.array(stats)
    print(f"mean ΔE: raw {s[:, 1].mean():.2f} -> current model {s[:, 0].mean():.2f}; "
          f"closer to reference in {int((s[:, 0] < s[:, 1]).sum())}/{N} photos; "
          f"PSNR raw {s[:, 3].mean():.2f} -> {s[:, 2].mean():.2f}")
    print(out)


if __name__ == "__main__":
    main()
