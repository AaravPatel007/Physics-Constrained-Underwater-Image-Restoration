"""UIEB validation: original BetaNet (beta_rgbd) vs longer run (beta_rgbd_big), both + gray-world, DAv2 depth.
Left: the first 6 validation images (fixed before looking at results). Right: per-image PSNR for all 90 images.
Reads results/uieb_outputs{,_big}.npz and results/real_eval{,_big}.json (written by eval_real.py)."""
import json
from pathlib import Path

import numpy as np

from vizstyle import INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N = 6
M = "BetaNet + gray-world"


def main():
    old, new = np.load(RES / "uieb_outputs.npz"), np.load(RES / "uieb_outputs_big.npz")
    po = json.load(open(RES / "real_eval.json"))["uieb"]["per_image"]
    pn = json.load(open(RES / "real_eval_big.json"))["uieb"]["per_image"]
    cols = [("Raw underwater", old, "raw", po["raw"]), ("BetaNet, original + gray-world", old, M, po[M]),
            ("BetaNet, longer run + gray-world", new, M, pn[M]), ("UIEB reference (preferred look)", old, "reference", None)]

    fig = plt.figure(figsize=(20, 3.0 * N))
    gs = fig.add_gridspec(N, 6, width_ratios=[1, 1, 1, 1, 0.12, 1.9])
    for i in range(N):
        for c, (title, src, key, per) in enumerate(cols):
            ax = fig.add_subplot(gs[i, c])
            no_axes(ax)
            ax.imshow(np.clip(src[f"{i}|{key}"].astype(np.float32), 0, 1))
            if i == 0:
                ax.set_title(title, fontsize=11)
            if per is not None:
                ax.text(0.02, 0.03, f"PSNR {per['psnr'][i]:.1f} dB · ΔE {per['de2000'][i]:.1f}", transform=ax.transAxes,
                        fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))

    ax = fig.add_subplot(gs[:, 5])
    a, b = np.array(po[M]["psnr"]), np.array(pn[M]["psnr"])
    lo, hi = min(a.min(), b.min()) - 0.5, max(a.max(), b.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.scatter(a, b, s=36, color=SERIES[1], edgecolor="white", lw=1, zorder=3)
    for i in range(N):
        ax.annotate(f"row {i + 1}", (a[i], b[i]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    d = b - a
    ax.text(0.03, 0.97, f"all 90 UIEB validation images\nmean: {a.mean():.2f} → {b.mean():.2f} dB\n"
            f"longer run better: {(d > 0).sum()} images\nlonger run worse: {(d < 0).sum()} images\n"
            f"(above the dashed line = longer run better)",
            transform=ax.transAxes, va="top", fontsize=10, color=INK2)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("PSNR, BetaNet original + gray-world (dB)")
    ax.set_ylabel("PSNR, BetaNet longer run + gray-world (dB)")
    ax.set_title("Per-image PSNR vs UIEB reference", fontsize=11)
    ax.set_aspect("equal")

    fig.suptitle("UIEB real photos (estimated DAv2 depth): original vs longer-trained BetaNet", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    out = ROOT / "figures" / "3_new_runs" / "uieb_original_vs_longer.png"
    fig.savefig(out)
    print(out, f"mean {a.mean():.2f} -> {b.mean():.2f}, better {(d > 0).sum()}, worse {(d < 0).sum()}")


if __name__ == "__main__":
    main()
