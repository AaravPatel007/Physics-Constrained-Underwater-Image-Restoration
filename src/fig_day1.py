"""Day 1 (review 2) figures from results/review2_day1/ (written by eval_day1.py), one set per model:
  fins_steps_<model>.png     : the fins photo (UIEB val #2) after each fix in turn, with the red channel below
  uieb_examples_<model>.png  : the first 6 UIEB val images (fixed before looking), current recipe vs all Day 1 fixes
<model> = original (beta_rgbd) or longer_run (beta_rgbd_big). Output: figures/5_review2_day1_safe_restore/.
"""
import json
from pathlib import Path

import numpy as np

from vizstyle import no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "review2_day1"
OUT = ROOT / "figures" / "5_review2_day1_safe_restore"
MODELS = {"beta_rgbd": "original", "beta_rgbd_big": "longer_run"}
STEPS = ["current (gray-world)", "+ fog cap", "+ fog cap + smoothing", "all Day 1 fixes"]
TITLES = {"raw": "Raw photo", "current (gray-world)": "Current: fog removal + gray-world",
          "+ fog cap": "+ fog cap", "+ fog cap + smoothing": "+ fog cap + red smoothing",
          "all Day 1 fixes": "+ careful white balance (all Day 1)", "reference": "UIEB reference (preferred look)"}


def label(ax, per, key, i):
    s = f"specks {per[key]['speck'][i]:.1f}"
    if "psnr" in per[key]:
        s = f"PSNR {per[key]['psnr'][i]:.1f} dB · ΔE {per[key]['de2000'][i]:.1f} · " + s
    ax.text(0.02, 0.03, s, transform=ax.transAxes, fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))


def fins(o, per, name, i=2):
    cols = ["raw"] + STEPS + ["reference"]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 6.2))
    for c, k in enumerate(cols):
        im = np.clip(o[f"{i}|{k}"].astype(np.float32), 0, 1)
        no_axes(axs[0, c]), no_axes(axs[1, c])
        axs[0, c].imshow(im)
        axs[0, c].set_title(TITLES[k], fontsize=10)
        if k in per:
            label(axs[0, c], per, k, i)
        axs[1, c].imshow(im[..., 0], cmap="gray", vmin=0, vmax=1)
        if c == 0:
            axs[1, c].text(0.02, 0.97, "red channel only", transform=axs[1, c].transAxes, va="top", fontsize=9,
                           color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
    fig.suptitle(f"Fins photo (UIEB val #2): each Day 1 fix in turn ({name.replace('_', ' ')} BetaNet, no retraining)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"fins_steps_{name}.png")


def examples(o, per, name, n=6):
    cols = ["raw", "current (gray-world)", "all Day 1 fixes", "reference"]
    fig, axs = plt.subplots(n, len(cols), figsize=(3.8 * len(cols), 2.9 * n))
    for i in range(n):
        for c, k in enumerate(cols):
            no_axes(axs[i, c])
            axs[i, c].imshow(np.clip(o[f"{i}|{k}"].astype(np.float32), 0, 1))
            if i == 0:
                axs[i, c].set_title(TITLES[k], fontsize=10)
            if k in per:
                label(axs[i, c], per, k, i)
    fig.suptitle(f"UIEB val (first 6, fixed in advance): current recipe vs all Day 1 fixes "
                 f"({name.replace('_', ' ')} BetaNet)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"uieb_examples_{name}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(RES / "eval.json"))["uieb"]
    for m, name in MODELS.items():
        o = np.load(RES / f"uieb_outputs_{m}.npz")
        fins(o, ev[m]["per_image"], name)
        examples(o, ev[m]["per_image"], name)
    print(OUT)


if __name__ == "__main__":
    main()
