"""Slide-shaped predicted-vs-true coefficient scatter (4 square panels, no title: the slide has one).
-> figures/4_presentation_slides/coef_pred_vs_true.png. Run after eval_synth.py."""
from pathlib import Path

import numpy as np

from vizstyle import CH, CH_MARK, INK2, MUTED, plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "4_presentation_slides" / "coef_pred_vs_true.png"

P = np.load(ROOT / "results" / "synth_per_image.npz")
keys = [("beta_D", r"$\beta^D$  attenuation (1/m)", True), ("beta_B", r"$\beta^B$  backscatter (1/m)", True),
        ("B_inf", r"$B_\infty$  veiling light", False), ("E", r"$E$  sunlight colour", False)]
fig, axs = plt.subplots(1, 4, figsize=(13, 4.1), gridspec_kw={"wspace": 0.28})
for ax, (k, t, log) in zip(axs, keys):
    tr, pr = P[f"true|{k}"], P[f"pred|beta_rgbd|{k}"]
    lo, hi = min(tr.min(), pr.min()), max(tr.max(), pr.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    txt = []
    for c, name in enumerate("RGB"):
        ax.scatter(tr[:, c], pr[:, c], s=7, alpha=0.35, color=CH[name], marker=CH_MARK[name], lw=0, zorder=2)
        rel = np.median(np.abs(pr[:, c] - tr[:, c]) / tr[:, c]) * 100
        r2 = 1 - np.sum((pr[:, c] - tr[:, c]) ** 2) / np.sum((tr[:, c] - tr[:, c].mean()) ** 2)
        txt.append(f"{name} {rel:3.0f}%  R² {r2:.2f}")
    if log:
        ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    if not log:
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0] if k == "E" else [0.2, 0.4, 0.6]
        ax.set_xticks(ticks), ax.set_yticks(ticks)
    ax.set_box_aspect(1)
    ax.set_title(t, fontsize=13)
    ax.set_xlabel("true")
    ax.text(0.97, 0.04, "\n".join(txt), transform=ax.transAxes, va="bottom", ha="right", fontsize=9.5,
            color=INK2, family="DejaVu Sans Mono")
axs[0].set_ylabel("predicted")
handles = [plt.Line2D([], [], color=CH[c], marker=CH_MARK[c], ls="", label=n)
           for c, n in zip("RGB", ["Red", "Green", "Blue"])]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=11,
           title="654 unseen test rooms, random water  ·  % = median error", title_fontsize=10)
fig.savefig(OUT)
print(OUT)
