"""Method overview diagram + depth-model evaluation figure."""
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from prep_seathru import to_display
from vizstyle import AXIS, INK, INK2, MUTED, SERIES, SURFACE, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec=AXIS, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", fontsize=9.5, color=INK2)


def arrow(ax, a, b, label=""):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=MUTED))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, label, ha="center", fontsize=9, color=INK2)


def method_diagram():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15), ax.set_ylim(0, 5.2)
    no_axes(ax)
    blue_bg, blue_ec = "#cde2fb", SERIES[0]
    box(ax, 0.2, 3.2, 2.2, 1.1, "Underwater photo", "RGB, linearised")
    box(ax, 0.2, 1.0, 2.2, 1.1, "Range map z", "SfM / stereo, or DAv2\n+ one range reading")
    box(ax, 3.4, 2.0, 2.6, 1.4, "BetaNet", "ResNet-18 (ImageNet init)\n4-ch input, 11 M params", fc=blue_bg, ec=blue_ec)
    box(ax, 7.0, 2.0, 2.7, 1.4, "12 numbers", "β_D, β_B, B∞, E\n(× R, G, B)", fc="#fff4dc", ec=SERIES[3])
    box(ax, 10.6, 2.0, 2.0, 1.4, "Physics layer", "closed-form\ninversion", fc=blue_bg, ec=blue_ec)
    box(ax, 13.1, 2.0, 1.7, 1.4, "Clean image", "J")
    arrow(ax, (2.4, 3.75), (3.4, 3.0))
    arrow(ax, (2.4, 1.55), (3.4, 2.4))
    arrow(ax, (6.0, 2.7), (7.0, 2.7))
    arrow(ax, (9.7, 2.7), (10.6, 2.7))
    arrow(ax, (12.6, 2.7), (13.1, 2.7))
    ax.annotate("", xy=(11.6, 2.0), xytext=(1.3, 1.0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                connectionstyle="arc3,rad=0.25", linestyle=(0, (4, 3))))
    ax.text(6.4, 0.55, "z also goes into the physics layer", ha="center", fontsize=9.5, color=INK2)
    ax.text(10.6, 4.2, r"$J=\dfrac{\left(I-B_\infty(1-e^{-\beta_B z})\right)\,e^{\beta_D z}}{E}$", fontsize=15,
            ha="left", color=INK)
    ax.text(7.0, 1.55, "Output 1: the coefficients\n(what the water is)", fontsize=10, color=INK2, ha="left", va="top")
    ax.text(13.1, 1.55, "Output 2: the image", fontsize=10, color=INK2, ha="left", va="top")
    ax.text(0.2, 4.85, "Training: synthetic water on NYU-v2 RGB-D (true coefficients known)  →  "
            "loss = coefficient error + image error", fontsize=10.5, color=INK2)
    fig.savefig(FIG / "fig4_method_overview.png")
    plt.close(fig)


def depth_figure():
    ev = json.load(open(ROOT / "results" / "depth_eval.json"))["summary"]
    d = np.load(ROOT / "data" / "seathru_small.npz")
    order = [("relative", "Relative\n(scale+shift from GT)"),
             ("metric-indoor (median-scaled)", "Metric-indoor\n+ 1 range reading"),
             ("metric-outdoor (median-scaled)", "Metric-outdoor\n+ 1 range reading"),
             ("metric-indoor", "Metric-indoor\nas is"),
             ("metric-outdoor", "Metric-outdoor\nas is")]
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    vals = [ev[k]["absrel"] for k, _ in order]
    cols = [SERIES[0] if i < 3 else SERIES[1] for i in range(len(order))]
    y = np.arange(len(order))[::-1]
    ax.set_axisbelow(True)
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [l for _, l in order], fontsize=9.5)
    ax.set_xlabel("AbsRel error vs SfM range (lower is better)")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Shape right, scale wrong", fontsize=12)
    i = list(d["names"]).index("D3/T_S04910")
    gt = d[f"gt_{i}"].copy()
    gt[gt <= 0.1] = np.nan
    vmax = np.nanpercentile(gt, 99)
    panels = [(to_display(d[f"lin_{i}"]), "Sea-thru D3 frame"),
              (gt, "SfM range (ground truth)"),
              (d[f"dav2_metric-indoor_{i}"], "DAv2 metric-indoor (as is)")]
    for k, (im, t) in enumerate(panels):
        a = fig.add_subplot(gs[k + 1])
        if im.ndim == 2:
            h = a.imshow(im, cmap="Blues", vmin=0, vmax=vmax)
        else:
            a.imshow(im)
        no_axes(a)
        a.set_title(t, fontsize=11)
    cb = fig.colorbar(h, ax=fig.axes[-2:], fraction=0.025, pad=0.01)
    cb.set_label("metres (same scale)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Depth Anything V2 on 24 real Sea-thru frames: one metric anchor fixes most of the error",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
    fig.savefig(FIG / "fig5_depth_eval.png")
    plt.close(fig)


if __name__ == "__main__":
    method_diagram()
    depth_figure()
