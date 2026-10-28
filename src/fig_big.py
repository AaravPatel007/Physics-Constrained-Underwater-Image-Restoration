"""fig22: example images, raw vs original BetaNet vs longer run (2 synthetic NYU test scenes, 2 real Sea-thru frames).
fig23: synthetic test results, original vs longer run (per Jerlov type + coefficient error), from synth_eval_big.json."""
import json
from pathlib import Path

import numpy as np
import torch

from eval_real import CHARTS, chart_error, predict, t
from eval_synth import DEV, load
from fig_results import srgb
from physics import JERLOV_N, add_water, gray_world, remove_water, restore, sample_water, srgb_to_linear
from train import load_split
from vizstyle import CH, CH_MARK, INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "3_new_runs"
NAMES = {"beta_rgbd": "BetaNet, original", "beta_rgbd_big": "BetaNet, longer run"}
SYNTH = [(120, "3C"), (430, "5C")]           # (NYU test index, exact Jerlov water)
REAL = ["D1/T_S03272", "D3/T_S04910"]


def psnr(a, b):
    return float(-10 * np.log10(np.mean((srgb(a) - srgb(b)) ** 2) + 1e-10))


@torch.no_grad()
def examples(nets):
    rgb, depth = load_split("test")
    rows = []
    for i, wt in SYNTH:
        J = srgb_to_linear(rgb[i:i + 1].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[i:i + 1].to(DEV).float()[:, None]
        w = sample_water(1, DEV, torch.Generator().manual_seed(1), water_type=wt)
        I = add_water(J, z, w)[0].clamp(0, 1)
        cells = [("Raw underwater", I, None)]
        for n, net in nets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            cells.append((NAMES[n], restore(I, z, p), f"β_D red {p['beta_D'][0, 0]:.2f} /m"))
        cells.append(("Clean truth", J, f"true β_D red {w['beta_D'][0, 0]:.2f} /m"))
        img = lambda x: x[0].permute(1, 2, 0).float().cpu().numpy()
        Jt = img(J)
        rows.append((f"NYU test room\nJerlov {wt} water", [(lab, img(x), note, None if lab == "Clean truth" else psnr(img(x), Jt))
                                                           for lab, x, note in cells]))

    d = np.load(ROOT / "data" / "seathru_small.npz")
    names = list(d["names"])
    for fr in REAL:
        k = names.index(fr)
        lin = d[f"lin_{k}"] / np.percentile(d[f"lin_{k}"], 99.5)
        gt = d[f"gt_{k}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        I, z = t(np.clip(lin, 0, 1)), t(gt[..., None])
        cells = [("Raw underwater", lin, None)]
        for n, net in nets.items():
            p, _ = predict(net, I, z)
            o = gray_world(remove_water(I, z, p).clamp(0, 1.5))[0].permute(1, 2, 0).cpu().numpy()
            note = f"grey error {chart_error(o, CHARTS[fr]):.1f}°" if fr in CHARTS else None
            cells.append((NAMES[n] + " + gray-world", o, note))
        if fr in CHARTS:
            cells[0] = ("Raw underwater", lin, f"grey error {chart_error(lin, CHARTS[fr]):.1f}°")
        cells.append(("", None, None))
        rows.append((f"Real Sea-thru\n{fr}", [(lab, im, note, None) for lab, im, note in cells]))

    fig, axs = plt.subplots(len(rows), 4, figsize=(16, 3.3 * len(rows)))
    for r, (rowlab, cells) in enumerate(rows):
        for c, (lab, im, note, ps) in enumerate(cells):
            ax = axs[r, c]
            no_axes(ax)
            if im is None:
                ax.axis("off")
                continue
            ax.imshow(srgb(im))
            if r in (0, 2):
                ax.set_title(lab, fontsize=11)
            text = " · ".join(s for s in (f"PSNR {ps:.1f} dB" if ps is not None else None, note) if s)
            if text:
                ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
        axs[r, 0].set_ylabel(rowlab, fontsize=10, color=INK2)
    fig.suptitle("Original vs longer-trained BetaNet: synthetic test rooms (top) and real Sea-thru frames (bottom)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig22_big_examples.png")


def results():
    S = json.load(open(ROOT / "results" / "synth_eval_big.json"))
    types = list(JERLOV_N)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    ax = axs[0]
    for m, lab, col, ls in [("raw", "Raw underwater", MUTED, ":"), ("oracle (true coefs)", "Oracle: true coefficients", INK2, "--"),
                            ("beta_rgbd", NAMES["beta_rgbd"], SERIES[0], "-"), ("beta_rgbd_big", NAMES["beta_rgbd_big"], SERIES[1], "-")]:
        ax.plot(types, [S["jerlov"][w][m] for w in types], ls=ls, color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_xlabel("Jerlov water type (clear ocean → murky coastal)")
    ax.set_ylabel("PSNR (dB), 200 test rooms")
    ax.set_title("Image quality by water type")
    ax.legend(fontsize=9)

    ax = axs[1]
    keys = ["beta_D", "beta_B", "B_inf", "E"]
    lab = {"beta_D": "β_D", "beta_B": "β_B", "B_inf": "B∞", "E": "E"}
    x = np.arange(len(keys) * 3)
    width = 0.38
    for j, n in enumerate(NAMES):
        v = [S["coef_median_rel_err_pct"][n][k][c] for k in keys for c in range(3)]
        bars = ax.bar(x + (j - 0.5) * width, v, width - 0.04, color=SERIES[j], label=NAMES[n])
        for b_, val in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, val + 0.3, f"{val:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{lab[k]}\n{c}" for k in keys for c in "RGB"], fontsize=9)
    ax.set_ylabel("median relative error (%), 654 test rooms")
    ax.set_title("Coefficient error (lower is better)")
    ax.legend(fontsize=9)
    fig.suptitle("Synthetic test (654 unseen NYU rooms): original vs longer-trained BetaNet", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig23_big_results.png")


if __name__ == "__main__":
    examples({n: load(n)[0] for n in NAMES})
    if (ROOT / "results" / "synth_eval_big.json").exists():
        results()
