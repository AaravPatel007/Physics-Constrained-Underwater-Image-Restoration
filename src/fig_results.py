"""Result figures from results/*.json|npz (run after eval_synth.py and eval_real.py)."""
import json
from pathlib import Path

import numpy as np

from physics import JERLOV_N
from vizstyle import AXIS, CH, CH_MARK, GRID, INK, INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OURS, ABL, BASE = SERIES[0], "#86b6ef", "#c3c2b7"
LABEL = {"raw": "Raw underwater", "gray-world": "Gray-world", "classical": "Sea-thru classical (true depth)",
         "unet_rgbd": "UNet black box (RGB-D)", "beta_rgb": "BetaNet, no depth (RGB)",
         "beta_coefonly": "BetaNet, coefficient loss only", "beta_imgonly": "BetaNet, image loss only",
         "beta_rgbd": "BetaNet (RGB-D)  [ours]", "beta_rgbd + DAv2 depth": "BetaNet + DAv2 depth from photo",
         "oracle (true coefs)": "Oracle: true coefficients"}


def srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def coef_scatter(P):
    keys = [("beta_D", r"$\beta_D$ attenuation (1/m)", True), ("beta_B", r"$\beta_B$ backscatter (1/m)", True),
            ("B_inf", r"$B_\infty$ veiling light", False), ("E", "E  sunlight colour", False)]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    out = {}
    for ax, (k, t, log) in zip(axs, keys):
        tr, pr = P[f"true|{k}"], P[f"pred|beta_rgbd|{k}"]
        lo, hi = min(tr.min(), pr.min()), max(tr.max(), pr.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        txt = []
        for c, name in enumerate("RGB"):
            ax.scatter(tr[:, c], pr[:, c], s=9, alpha=0.35, color=CH[name], marker=CH_MARK[name], lw=0, zorder=2)
            rel = np.median(np.abs(pr[:, c] - tr[:, c]) / tr[:, c]) * 100
            r2 = 1 - np.sum((pr[:, c] - tr[:, c]) ** 2) / np.sum((tr[:, c] - tr[:, c].mean()) ** 2)
            out[f"{k}_{name}"] = {"median_rel_err_pct": float(rel), "r2": float(r2)}
            txt.append(f"{name}: {rel:4.0f}% err, R² {r2:.2f}")
        if log:
            ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(t, fontsize=12)
        ax.set_xlabel("true")
        ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=9, color=INK2,
                family="DejaVu Sans Mono")
    axs[0].set_ylabel("predicted by BetaNet")
    handles = [plt.Line2D([], [], color=CH[c], marker=CH_MARK[c], ls="", label=n)
               for c, n in zip("RGB", ["Red", "Green", "Blue"])]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.06))
    fig.suptitle("Predicted vs true coefficients, 654 unseen test rooms with random water", x=0.01,
                 ha="left", fontsize=14, fontweight="bold", y=1.06)
    fig.savefig(FIG / "fig6_coef_scatter.png")
    plt.close(fig)
    return out


def identifiability(P, S):
    """Same image quality, very different coefficients: image loss alone does not pin down the physics."""
    runs = [("beta_rgbd", "Coefficient + image loss"), ("beta_imgonly", "Image loss only")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    tr = P["true|beta_D"]
    for ax, (r, t) in zip(axs, runs):
        pr = P[f"pred|{r}|beta_D"]
        lo, hi = 0.01, 4
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        for c, name in enumerate("RGB"):
            ax.scatter(tr[:, c], pr[:, c], s=9, alpha=0.35, color=CH[name], marker=CH_MARK[name], lw=0)
        rel = np.median(np.abs(pr - tr) / tr) * 100
        psnr = S["random"][r]["psnr"]
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_title(t, fontsize=12)
        ax.text(0.03, 0.97, f"image PSNR {psnr:.1f} dB\nβ_D median error {rel:.0f}%", transform=ax.transAxes,
                va="top", fontsize=10.5, color=INK)
        ax.set_xlabel(r"true $\beta_D$ (1/m)")
    axs[0].set_ylabel(r"predicted $\beta_D$ (1/m)")
    handles = [plt.Line2D([], [], color=CH[c], marker=CH_MARK[c], ls="", label=n)
               for c, n in zip("RGB", ["Red", "Green", "Blue"])]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.07))
    fig.suptitle("Good images do not imply right coefficients", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", y=1.07)
    fig.savefig(FIG / "fig14_identifiability.png")
    plt.close(fig)


def synth_bars(P, S, classical, n=112):
    """All methods on the same first-n test scenes (classical Sea-thru only ran on those)."""
    methods = ["raw", "gray-world", "classical", "unet_rgbd", "beta_rgb", "beta_coefonly", "beta_imgonly",
               "beta_rgbd", "beta_rgbd + DAv2 depth", "oracle (true coefs)"]
    vals = {}
    for m in methods:
        if m == "classical":
            vals[m] = {k: np.mean(classical["per_image"][k][:n]) for k in ["psnr", "ssim", "de2000"]}
        elif f"metric|{m}|psnr" in P:
            vals[m] = {k: P[f"metric|{m}|{k}"][:n].mean() for k in ["psnr", "ssim", "de2000"]}
    methods = [m for m in methods if m in vals]
    color = lambda m: OURS if m == "beta_rgbd" else (ABL if m.startswith("beta") else (
        "#ffffff" if m.startswith("oracle") else BASE))
    fig, axs = plt.subplots(1, 3, figsize=(17, 0.5 * len(methods) + 1.6), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (k, t) in zip(axs, [("psnr", "PSNR (dB)  higher is better"), ("ssim", "SSIM  higher is better"),
                                ("de2000", "Colour error ΔE₀₀  lower is better")]):
        v = [vals[m][k] for m in methods]
        ax.set_axisbelow(True)
        ax.barh(y, v, color=[color(m) for m in methods], edgecolor=[OURS if m.startswith("oracle") else "none" for m in methods],
                height=0.62, lw=1.2, ls="--")
        for yi, vi in zip(y, v):
            ax.text(vi, yi, f" {vi:.3f}" if k == "ssim" else f" {vi:.1f}", va="center", fontsize=9.5, color=INK)
        ax.set_title(t, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v) * 1.18)
    axs[0].set_yticks(y, [LABEL[m] for m in methods], fontsize=10)
    fig.suptitle(f"Synthetic test: {n} unseen rooms, random water (same scenes for every method)", x=0.01,
                 ha="left", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(FIG / "fig7_synth_results.png")
    plt.close(fig)
    return {m: {k: float(v) for k, v in d.items()} for m, d in vals.items()}


def jerlov_plot(S):
    types = list(JERLOV_N)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    rows = [("raw", BASE, "o", "Raw"), ("gray-world", SERIES[2], "^", "Gray-world"),
            ("unet_rgbd", SERIES[1], "s", "UNet black box"), ("beta_rgbd", OURS, "D", "BetaNet"),
            ("oracle (true coefs)", INK2, "", "Oracle (true coefficients)")]
    x = np.arange(len(types))
    for m, c, mk, lab in rows:
        if m not in S["jerlov"][types[0]]:
            continue
        v = [S["jerlov"][t][m]["psnr"] for t in types]
        ax.plot(x, v, color=c, marker=mk or None, markersize=8, ls="--" if not mk else "-", lw=2, label=lab)
    ax.set_xticks(x, ["I", "IA", "IB", "II", "III", "1", "3", "5", "7", "9"])
    ax.axvline(4.5, color=AXIS, lw=1)
    ax.text(2, ax.get_ylim()[1], "open ocean", ha="center", va="bottom", color=INK2)
    ax.text(7, ax.get_ylim()[1], "coastal", ha="center", va="bottom", color=INK2)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-0.3, 9.3)
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("Jerlov water type (clear → murky)")
    fig.suptitle("Per water type (200 test rooms each): big gains in clear water; in the murkiest water "
                 "the signal is gone", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.05)
    fig.savefig(FIG / "fig8_jerlov_breakdown.png")
    plt.close(fig)


def training_curves():
    runs = [("beta_rgbd", OURS, "BetaNet RGB-D"), ("beta_rgb", SERIES[1], "BetaNet, no depth"),
            ("beta_coefonly", SERIES[2], "BetaNet, coefficient loss only"),
            ("beta_imgonly", SERIES[3], "BetaNet, image loss only"), ("unet_rgbd", MUTED, "UNet black box")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for r, c, lab in runs:
        f = RES / r / "log.json"
        if not f.exists():
            continue
        L = json.load(open(f))
        e, v = [x["epoch"] for x in L], [x["val_psnr"] for x in L]
        ax.plot(e, v, color=c, label=lab, ls="--" if r.startswith("unet") else "-")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PSNR (dB)")
    fig.suptitle("Training curves (BetaNet ~15 s / epoch, UNet ~60 s / epoch on a laptop RTX 3050)", x=0.01,
                 ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(FIG / "fig9_training_curves.png")
    plt.close(fig)


def real_grid(R, R1):
    o = np.load(RES / "real_seathru_outputs.npz")
    o1 = np.load(RES / "v1_no_illuminant" / "real_seathru_outputs.npz")
    frames = ["D1/T_S03272", "D2/T_S03647", "D3/T_S04910", "D5/LFT_3374"]
    # every BetaNet column ends with gray-world, as Sea-thru itself does (v1 was evaluated that way too)
    cols = [("raw", o, "raw", "Raw"),
            ("gray-world", o, "gray-world", "Gray-world only"),
            ("st", o, "Sea-thru (classical, SfM range)", "Sea-thru classical (SfM)"),
            ("v1", o1, "BetaNet [SfM range]", "BetaNet v1: no sunlight colour"),
            ("v2", o, "BetaNet + gray-world [SfM range]", "BetaNet v2 (SfM range)"),
            ("v2d", o, "BetaNet + gray-world [DAv2 + 1 range reading]", "BetaNet v2 (DAv2 + 1 reading)"),
            ("unet", o, "UNet (black box, SfM range)", "UNet black box (SfM)")]
    fig, axs = plt.subplots(len(frames), len(cols), figsize=(24, 2.4 * len(frames)))
    for i, fr in enumerate(frames):
        for j, (key, src, m, title) in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(srgb(src[f"{fr}|{m}"]))
            no_axes(ax)
            if i == 0:
                ax.set_title(title, fontsize=10.5)
            if j == 0:
                ax.text(-0.04, 0.5, fr, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10, color=INK2)
            chart = (R1 if key == "v1" else R)["seathru"]["chart"].get(fr)
            if chart and m in chart:
                ax.text(0.02, 0.03, f"grey error {chart[m]:.1f}°", transform=ax.transAxes, fontsize=10,
                        color="white", bbox=dict(fc="#0b0b0b", alpha=0.6, lw=0, pad=2))
    fig.subplots_adjust(left=0.03, right=0.995, wspace=0.03, hspace=0.06)
    fig.suptitle("Real Sea-thru frames (Red Sea, 4–10 m depth)", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", y=0.96)
    fig.savefig(FIG / "fig10_real_seathru.png")
    plt.close(fig)


def site_consistency(R):
    fr = R["seathru"]["frames"]
    sites = ["D1", "D2", "D3", "D4", "D5"]
    desc = {"D1": "reef 10 m\nclear", "D2": "reef 10 m\nclear, hazy", "D3": "reef 4 m\nclear",
            "D4": "canyon 4-9 m\nturbid", "D5": "reef 5 m\nclear"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4), sharey=False)
    stats = {}
    for ax, (k, t) in zip(axs, [("beta_D", r"$\beta_D$ (1/m)"), ("beta_B", r"$\beta_B$ (1/m)"), ("B_inf", r"$B_\infty$")]):
        for c, name in enumerate("RGB"):
            for s_i, s in enumerate(sites):
                v = [f["SfM range"][k][c] for n, f in fr.items() if n.startswith(s)]
                x = s_i + (c - 1) * 0.22 + np.linspace(-0.05, 0.05, len(v))
                ax.scatter(x, v, color=CH[name], marker=CH_MARK[name], s=30, alpha=0.85, lw=0,
                           label=["Red", "Green", "Blue"][c] if s_i == 0 else None)
                stats.setdefault(k, {}).setdefault(s, {})[name] = {"mean": float(np.mean(v)), "cv": float(np.std(v) / np.mean(v))}
        ax.set_xticks(range(len(sites)), [f"{s}\n{desc[s]}" for s in sites], fontsize=9)
        ax.set_title(t, fontsize=12)
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Same dive → similar coefficients: BetaNet as a water-type sensor (no labels used)", x=0.01,
                 ha="left", fontsize=14, fontweight="bold", y=1.04)
    fig.savefig(FIG / "fig11_site_consistency.png")
    plt.close(fig)
    return stats


def hero():
    o = np.load(RES / "real_seathru_outputs.npz")
    frames = ["D3/T_S04910", "D2/T_S03647", "D5/LFT_3374"]
    m = "BetaNet + gray-world [DAv2 + 1 range reading]"
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.2))
    for j, fr in enumerate(frames):
        for i, (key, lab) in enumerate([("raw", "As photographed"), (m, "After BetaNet")]):
            axs[i, j].imshow(srgb(o[f"{fr}|{key}"]))
            no_axes(axs[i, j])
            if j == 0:
                axs[i, j].text(-0.03, 0.5, lab, transform=axs[i, j].transAxes, rotation=90, ha="right", va="center",
                               fontsize=11, color=INK2)
    fig.subplots_adjust(wspace=0.02, hspace=0.03)
    fig.text(0.13, 0.08, "Real Red Sea dive photos (Sea-thru D3, D2, D5). Depth from Depth Anything V2 + one range "
             "reading; ~3 ms per image for the coefficient network.", fontsize=10, color=INK2)
    fig.savefig(FIG / "fig0_hero.png")
    plt.close(fig)


def uieb_grid():
    o = np.load(RES / "uieb_outputs.npz")
    idx = [0, 2, 4, 6]
    cols = [("raw", "Raw"), ("depth", "DAv2 depth"), ("gray-world", "Gray-world"), ("BetaNet", "BetaNet"),
            ("BetaNet + gray-world", "BetaNet + gray-world"), ("UNet (black box)", "UNet black box"),
            ("reference", "UIEB reference")]
    cols = [c for c in cols if f"0|{c[0]}" in o]
    fig, axs = plt.subplots(len(idx), len(cols), figsize=(3.3 * len(cols), 2.6 * len(idx)))
    for i, k in enumerate(idx):
        for j, (m, t) in enumerate(cols):
            ax = axs[i, j]
            im = o[f"{k}|{m}"].astype(np.float32)
            ax.imshow(im, cmap="Blues") if im.ndim == 2 else ax.imshow(np.clip(im, 0, 1))
            no_axes(ax)
            if i == 0:
                ax.set_title(t, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.995, wspace=0.03, hspace=0.05)
    fig.suptitle("UIEB (no depth available: DAv2 used)", x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.95)
    fig.savefig(FIG / "fig12_uieb.png")
    plt.close(fig)


def main():
    P = np.load(RES / "synth_per_image.npz")
    S = json.load(open(RES / "synth_eval.json"))
    classical = json.load(open(RES / "classical_seathru_synth.json"))["synth"]
    R = json.load(open(RES / "real_eval.json"))
    R1 = json.load(open(RES / "v1_no_illuminant" / "real_eval.json"))
    summary = {"coef_accuracy": coef_scatter(P), "synth_subset112": synth_bars(P, S, classical)}
    jerlov_plot(S)
    identifiability(P, S)
    hero()
    training_curves()
    real_grid(R, R1)
    summary["site_consistency"] = site_consistency(R)
    uieb_grid()
    json.dump(summary, open(RES / "figure_numbers.json", "w"), indent=1)
    print(json.dumps(summary["coef_accuracy"], indent=1))


if __name__ == "__main__":
    main()
