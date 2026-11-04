"""Figures for the sunlight-from-depth experiment. Reads sunlight_phy/results/*, writes sunlight_phy/figures/*.png.

Each figure answers one question (see README.md, "Figures"):
  1_depth_pred_vs_true        does the network read the depth below the surface?
  2_depth_error_distribution  how big are its errors, and how close is that to the best possible?
  3_depth_by_condition        where does it fail: water type, depth, sunlight colour?
  4_depth_gallery             are its depths sensible for individual scenes?
  5_metrics_comparison        old vs new: images, coefficients, sunlight, latency
  6_paired_psnr               old vs new, image by image
  7_jerlov_comparison         old vs new per Jerlov water type
  8_depth_bins_comparison     old vs new per depth below the surface
  9_illumination_attribution  how much image quality does each model lose to its sunlight estimate?
  10_error_correlation        do depth / sunlight errors cause the image errors?
  11_qualitative_synthetic    restorations next to the ground truth
  12_qualitative_uieb         restorations of real photos next to the UIEB reference
  13_real_results             real photos: UIEB scores, colour charts, predicted depths
  14_training_curves          training (incl. the v1 reproduction check)

  python sunlight_phy/figs_sun.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from vizstyle import AXIS, INK, INK2, MUTED, SERIES, SURFACE, no_axes, plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RES, FIG, RUNS = HERE / "results", HERE / "figures", HERE / "runs"
C = {"baseline": SERIES[0], "sun": SERIES[1], "oracle": MUTED, "floor": SERIES[2]}
M = {"baseline": "o", "sun": "s"}
LBL = {"baseline": "Baseline BetaNet (predicts E)", "sun": "Physics-informed (predicts d_s)"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cfe0f5", "#2a78d6", "#0b2f5c"])
JER = ["I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"]
DBINS = [(0.5, 3), (3, 6), (6, 9), (9, 12.01)]
plt.rcParams["axes.axisbelow"] = True


def ygrid(ax):
    ax.grid(axis="x", visible=False)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name, flush=True)


def load():
    S = json.load(open(RES / "synth_eval_sun.json"))
    P = dict(np.load(RES / "synth_per_image_sun.npz"))
    Jr = dict(np.load(RES / "synth_jerlov_sun.npz"))
    return S, P, Jr


def angle(a, b):
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stat_text(ax, s, x=0.03, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK2,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=0.6))


def fig_depth_scatter(S, P):
    dt, dp = P["true|d_s"][:, 0], P["pred|sun|d_s"][:, 0]
    dE = P["true|kappa"][:, 0] * dt / 0.8
    turb = P["true|beta_D"].mean(1)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, x, xl, key in [(axs[0], dt, "True depth below the surface d_s [m]", "vs_true_d_s"),
                           (axs[1], dE, "Depth that reproduces the true E with K_d = 0.8 β_D  [m]", "vs_E_equivalent_depth")]:
        sc = ax.scatter(x, dp, c=turb, cmap=SEQ, s=14, lw=0.3, ec=SURFACE, vmin=np.percentile(turb, 2),
                        vmax=np.percentile(turb, 98))
        lim = [0, 15.2]
        ax.plot(lim, lim, color=INK2, lw=1.2, ls="--", zorder=0)
        ax.text(12.6, 13.4, "perfect", color=INK2, fontsize=9, rotation=45, ha="center", va="bottom")
        ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
        ax.set_xlabel(xl), ax.set_ylabel("Predicted depth  d̂_s [m]")
        d = S["depth"][key]
        stat_text(ax, f"MAE {d['MAE_m']:.2f} m   RMSE {d['RMSE_m']:.2f} m\nrel. error {d['rel_err_pct']:.1f} %   "
                      f"R² {d['R2']:.2f}")
    axs[0].set_title("a  Predicted vs true depth", fontsize=12)
    axs[1].set_title("b  Predicted vs depth implied by true E", fontsize=12)
    fig.suptitle(f"{len(dt)} synthetic test scenes", x=0.01, ha="left", color=INK2, fontsize=10.5)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.02)
    cb.set_label("Water turbidity: mean β_D [1/m]", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "1_depth_pred_vs_true")


def fig_depth_hist(S, P):
    dt, dp, fl = P["true|d_s"][:, 0], P["pred|sun|d_s"][:, 0], P["floor_est"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
    bins = np.linspace(-6, 6, 49)
    axs[0].hist(dp - dt, bins, color=C["sun"], alpha=0.85, label="Network", rwidth=0.9)
    axs[0].hist(fl - dt, bins, histtype="step", color=C["floor"], lw=2, label="Best possible from E (perfect τ = κ d_s)")
    axs[0].axvline(0, color=INK2, lw=1)
    axs[0].set_xlabel("Depth error  d̂_s − d_s [m]"), axs[0].set_ylabel("Test scenes")
    axs[0].set_title("a  Depth error distribution")
    axs[0].legend(loc="upper left", fontsize=9)
    rows = [("Network", S["depth"]["vs_true_d_s"]), ("Best possible", S["depth"]["floor_from_perfect_tau"])]
    ax = axs[1]
    no_axes(ax)
    ax.set_title("b  Depth error summary")
    keys = [("MAE_m", "MAE [m]"), ("RMSE_m", "RMSE [m]"), ("median_abs_m", "median |err| [m]"),
            ("rel_err_pct", "mean rel. error [%]"), ("bias_m", "bias [m]"), ("R2", "R²")]
    for j, (k, lab) in enumerate(keys):
        y = 0.74 - j * 0.12
        ax.text(0.02, y, lab, color=INK2, fontsize=11, transform=ax.transAxes)
        for c, (nm, d) in enumerate(rows):
            ax.text(0.66 + c * 0.3, y, f"{d[k]:.2f}", color=INK, fontsize=11, transform=ax.transAxes, ha="right")
    for c, (nm, _) in enumerate(rows):
        ax.text(0.66 + c * 0.3, 0.87, nm, color=INK, fontsize=10.5, weight="bold", transform=ax.transAxes, ha="right")
    save(fig, "2_depth_error_distribution")


def fig_depth_conditions(S, P, Jr):
    dt, dp, fl = P["true|d_s"][:, 0], P["pred|sun|d_s"][:, 0], P["floor_est"]
    ER = P["true|E"][:, 0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    # a: Jerlov type (first 200 test rooms under each exact type)
    e = [np.abs(Jr[f"{t}|d_pred"] - Jr[f"{t}|d_true"]) for t in JER]
    bp = axs[0].boxplot(e, showfliers=False, widths=0.55, patch_artist=True, medianprops=dict(color=INK, lw=1.5))
    for b in bp["boxes"]:
        b.set(facecolor=C["sun"], alpha=0.75, edgecolor=SURFACE)
    axs[0].set_xticks(range(1, 11), JER)
    axs[0].set_xlabel("Jerlov water type (clear ocean → murky coast)")
    axs[0].set_ylabel("|d̂_s − d_s| [m]")
    axs[0].set_title("a  By water type")

    def binned(ax, key, edges, labels, title, xl):
        x = np.arange(len(edges))
        for off, (vals, col, lab) in zip((-0.18, 0.18), ((np.abs(dp - dt), C["sun"], "Network"),
                                                         (np.abs(fl - dt), C["floor"], "Best possible from E"))):
            q = lambda m, p: np.percentile(vals[m], p) if m.any() else np.nan
            med, q1, q3 = [q(m, 50) for m in edges], [q(m, 25) for m in edges], [q(m, 75) for m in edges]
            ax.bar(x + off, med, 0.34, color=col, alpha=0.85, label=lab)
            ax.errorbar(x + off, med, [np.subtract(med, q1), np.subtract(q3, med)], fmt="none", ecolor=INK2, lw=1, capsize=3)
        ax.set_xticks(x, [f"{l}\n(n={m.sum()})" for l, m in zip(labels, edges)])
        ax.set_title(title), ax.set_xlabel(xl)

    binned(axs[1], dt, [(dt >= lo) & (dt < hi) for lo, hi in DBINS], ["0.5–3", "3–6", "6–9", "9–12"],
           "b  By true depth", "Depth below the surface [m]")
    eb = [ER <= 0.25, (ER > 0.25) & (ER <= 0.5), (ER > 0.5) & (ER <= 0.8), ER > 0.8]
    binned(axs[2], ER, eb, ["≤ 0.25", "0.25–0.5", "0.5–0.8", "> 0.8"], "c  By sunlight colour (illumination)",
           "True red share of the sunlight E_R (low = red-starved)")
    axs[2].legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(bottom=0.27)
    fig.text(0.01, 0.0, "a: box = interquartile range, line = median, whiskers = 1.5 × IQR; 10 exact Jerlov types × 200 test "
             "rooms. b, c: bar = median, whiskers = interquartile range; 654-scene random-water test.", color=MUTED, fontsize=9)
    save(fig, "3_depth_by_condition")


def swatch(ax, x, y, rgb, s=0.07):
    ax.add_patch(plt.Rectangle((x, y), s, s, transform=ax.transAxes, fc=np.clip(rgb, 0, 1) ** (1 / 2.2), ec=AXIS, lw=0.6,
                               clip_on=False))


def fig_depth_gallery(S, P):
    G = np.load(RES / "synth_gallery_sun.npz")
    idx = S["gallery_idx"]
    fig, axs = plt.subplots(3, 4, figsize=(15, 9.6))
    for k, i in enumerate(idx[:12]):
        ax = axs[k % 3, k // 3]  # columns: depth bins; rows: turbidity terciles (label-only selection)
        ax.imshow(G[f"{i}|raw"].astype(np.float32))
        no_axes(ax)
        dt, dp = P["true|d_s"][i, 0], P["pred|sun|d_s"][i, 0]
        ax.set_title(f"true {dt:.1f} m · pred {dp:.1f} m · |err| {abs(dp - dt):.1f} m", fontsize=10, loc="left",
                     weight="normal", color=INK)
        ax.text(0.01, -0.05, f"β_D {P['true|beta_D'][i].mean():.2f}/m   E: true · old · new", transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top")
        for j, E in enumerate([P["true|E"][i], P["pred|baseline|E"][i], P["pred|sun|E"][i]]):
            swatch(ax, 0.74 + j * 0.085, -0.13, E / E.max())
    for c, lab in enumerate(["0.5–3 m", "3–6 m", "6–9 m", "9–12 m"]):
        axs[0, c].annotate(f"true depth {lab}", (0.5, 1.22), xycoords="axes fraction", ha="center", fontsize=12,
                           weight="bold", color=INK)
    for r, lab in enumerate(["clearer water", "medium", "more turbid"]):
        axs[r, 0].annotate(lab, (-0.08, 0.5), xycoords="axes fraction", ha="right", va="center", rotation=90,
                           fontsize=12, weight="bold", color=INK)
    fig.suptitle("Synthetic test scenes (input underwater image) with true and predicted depth below the surface. "
                 "One scene per depth × turbidity cell, chosen from the labels before evaluation.", x=0.01, ha="left",
                 fontsize=11, color=INK2, y=1.02)
    fig.subplots_adjust(hspace=0.45, wspace=0.05)
    save(fig, "4_depth_gallery")


def bars(ax, vals, labels, title, fmt="{:.2f}", ref=None, lower_better=False):
    x = np.arange(len(vals))
    cols = [C["baseline"], C["sun"]][:len(vals)]
    ax.bar(x, vals, 0.6, color=cols)
    if ref is not None:  # labelled once, in the figure legend
        ax.axhline(ref, color=MUTED, lw=1.2, ls="--", zorder=1)
    for xi, v in zip(x, vals):
        ax.annotate(fmt.format(v), (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color=INK, zorder=4, bbox=dict(fc=SURFACE, ec="none", pad=0.6))
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_title(title + ("\n(lower = better)" if lower_better else ""), fontsize=11)
    ax.margins(y=0.18)
    ygrid(ax)


def fig_metrics(S):
    r, cr, su, tm = S["random"], S["coef_median_rel_err_pct"], S["sunlight"], S["timing_ms"]
    lab = ["Baseline", "Physics-\ninformed"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    bars(axs[0, 0], [r["baseline"]["psnr"], r["sun"]["psnr"]], lab, "PSNR [dB]", ref=r["oracle (true coefs)"]["psnr"])
    bars(axs[0, 1], [r["baseline"]["ssim"], r["sun"]["ssim"]], lab, "SSIM", fmt="{:.3f}", ref=r["oracle (true coefs)"]["ssim"])
    bars(axs[0, 2], [r["baseline"]["de2000"], r["sun"]["de2000"]], lab, "CIEDE2000", lower_better=True,
         ref=r["oracle (true coefs)"]["de2000"])
    bars(axs[0, 3], [su["baseline"]["angle_deg_mean"], su["sun"]["angle_deg_mean"]], lab, "Sunlight colour error [°]",
         lower_better=True)
    for j, k in enumerate(["beta_D", "beta_B", "B_inf"]):
        ax = axs[1, j]
        x = np.arange(3)
        for o, n in zip((-0.2, 0.2), ("baseline", "sun")):
            ax.bar(x + o, cr[n][k], 0.38, color=C[n], label=LBL[n])
            for xi, v in zip(x + o, cr[n][k]):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, ["R", "G", "B"])
        ax.set_title({"beta_D": "β_D", "beta_B": "β_B", "B_inf": "B∞"}[k] + " median rel. error [%]\n(lower = better)",
                     fontsize=11)
        ax.margins(y=0.18)
        ygrid(ax)
    bars(axs[1, 3], [tm["baseline + restore"]["median_ms"], tm["sun + restore"]["median_ms"]], lab,
         "Latency: network + restore [ms]", lower_better=True)
    fig.suptitle("Synthetic test, 654 unseen NYU rooms, identical water / noise for both models", x=0.01, y=0.985,
                 ha="left", color=INK2, fontsize=11)
    h = [plt.Rectangle((0, 0), 1, 1, color=C["baseline"]), plt.Rectangle((0, 0), 1, 1, color=C["sun"]),
         plt.Line2D([], [], color=MUTED, ls="--", lw=1.2)]
    fig.legend(h, [LBL["baseline"], LBL["sun"], "true coefficients (oracle)"], loc="upper right", ncol=3,
               bbox_to_anchor=(1.0, 1.0), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "5_metrics_comparison")


def fig_paired(S, P):
    b, s = P["metric|baseline|psnr"], P["metric|sun|psnr"]
    d = S["paired_sun_minus_baseline"]["psnr"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axs[0]
    ax.scatter(b, s, s=12, color=C["sun"], lw=0.3, ec=SURFACE)
    lim = [min(b.min(), s.min()) - 1, max(b.max(), s.max()) + 1]
    ax.plot(lim, lim, color=INK2, ls="--", lw=1.2)
    ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
    ax.set_xlabel("Baseline PSNR [dB]"), ax.set_ylabel("Physics-informed PSNR [dB]")
    ax.set_title("a  PSNR per test scene")
    stat_text(ax, "above the line: new model better")
    ax = axs[1]
    ax.hist(s - b, np.linspace(-6, 6, 49), color=C["sun"], rwidth=0.9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("PSNR difference, new − old [dB]"), ax.set_ylabel("Test scenes")
    ax.set_title("b  Paired difference (new − old)")
    stat_text(ax, f"mean {d['mean']:+.2f} dB\n95 % CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]\n"
                  f"new better on {d['share_sun_better'] * 100:.0f} %")
    save(fig, "6_paired_psnr")


def fig_jerlov(S):
    J = S["jerlov"]
    x = np.arange(10)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for n in ("baseline", "sun"):
        axs[0].plot(x, [J[t][f"psnr|{n}"] for t in JER], marker=M[n], color=C[n], label=LBL[n])
        axs[1].plot(x, [J[t][f"E_angle_deg|{n}"] for t in JER], marker=M[n], color=C[n], label=LBL[n])
    axs[0].plot(x, [J[t]["psnr|oracle (true coefs)"] for t in JER], color=C["oracle"], ls="--", label="True coefficients")
    axs[0].plot(x, [J[t]["psnr|raw"] for t in JER], color=C["oracle"], ls=":", label="Raw underwater image")
    axs[0].set_ylabel("PSNR [dB]"), axs[0].set_title("a  Image quality per water type")
    axs[1].set_ylabel("Sunlight colour error [°]"), axs[1].set_title("b  Sunlight error per water type (lower = better)")
    for ax in axs:
        ax.set_xticks(x, JER), ax.set_xlabel("Jerlov water type (200 test rooms each)")
    axs[0].legend(fontsize=9, loc="lower left")
    save(fig, "7_jerlov_comparison")


def fig_depth_bins(P):
    dt = P["true|d_s"][:, 0]
    ms = [(dt >= lo) & (dt < hi) for lo, hi in DBINS]
    x = np.arange(4)
    ang = {n: angle(P[f"pred|{n}|E"], P["true|E"]) for n in ("baseline", "sun")}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for n in ("baseline", "sun"):  # ordered bins, value axes not from 0: lines like figure 7, not bars
        axs[0].plot(x, [P[f"metric|{n}|psnr"][m].mean() for m in ms], marker=M[n], color=C[n], label=LBL[n])
        axs[1].plot(x, [ang[n][m].mean() for m in ms], marker=M[n], color=C[n], label=LBL[n])
    axs[0].plot(x, [P["metric|oracle (true coefs)|psnr"][m].mean() for m in ms], color=C["oracle"], ls="--",
                label="True coefficients")
    labels = [f"{a}–{int(b)} m\n(n={m.sum()})" for (a, b), m in zip(DBINS, ms)]
    for ax in axs:
        ax.set_xticks(x, labels), ax.set_xlabel("True depth below the surface")
        ax.set_xlim(-0.3, 3.3)
    axs[0].set_ylabel("Mean PSNR [dB]"), axs[0].set_title("a  Image quality vs depth")
    axs[1].set_ylabel("Mean sunlight colour error [°]"), axs[1].set_title("b  Sunlight error vs depth (lower = better)")
    axs[0].legend(fontsize=9, loc="lower left")
    save(fig, "8_depth_bins_comparison")


def fig_attribution(S):
    r = S["random"]
    rows = [("Baseline, own E", r["baseline"]["psnr"], C["baseline"]),
            ("Baseline, true E", r["baseline | true E"]["psnr"], C["baseline"]),
            ("New, own depth", r["sun"]["psnr"], C["sun"]),
            ("New, true depth", r["sun | true d_s"]["psnr"], C["sun"]),
            ("New, true E", r["sun | true E"]["psnr"], C["sun"]),
            ("True coefs, E from true depth\n(K_d = 0.8 β_D)", r["true coefs | E(true d_s, K_d=0.8 beta_D)"]["psnr"], C["oracle"]),
            ("True coefficients", r["oracle (true coefs)"]["psnr"], C["oracle"])]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):  # dot plot: the x axis does not start at 0, so no bars
        own = lab.startswith(("Baseline, own", "New, own"))
        ax.scatter(v, yi, s=90, color=c, marker="o" if own else "D", zorder=3, ec=SURFACE, lw=1)
        ax.text(v, yi + 0.28, f"{v:.2f} dB", ha="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [r_[0] for r_ in rows])
    vals = [r_[1] for r_ in rows]
    ax.set_xlim(np.floor(min(vals)) - 1, np.ceil(max(vals)) + 1)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=60, color=MUTED, marker="o", label="model's own estimate")
    ax.scatter([], [], s=60, color=MUTED, marker="D", label="part of the estimate replaced by the truth")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Mean PSNR on the synthetic test [dB]")
    gb = r["baseline | true E"]["psnr"] - r["baseline"]["psnr"]
    gs = r["sun | true E"]["psnr"] - r["sun"]["psnr"]
    ax.set_title(f"PSNR lost to the sunlight estimate: baseline {gb:.2f} dB, physics-informed {gs:.2f} dB", fontsize=12)
    save(fig, "9_illumination_attribution")


def fig_correlation(P):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ad = np.abs(P["pred|sun|d_s"][:, 0] - P["true|d_s"][:, 0])
    cost = P["metric|sun | true d_s|psnr"] - P["metric|sun|psnr"]
    rho = spearmanr(ad, cost).statistic
    axs[0].scatter(ad, cost, s=12, color=C["sun"], lw=0.3, ec=SURFACE)
    axs[0].axhline(0, color=INK2, lw=1)
    axs[0].set_xlabel("|d̂_s − d_s| [m]"), axs[0].set_ylabel("PSNR gained by giving the true depth [dB]")
    axs[0].set_title("a  Physics-informed: depth error vs image cost")
    stat_text(axs[0], f"Spearman ρ = {rho:.2f}")
    txt = []
    for n in ("baseline", "sun"):
        a = angle(P[f"pred|{n}|E"], P["true|E"])
        c = P[f"metric|{n} | true E|psnr"] - P[f"metric|{n}|psnr"]
        axs[1].scatter(np.maximum(a, 0.1), c, s=12, color=C[n], marker=M[n], lw=0.3, ec=SURFACE, label=LBL[n], alpha=0.8)
        txt.append(f"{'old' if n == 'baseline' else 'new'}: ρ = {spearmanr(a, c).statistic:.2f}")
    axs[1].axhline(0, color=INK2, lw=1)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Sunlight colour error [°] (log scale, < 0.1° drawn at 0.1°)")
    axs[1].set_ylabel("PSNR gained by giving the true E [dB]")
    axs[1].set_title("b  Sunlight error vs image cost")
    axs[1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.86))
    stat_text(axs[1], "Spearman " + "   ".join(txt))
    save(fig, "10_error_correlation")


def fig_qual_synth(S, P):
    G = np.load(RES / "synth_gallery_sun.npz")
    idx = S["gallery_idx"][1::2][:6]  # every other gallery cell: all 4 depth bins, all 3 turbidity levels
    cols = [("raw", "Underwater input"), ("baseline", "Baseline BetaNet"), ("sun", "Physics-informed"), ("clean", "Ground truth")]
    fig, axs = plt.subplots(len(idx), 4, figsize=(12.5, 2.55 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        for c, (k, lab) in enumerate(cols):
            ax = axs[r, c]
            ax.imshow(G[f"{i}|{k}"].astype(np.float32))
            no_axes(ax)
            if r == 0:
                ax.set_title(lab, fontsize=12)
            if k in ("baseline", "sun"):
                ax.text(0.02, 0.04, f"{P[f'metric|{k}|psnr'][i]:.1f} dB  ΔE {P[f'metric|{k}|de2000'][i]:.1f}",
                        transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
            if k == "raw":
                ax.text(0.02, 0.04, f"d_s {P['true|d_s'][i, 0]:.1f} m (pred {P['pred|sun|d_s'][i, 0]:.1f})",
                        transform=ax.transAxes, fontsize=9, color="white", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    save(fig, "11_qualitative_synthetic")


def fig_qual_uieb(R):
    O = np.load(RES / "real_outputs_sun.npz")
    pi = R["uieb"]["per_image"]
    idx = [i for i in (0, 15, 30, 45, 60, 75) if f"uieb{i}|raw" in O]  # evenly spaced, fixed before looking at outputs
    cols = [("raw", "Real photo (UIEB)"), ("baseline", "Baseline BetaNet"), ("sun", "Physics-informed"),
            ("reference", "UIEB reference")]
    fig, axs = plt.subplots(len(idx), 4, figsize=(12.5, 2.7 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        for c, (k, lab) in enumerate(cols):
            ax = axs[r, c]
            ax.imshow(np.clip(O[f"uieb{i}|{k}"].astype(np.float32), 0, 1))
            no_axes(ax)
            if r == 0:
                ax.set_title(lab, fontsize=12)
            if k in ("baseline", "sun"):
                ax.text(0.02, 0.04, f"ΔE {pi[k]['de2000'][i]:.1f}", transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
            if k == "sun":
                ax.text(0.98, 0.04, f"d̂_s {R['uieb']['coefs']['sun'][i]['d_s'][0]:.1f} m", transform=ax.transAxes,
                        ha="right", fontsize=9, color="white", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    save(fig, "12_qualitative_uieb")


def fig_real(R):
    U = R["uieb"]["summary"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    groups = [("", "plain physics\n(own E)"), (" + Day 1b, own E", "Day 1b fixes\n+ own E"),
              (" + gray-world", "gray-world\n(E unused)"), (" + Day 1b", "Day 1b\n(E unused)")]
    x = np.arange(len(groups))
    for o, n in zip((-0.2, 0.2), ("baseline", "sun")):
        v = [U[n + g]["de2000"] for g, _ in groups]
        axs[0].bar(x + o, v, 0.38, color=C[n], label=LBL[n])
        for xi, vi in zip(x + o, v):
            axs[0].text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=8.5)
    axs[0].axhline(U["raw"]["de2000"], color=MUTED, ls=":", lw=1.2)
    axs[0].text(3.45, U["raw"]["de2000"], "raw photo", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    axs[0].set_xticks(x, [g[1] for g in groups], fontsize=9)
    axs[0].set_ylabel("CIEDE2000 to the UIEB reference"), axs[0].set_title("a  UIEB, 90 val photos (lower = better)")
    axs[0].margins(y=0.15)
    ygrid(axs[0])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=10)
    if "seathru" in R:
        ch = R["seathru"]["chart"]
        names = list(ch)
        x2 = np.arange(len(names))
        for o, n in zip((-0.2, 0.2), ("baseline", "sun")):
            v = [ch[f][n] for f in names]
            axs[1].bar(x2 + o, v, 0.38, color=C[n])
            for xi, vi in zip(x2 + o, v):
                axs[1].text(xi, vi, f"{vi:.1f}°", ha="center", va="bottom", fontsize=9)
            axs[1].scatter(x2 + o, [ch[f][f"{n} + gray-world"] for f in names], color=INK, marker="_", s=300, zorder=3)
        axs[1].scatter([], [], color=INK, marker="_", s=200, label="with gray-world instead of E")
        axs[1].set_xticks(x2, names), axs[1].set_ylabel("Grey-patch angular error [°]")
        axs[1].set_title("b  Sea-thru colour chart, own E (lower = better)")
        axs[1].legend(fontsize=8.5, loc="upper left")
        axs[1].margins(y=0.2)
        ygrid(axs[1])
        fr = R["seathru"]["frames"]
        dives = sorted({f.split("/")[0] for f in fr})
        for k, dv in enumerate(dives):
            ds = [fr[f]["sun"]["d_s"][0] for f in fr if f.startswith(dv + "/")]
            axs[2].scatter(np.full(len(ds), k) + np.linspace(-0.12, 0.12, len(ds)), ds, color=C["sun"], marker="s", s=30)
        ud = [c["d_s"][0] for c in R["uieb"]["coefs"]["sun"]]
        axs[2].scatter(np.full(len(ud), len(dives)) + np.random.default_rng(0).uniform(-0.2, 0.2, len(ud)), ud,
                       color=C["sun"], marker="s", s=12, alpha=0.6)
        axs[2].set_xticks(range(len(dives) + 1), dives + ["UIEB"])
        axs[2].set_ylabel("Predicted depth below the surface d̂_s [m]")
        axs[2].set_title("c  Predicted depth on real photos (no ground truth)")
        axs[2].set_ylim(0, 15.5)
    fig.tight_layout()
    save(fig, "13_real_results")


def fig_training():
    L = {"original v1 (committed log)": json.load(open(HERE.parent / "results/baseline_v1_60ep_bigdata/log.json")),
         "baseline, retrained here": json.load(open(RUNS / "baseline_v1_repro/log.json")),
         "physics-informed": json.load(open(RUNS / "sun_v1/log.json"))}
    sty = {"original v1 (committed log)": dict(color=MUTED, ls="--"), "baseline, retrained here": dict(color=C["baseline"]),
           "physics-informed": dict(color=C["sun"])}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for n, log in L.items():
        ep = [r["epoch"] for r in log]
        axs[0].plot(ep, [r["val_psnr"] for r in log], label=n, **sty[n])
        axs[1].plot(ep, [r["val_coef"] for r in log], label=n, **sty[n])
    sl = L["physics-informed"]
    axs[2].plot([r["epoch"] for r in sl], [r["val_depth_abs"] for r in sl], color=C["sun"], label="validation |d̂_s − d_s|")
    axs[0].set_ylabel("Validation PSNR [dB]"), axs[0].set_title("a  Validation image quality")
    axs[1].set_ylabel("Coefficient loss (lower = better)"), axs[1].set_title("b  Validation coefficient loss")
    axs[1].set_yscale("log")
    axs[2].set_ylabel("Mean |d̂_s − d_s| [m]"), axs[2].set_title("c  Validation depth error (new model)")
    for ax in axs:
        ax.set_xlabel("Epoch")
    axs[0].legend(fontsize=9, loc="lower right")
    save(fig, "14_training_curves")


def main():
    import argparse
    global RES, FIG, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--fig", default=str(FIG))
    ap.add_argument("--runs", default=str(RUNS))
    a = ap.parse_args()
    RES, FIG, RUNS = Path(a.res), Path(a.fig), Path(a.runs)
    S, P, Jr = load()
    fig_depth_scatter(S, P)
    fig_depth_hist(S, P)
    fig_depth_conditions(S, P, Jr)
    fig_depth_gallery(S, P)
    fig_metrics(S)
    fig_paired(S, P)
    fig_jerlov(S)
    fig_depth_bins(P)
    fig_attribution(S)
    fig_correlation(P)
    fig_qual_synth(S, P)
    if (RES / "real_eval_sun.json").exists():
        R = json.load(open(RES / "real_eval_sun.json"))
        fig_qual_uieb(R)
        fig_real(R)
    fig_training()


if __name__ == "__main__":
    main()
