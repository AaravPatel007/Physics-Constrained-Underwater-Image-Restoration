"""All figures and tables of the v2 experiment. Reads three_model_comparisons/*.json and results_16_output/*.npz.

three_model_comparisons/    3m_image_quality, 3m_physical_parameters, 3m_jerlov, 3m_depth_bins, 3m_attribution,
                            3m_real, three_model_table.md
baseline_vs_16_comparisons/ b16_image_quality, b16_paired, b16_by_condition, b16_real, baseline_vs_16_table.md
results_16_output/          v2_training_curves, v2_kd_scatter, v2_e0, v2_depth, v2_sunlight, v2_error_vs_image
figures/                    individual_parameter_scatter, depth_meaning, real_unvalidated_predictions,
                            real_generalisation_training_water
qualitative_images/         gallery_3m_depth_turbidity, gallery_3m_jerlov_types, gallery_b16, gallery_uieb_3m,
                            gallery_uieb_b16

  python sunlight_depth_v2/figs_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
from scipy.stats import spearmanr  # noqa: E402
from vizstyle import AXIS, INK, INK2, MUTED, SERIES, SURFACE, CH, no_axes, plt  # noqa: E402

D3, D2, D16 = HERE / "three_model_comparisons", HERE / "baseline_vs_16_comparisons", HERE / "results_16_output"
DF, DQ = HERE / "figures", HERE / "qualitative_images"
M3 = ["baseline", "sun_v1", "sun_v2"]
C = {"baseline": SERIES[0], "sun_v1": SERIES[1], "sun_v2": SERIES[2], "oracle": MUTED}
MK = {"baseline": "o", "sun_v1": "s", "sun_v2": "D"}
LBL = {"baseline": "BetaNet (12, predicts E)", "sun_v1": "SunDepth v1 (10, d_s; K_d = 0.8 β_D)",
       "sun_v2": "SunDepth v2 (16, E0, K_d, d_s)"}
SHORT = {"baseline": "BetaNet\n(12)", "sun_v1": "v1\n(10)", "sun_v2": "v2\n(16)"}
JER = ["I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"]
DBINS = [(0.5, 3), (3, 6), (6, 9), (9, 12.01)]
plt.rcParams["axes.axisbelow"] = True


def save(fig, d, name):
    d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.png")
    plt.close(fig)
    print("wrote", d.name + "/" + name, flush=True)


def ygrid(ax):
    ax.grid(axis="x", visible=False)


def label_bars(ax, xs, vs, fmt, fs=9.5):
    for x, v in zip(xs, vs):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=fs, color=INK, bbox=dict(fc=SURFACE, ec="none", pad=0.4))


def bars(ax, models, vals, title, fmt="{:.2f}", ref=None, lower=False):
    x = np.arange(len(models))
    ax.bar(x, vals, 0.62, color=[C[m] for m in models])
    if ref is not None:
        ax.axhline(ref, color=MUTED, ls="--", lw=1.2, zorder=1)
    label_bars(ax, x, vals, fmt)
    ax.set_xticks(x, [SHORT[m] for m in models], fontsize=9.5)
    ax.set_title(title + ("\n(lower = better)" if lower else ""), fontsize=11)
    ax.margins(y=0.2)
    ygrid(ax)


def grouped(ax, models, vals, groups, title, fmt="{:.1f}", lower=True):
    """vals[m] = list per group."""
    x = np.arange(len(groups))
    wd = 0.8 / len(models)
    for j, m in enumerate(models):
        xs = x - 0.4 + wd / 2 + j * wd
        ax.bar(xs, vals[m], wd * 0.94, color=C[m], label=LBL[m])
        label_bars(ax, xs, vals[m], fmt, fs=7.5)
    ax.set_xticks(x, groups)
    ax.set_title(title + ("\n(lower = better)" if lower else ""), fontsize=11)
    ax.margins(y=0.22)
    ygrid(ax)


def legend_top(fig, models, extra=()):
    h = [plt.Rectangle((0, 0), 1, 1, color=C[m]) for m in models] + [e[0] for e in extra]
    fig.legend(h, [LBL[m] for m in models] + [e[1] for e in extra], loc="upper right", ncol=len(h),
               bbox_to_anchor=(1, 1.0), fontsize=9.5)


def oracle_line():
    return (plt.Line2D([], [], color=MUTED, ls="--", lw=1.2), "true coefficients (oracle)")


def load():
    S = json.load(open(D3 / "synth_eval.json"))
    P = dict(np.load(D16 / "synth_per_image.npz"))
    Jn = dict(np.load(D16 / "synth_jerlov.npz"))
    R = json.load(open(D3 / "real_eval.json")) if (D3 / "real_eval.json").exists() else None
    return S, P, Jn, R


def angle(a, b):
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def implied(P, m):
    if m == "sun_v2":
        return P["pred|sun_v2|K_d"], P["pred|sun_v2|E0"]
    if m == "sun_v1":
        return 0.8 * P["pred|sun_v1|beta_D"], np.ones_like(P["pred|sun_v1|beta_D"])
    return None, None


def boot_ci(x, n=4000, seed=0):
    r = np.random.default_rng(seed)
    m = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return np.percentile(m, [2.5, 97.5])


# ----------------------------------------------------------------------------------------------- three models
def fig_3m_image_quality(S, models=M3, d=D3, name="3m_image_quality"):
    M = S["metrics"]
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.4))
    o = M["oracle (true coefs)"]
    bars(axs[0], models, [M[m]["psnr"] for m in models], "PSNR [dB]", ref=o["psnr"])
    bars(axs[1], models, [M[m]["ssim"] for m in models], "SSIM", "{:.3f}", ref=o["ssim"])
    bars(axs[2], models, [M[m]["de2000"] for m in models], "CIEDE2000", ref=o["de2000"], lower=True)
    bars(axs[3], models, [S["sunlight_angle_deg"][m]["mean"] for m in models], "Sunlight colour error [°]", lower=True)
    bars(axs[4], models, [S["timing_ms"][f"{m} + restore"]["median_ms"] for m in models], "Latency, net + restore [ms]",
         lower=True)
    fig.suptitle(f"Synthetic test, {S['n_test']} unseen NYU rooms, v2 water; all models trained on the v2 water",
                 x=0.01, y=1.06, ha="left", color=INK2, fontsize=11)
    legend_top(fig, models, [oracle_line()])
    fig.tight_layout()
    save(fig, d, name)


def fig_3m_physical(S, P):
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    cr = S["coef_median_rel_err_pct"]
    for ax, k, t in zip(axs[0, :3], ("beta_D", "beta_B", "B_inf"), ("β_D", "β_B", "B∞")):
        grouped(ax, M3, {m: cr[m][k] for m in M3}, ["R", "G", "B"], f"{t} median rel. error [%]")
    bars(axs[0, 3], M3, [S["sunlight_angle_deg"][m]["mean"] for m in M3], "Sunlight E colour error [°]", lower=True)
    m2 = ["sun_v1", "sun_v2"]
    grouped(axs[1, 0], m2, {m: S["Kd_median_rel_err_pct"][m] for m in m2}, ["R", "G", "B"],
            "K_d median rel. error [%]\n(v1: implied 0.8·β̂_D)")
    bars(axs[1, 1], m2, [S["E0_angle_deg"][m]["mean"] for m in m2], "E0 colour error [°]\n(v1: fixed white)",
         lower=True)
    bars(axs[1, 2], m2, [S["depth"][m]["MAE_m"] for m in m2], "Depth d_s MAE [m]", lower=True)
    bars(axs[1, 3], m2, [S["depth"][m]["RMSE_m"] for m in m2], "Depth d_s RMSE [m]", lower=True)
    legend_top(fig, M3)
    fig.suptitle("Physical parameters on the synthetic test (ground truth from the v2 generator)", x=0.01, y=1.03,
                 ha="left", color=INK2, fontsize=11)
    fig.tight_layout()
    save(fig, D3, "3m_physical_parameters")


def fig_3m_jerlov(S, models=M3, d=D3, name="3m_jerlov"):
    J = S["jerlov"]
    x = np.arange(10)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for m in models:
        axs[0].plot(x, [J[t][f"psnr|{m}"] for t in JER], marker=MK[m], color=C[m], label=LBL[m])
        axs[1].plot(x, [J[t][f"E_angle|{m}"] for t in JER], marker=MK[m], color=C[m], label=LBL[m])
        if m != "baseline":
            axs[2].plot(x, [J[t][f"depth_MAE|{m}"] for t in JER], marker=MK[m], color=C[m], label=LBL[m])
    axs[0].plot(x, [J[t]["psnr|oracle (true coefs)"] for t in JER], color=MUTED, ls="--", label="true coefficients")
    axs[0].plot(x, [J[t]["psnr|raw"] for t in JER], color=MUTED, ls=":", label="raw underwater image")
    for ax, t, yl in zip(axs, ("a  Image quality", "b  Sunlight colour error (lower = better)",
                               "c  Depth error (lower = better)"), ("PSNR [dB]", "E error [°]", "d_s MAE [m]")):
        ax.set_xticks(x, JER), ax.set_xlabel("Jerlov water type (200 test rooms each)"), ax.set_title(t), ax.set_ylabel(yl)
    axs[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, d, name)


def fig_3m_depth_bins(P, models=M3, d=D3, name="3m_depth_bins"):
    dt = P["true|d_s"][:, 0]
    ms = [(dt >= lo) & (dt < hi) for lo, hi in DBINS]
    x = np.arange(4)
    labels = [f"{a}–{int(b)} m\n(n={m.sum()})" for (a, b), m in zip(DBINS, ms)]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for m in models:
        ang = angle(P[f"pred|{m}|E"], P["true|E"])
        axs[0].plot(x, [P[f"metric|{m}|psnr"][k].mean() for k in ms], marker=MK[m], color=C[m], label=LBL[m])
        axs[1].plot(x, [ang[k].mean() for k in ms], marker=MK[m], color=C[m], label=LBL[m])
        if f"pred|{m}|d_s" in P:
            e = np.abs(P[f"pred|{m}|d_s"][:, 0] - dt)
            axs[2].plot(x, [e[k].mean() for k in ms], marker=MK[m], color=C[m], label=LBL[m])
    axs[0].plot(x, [P["metric|oracle (true coefs)|psnr"][k].mean() for k in ms], color=MUTED, ls="--",
                label="true coefficients")
    for ax, t, yl in zip(axs, ("a  Image quality vs depth", "b  Sunlight error vs depth (lower = better)",
                               "c  Depth error vs depth (lower = better)"), ("Mean PSNR [dB]", "Mean E error [°]",
                                                                            "Mean |d̂_s − d_s| [m]")):
        ax.set_xticks(x, labels), ax.set_xlabel("True depth below the surface d_s"), ax.set_title(t), ax.set_ylabel(yl)
        ax.set_xlim(-0.3, 3.3)
    axs[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, d, name)


def fig_3m_attribution(S):
    M = S["metrics"]
    rows = [("BetaNet, own E", "baseline", "baseline", True), ("BetaNet, true E", "baseline | true E", "baseline", False),
            ("v1, own sunlight", "sun_v1", "sun_v1", True), ("v1, true d_s", "sun_v1 | true d_s", "sun_v1", False),
            ("v1, true E", "sun_v1 | true E", "sun_v1", False),
            ("v2, own sunlight", "sun_v2", "sun_v2", True), ("v2, true E0", "sun_v2 | true E0", "sun_v2", False),
            ("v2, true K_d", "sun_v2 | true K_d", "sun_v2", False), ("v2, true d_s", "sun_v2 | true d_s", "sun_v2", False),
            ("v2, true E", "sun_v2 | true E", "sun_v2", False),
            ("true coefs, v1 sunlight model", "true coefs | v1 sunlight (K_d = 0.8 beta_D, E0 white)", "oracle", False),
            ("true coefficients", "oracle (true coefs)", "oracle", False)]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, key, m, own) in zip(y, rows):
        v = M[key]["psnr"]
        ax.scatter(v, yi, s=85, color=C[m], marker="o" if own else "D", zorder=3, ec=SURFACE, lw=1)
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5)
    ax.set_yticks(y, [r[0] for r in rows])
    vals = [M[r[1]]["psnr"] for r in rows]
    ax.set_xlim(np.floor(min(vals)) - 0.5, np.ceil(max(vals)) + 1)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=60, color=MUTED, marker="o", label="model's own estimate")
    ax.scatter([], [], s=60, color=MUTED, marker="D", label="one quantity replaced by the truth")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Mean PSNR on the synthetic test [dB]")
    ax.set_title("Where each model loses image quality (synthetic test, v2 water)", fontsize=12)
    save(fig, D3, "3m_attribution")


def fig_3m_real(R, models=M3, d=D3, name="3m_real"):
    U = R["uieb"]["summary"]
    groups = [("", "plain physics\n(own E)"), (" + gray-world", "gray-world\n(E unused)"), (" + Day 1b", "Day 1b\n(E unused)")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.4, 1, 1.1]})
    grouped(axs[0], models, {m: [U[m + g]["de2000"] for g, _ in groups] for m in models}, [g[1] for g in groups],
            "a  UIEB, 90 val photos: CIEDE2000 to reference", fmt="{:.2f}")
    axs[0].axhline(U["raw"]["de2000"], color=MUTED, ls=":", lw=1.2)
    axs[0].annotate("raw photo", (2.45, U["raw"]["de2000"]), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=8.5, color=MUTED)
    ch = R["seathru"]["chart"]["D1/T_S03272"]
    x = np.arange(len(models))
    axs[1].bar(x, [ch[m] for m in models], 0.6, color=[C[m] for m in models])
    label_bars(axs[1], x, [ch[m] for m in models], "{:.1f}°")
    axs[1].scatter(x, [ch[f"{m} + gray-world"] for m in models], marker="_", s=400, color=INK, zorder=3,
                   label="with gray-world instead of own E")
    axs[1].set_xticks(x, [SHORT[m] for m in models]), axs[1].legend(fontsize=8.5, loc="upper left")
    axs[1].set_title("b  Sea-thru colour chart, own E\n(grey-patch error, lower = better)", fontsize=11)
    axs[1].margins(y=0.25), ygrid(axs[1])
    fr = R["seathru"]["frames"]
    for j, m in enumerate(m for m in models if m != "baseline"):
        ds_st = [fr[f][m]["d_s"][0] for f in fr]
        ds_u = [c["d_s"][0] for c in R["uieb"]["coefs"][m]]
        r = np.random.default_rng(j)
        axs[2].scatter(j * 2 + r.uniform(-0.25, 0.25, len(ds_u)), ds_u, s=12, color=C[m], marker=MK[m], alpha=0.6)
        axs[2].scatter(j * 2 + 1 + r.uniform(-0.2, 0.2, len(ds_st)), ds_st, s=22, color=C[m], marker=MK[m])
    axs[2].set_xticks([0, 1, 2, 3][:2 * len([m for m in models if m != "baseline"])],
                      ["v1\nUIEB", "v1\nSea-thru", "v2\nUIEB", "v2\nSea-thru"][:2 * len([m for m in models if m != "baseline"])])
    axs[2].set_ylabel("predicted d̂_s [m]"), axs[2].set_ylim(0, 15.5)
    axs[2].set_title("c  Predicted depth below the surface\n(UNVALIDATED: no ground truth)", fontsize=11)
    legend_top(fig, models)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, d, name)


def table_3m(S, R, models=M3, d=D3, name="three_model_table"):
    M = S["metrics"]
    L = [f"| metric | " + " | ".join(LBL[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    def row(lab, vals):
        L.append(f"| {lab} | " + " | ".join(vals) + " |")
    row("PSNR [dB] ↑", [f"{M[m]['psnr']:.2f}" for m in models])
    row("SSIM ↑", [f"{M[m]['ssim']:.3f}" for m in models])
    row("CIEDE2000 ↓", [f"{M[m]['de2000']:.2f}" for m in models])
    for k, t in (("beta_D", "β_D"), ("beta_B", "β_B"), ("B_inf", "B∞"), ("E", "E")):
        row(f"{t} median rel. error R/G/B [%] ↓", [" / ".join(f"{v:.1f}" for v in S["coef_median_rel_err_pct"][m][k]) for m in models])
    row("sunlight E colour error, mean [°] ↓", [f"{S['sunlight_angle_deg'][m]['mean']:.2f}" for m in models])
    row("E0 colour error, mean [°] ↓", [f"{S['E0_angle_deg'][m]['mean']:.2f}" + (" (fixed white)" if m == "sun_v1" else "")
                                        if m in S["E0_angle_deg"] else "n/a" for m in models])
    row("K_d median rel. error R/G/B [%] ↓", [(" / ".join(f"{v:.1f}" for v in S["Kd_median_rel_err_pct"][m])
                                              + (" (0.8 β̂_D)" if m == "sun_v1" else "")) if m in S["Kd_median_rel_err_pct"] else "n/a"
                                             for m in models])
    row("depth d_s MAE / RMSE [m] ↓", [f"{S['depth'][m]['MAE_m']:.2f} / {S['depth'][m]['RMSE_m']:.2f}" if m in S["depth"]
                                       else "n/a" for m in models])
    row("latency, network + restore [ms] ↓", [f"{S['timing_ms'][m + ' + restore']['median_ms']:.2f}" for m in models])
    if R:
        U = R["uieb"]["summary"]
        row("UIEB CIEDE2000, own E ↓", [f"{U[m]['de2000']:.2f}" for m in models])
        row("UIEB CIEDE2000, Day 1b ↓", [f"{U[m + ' + Day 1b']['de2000']:.2f}" for m in models])
        row("Sea-thru chart error, own E [°] ↓", [f"{R['seathru']['chart']['D1/T_S03272'][m]:.1f}" for m in models])
    (d / f"{name}.md").write_text("\n".join(L) + "\n")
    print("wrote", d.name + "/" + name + ".md")


# ----------------------------------------------------------------------------------------------- baseline vs 16
def fig_b16_paired(S, P):
    b, s = P["metric|baseline|psnr"], P["metric|sun_v2|psnr"]
    d = S["paired"]["sun_v2 - baseline"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    lim = [min(b.min(), s.min()) - 1, max(b.max(), s.max()) + 1]
    axs[0].scatter(b, s, s=11, color=C["sun_v2"], lw=0.3, ec=SURFACE)
    axs[0].plot(lim, lim, color=INK2, ls="--", lw=1.2)
    axs[0].set_xlim(lim), axs[0].set_ylim(lim), axs[0].set_aspect("equal")
    axs[0].set_xlabel("BetaNet (12) PSNR [dB]"), axs[0].set_ylabel("v2 (16) PSNR [dB]")
    axs[0].set_title("a  PSNR per test scene")
    axs[0].text(0.03, 0.97, "above the line: v2 better", transform=axs[0].transAxes, va="top", fontsize=9.5, color=INK2)
    for ax, m, t in ((axs[1], "psnr", "b  Paired PSNR difference"), (axs[2], "de2000", "c  Paired CIEDE2000 difference")):
        diff = P[f"metric|sun_v2|{m}"] - P[f"metric|baseline|{m}"]
        ax.hist(np.clip(diff, -6, 6), np.linspace(-6, 6, 49), color=C["sun_v2"], rwidth=0.9)
        ax.axvline(0, color=INK2, lw=1)
        q = d[m]
        ax.text(0.03, 0.97, f"mean {q['mean']:+.2f}\n95 % CI [{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}]\n"
                            f"v2 better on {q['share_x_better'] * 100:.0f} %", transform=ax.transAxes, va="top",
                fontsize=9.5, color=INK2, bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=0.6))
        ax.set_xlabel(f"v2 − BetaNet, {'PSNR [dB]' if m == 'psnr' else 'CIEDE2000 (negative = v2 better)'}; clipped at ±6")
        ax.set_ylabel("test scenes"), ax.set_title(t)
    fig.tight_layout()
    save(fig, D2, "b16_paired")


def fig_b16_by_condition(S, P, Jn):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(10)
    mean, lo, hi = [], [], []
    for t in JER:  # per Jerlov type, paired over the same 200 rooms: recomputed from the stored per-type means
        mean.append(S["jerlov"][t]["psnr|sun_v2"] - S["jerlov"][t]["psnr|baseline"])
    axs[0].bar(x, mean, 0.6, color=[C["sun_v2"] if v >= 0 else C["baseline"] for v in mean])
    label_bars(axs[0], x, mean, "{:+.2f}", fs=8.5)
    axs[0].axhline(0, color=INK2, lw=1)
    axs[0].set_xticks(x, JER), axs[0].set_xlabel("Jerlov water type (200 test rooms each)")
    axs[0].set_ylabel("mean PSNR, v2 − BetaNet [dB]")
    axs[0].set_title("a  By water type (green: v2 better, blue: BetaNet better)")
    ygrid(axs[0]), axs[0].margins(y=0.25)
    dt = P["true|d_s"][:, 0]
    diff = P["metric|sun_v2|psnr"] - P["metric|baseline|psnr"]
    ms = [(dt >= a) & (dt < b) for a, b in DBINS]
    mu = [diff[k].mean() for k in ms]
    ci = np.array([boot_ci(diff[k]) for k in ms])
    xb = np.arange(4)
    axs[1].bar(xb, mu, 0.6, color=[C["sun_v2"] if v >= 0 else C["baseline"] for v in mu])
    axs[1].errorbar(xb, mu, [np.subtract(mu, ci[:, 0]), np.subtract(ci[:, 1], mu)], fmt="none", ecolor=INK2, capsize=4, lw=1)
    axs[1].axhline(0, color=INK2, lw=1)
    axs[1].set_xticks(xb, [f"{a}–{int(b)} m\n(n={k.sum()})" for (a, b), k in zip(DBINS, ms)])
    axs[1].set_xlabel("True depth below the surface d_s"), axs[1].set_ylabel("mean PSNR, v2 − BetaNet [dB]")
    axs[1].set_title("b  By depth (bars: mean, whiskers: 95 % CI)"), ygrid(axs[1])
    fig.tight_layout()
    save(fig, D2, "b16_by_condition")


def fig_b16_real(R):
    pi = R["uieb"]["per_image"]
    b, s = np.array(pi["baseline"]["de2000"]), np.array(pi["sun_v2"]["de2000"])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    lim = [min(b.min(), s.min()) - 1, max(b.max(), s.max()) + 1]
    axs[0].scatter(b, s, s=16, color=C["sun_v2"])
    axs[0].plot(lim, lim, color=INK2, ls="--", lw=1.2)
    axs[0].set_xlim(lim), axs[0].set_ylim(lim), axs[0].set_aspect("equal")
    axs[0].set_xlabel("BetaNet (12) CIEDE2000"), axs[0].set_ylabel("v2 (16) CIEDE2000")
    ci = boot_ci(s - b)
    axs[0].set_title("a  UIEB, own E, per photo")
    axs[0].text(0.03, 0.97, f"below the line: v2 better\nmean v2 − BetaNet {np.mean(s - b):+.2f} "
                            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]\nv2 better on {(s < b).mean() * 100:.0f} %",
                transform=axs[0].transAxes, va="top", fontsize=9.5, color=INK2)
    fig_real_bars(axs[1], R, ["baseline", "sun_v2"])
    fig.tight_layout()
    save(fig, D2, "b16_real")


def fig_real_bars(ax, R, models):
    U = R["uieb"]["summary"]
    groups = [("", "own E"), (" + gray-world", "gray-world"), (" + Day 1b", "Day 1b")]
    grouped(ax, models, {m: [U[m + g]["de2000"] for g, _ in groups] for m in models}, [g[1] for g in groups],
            "b  UIEB mean CIEDE2000 by restoration", fmt="{:.2f}")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    ax.margins(y=0.3)


def table_b16(S, R):
    table_3m(S, R, ["baseline", "sun_v2"], D2, "baseline_vs_16_table")
    q = S["paired"]["sun_v2 - baseline"]
    add = ["", "Paired over the same 654 test scenes (v2 − BetaNet, 95 % bootstrap interval over scenes):", ""]
    for m, t in (("psnr", "PSNR [dB]"), ("ssim", "SSIM"), ("de2000", "CIEDE2000")):
        add.append(f"- {t}: {q[m]['mean']:+.3f} [{q[m]['ci95'][0]:+.3f}, {q[m]['ci95'][1]:+.3f}], v2 better on "
                   f"{q[m]['share_x_better'] * 100:.0f} % of scenes")
    with open(D2 / "baseline_vs_16_table.md", "a") as f:
        f.write("\n".join(add) + "\n")


# ----------------------------------------------------------------------------------------------- 16-output model
def fig_v2_training():
    logs = {"baseline": HERE / "runs/baseline_v2data/log.json", "sun_v1": HERE / "runs/sun_v1_v2data/log.json",
            "sun_v2": HERE / "runs/sun_v2/log.json"}
    L = {m: json.load(open(p)) for m, p in logs.items() if p.exists()}
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
    for m, log in L.items():
        ep = [r["epoch"] for r in log]
        axs[0].plot(ep, [r["val_psnr"] for r in log], color=C[m], label=LBL[m])
        axs[1].plot(ep, [r["val_coef"] for r in log], color=C[m], label=LBL[m])
        if "val_depth_abs" in log[0]:
            axs[2].plot(ep, [r["val_depth_abs"] for r in log], color=C[m], label=LBL[m])
    v2 = L["sun_v2"]
    ep = [r["epoch"] for r in v2]
    axs[3].plot(ep, [r["val_Kd_logl1"] for r in v2], color=C["sun_v2"], label="K_d  |log error|")
    axs[3].plot(ep, [r["val_E0_angle"] / 10 for r in v2], color=C["sun_v2"], ls="--", label="E0 colour error [°] / 10")
    axs[3].plot(ep, [r["train_Kd"] for r in v2], color=MUTED, lw=1.2, label="K_d |log error|, training")
    for ax, t, yl in zip(axs, ("a  Validation PSNR", "b  Validation coefficient loss", "c  Validation depth error",
                               "d  v2: K_d and E0 errors"),
                         ("PSNR [dB]", "coef. loss (lower = better)", "mean |d̂_s − d_s| [m]", "error")):
        ax.set_title(t), ax.set_xlabel("epoch"), ax.set_ylabel(yl)
    axs[1].set_yscale("log")
    axs[0].legend(fontsize=8.5, loc="lower right"), axs[3].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, D16, "v2_training_curves")


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def fig_v2_kd(P):
    kp, kt = P["pred|sun_v2|K_d"], P["true|K_d"]
    k1 = 0.8 * P["pred|sun_v1|beta_D"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    for c, (ax, ch) in enumerate(zip(axs, "RGB")):
        lim = [min(kt[:, c].min(), kp[:, c].min()) * 0.8, max(kt[:, c].max(), kp[:, c].max()) * 1.2]
        ax.scatter(kt[:, c], k1[:, c], s=9, color=C["sun_v1"], alpha=0.35, lw=0, label="v1: 0.8 β̂_D (implied)")
        ax.scatter(kt[:, c], kp[:, c], s=11, color=C["sun_v2"], lw=0.3, ec=SURFACE, label="v2: predicted K_d")
        ax.plot(lim, lim, color=INK2, ls="--", lw=1.2)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
        lp, lt, l1 = np.log(kp[:, c]), np.log(kt[:, c]), np.log(k1[:, c])
        ax.text(0.03, 0.97, f"v2: R² (log) {r2(lp, lt):.2f}, median rel. err {np.median(np.abs(kp[:, c] / kt[:, c] - 1)) * 100:.0f} %\n"
                            f"v1: R² (log) {r2(l1, lt):.2f}, median rel. err {np.median(np.abs(k1[:, c] / kt[:, c] - 1)) * 100:.0f} %",
                transform=ax.transAxes, va="top", fontsize=9, color=INK2,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=0.6))
        ax.set_xlabel(f"true K_d,{ch} [1/m]"), ax.set_ylabel(f"predicted K_d,{ch} [1/m]")
        ax.set_title(f"{'abc'[c]}  {ch} channel")
    axs[2].legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, D16, "v2_kd_scatter")


def fig_v2_e0(P):
    ep, et, cct = P["pred|sun_v2|E0"], P["true|E0"], P["true|cct"][:, 0]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (i, j, lab) in zip(axs[:2], ((0, 2, "log(E0_R / E0_B)"), (1, 2, "log(E0_G / E0_B)"))):
        t, p = np.log(et[:, i] / et[:, j]), np.log(ep[:, i] / ep[:, j])
        sc = ax.scatter(t, p, c=cct, cmap="coolwarm_r", s=11, lw=0)
        lim = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
        ax.plot(lim, lim, color=INK2, ls="--", lw=1.2)
        ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
        ax.set_xlabel(f"true {lab}"), ax.set_ylabel(f"predicted {lab}")
        ax.text(0.03, 0.97, f"R² {r2(p, t):.2f}   Spearman {spearmanr(p, t).statistic:.2f}", transform=ax.transAxes,
                va="top", fontsize=9.5, color=INK2)
    axs[0].set_title("a  E0, red / blue"), axs[1].set_title("b  E0, green / blue")
    cb = fig.colorbar(sc, ax=axs[:2], shrink=0.85, pad=0.02)
    cb.set_label("true colour temperature [K]", color=INK2), cb.outline.set_visible(False)
    a2 = angle(ep, et)
    axs[2].hist(a2, np.linspace(0, max(15, np.percentile(a2, 99)), 40), color=C["sun_v2"], rwidth=0.9, label="v2 E0")
    a1 = angle(np.ones_like(et), et)
    axs[2].hist(a1, np.linspace(0, max(15, np.percentile(a2, 99)), 40), histtype="step", color=C["sun_v1"], lw=2,
                label="v1 (fixed white)")
    axs[2].set_xlabel("E0 colour error [°]"), axs[2].set_ylabel("test scenes"), axs[2].legend(fontsize=9)
    axs[2].set_title(f"c  E0 error: v2 mean {a2.mean():.2f}°, white {a1.mean():.2f}°")
    save(fig, D16, "v2_e0")


def fig_v2_depth(P):
    dt = P["true|d_s"][:, 0]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    for m in ("sun_v1", "sun_v2"):
        dp = P[f"pred|{m}|d_s"][:, 0]
        axs[0].scatter(dt, dp, s=10, color=C[m], lw=0.3, ec=SURFACE, alpha=0.45 if m == "sun_v1" else 0.9,
                       label=f"{LBL[m]}: MAE {np.abs(dp - dt).mean():.2f} m, R² {r2(dp, dt):.2f}")
        axs[1].hist(np.clip(dp - dt, -8, 8), np.linspace(-8, 8, 49), histtype="step" if m == "sun_v1" else "bar",
                    color=C[m], lw=2, rwidth=0.9, label=LBL[m])
    axs[0].plot([0, 15], [0, 15], color=INK2, ls="--", lw=1.2)
    axs[0].set_xlim(0, 15), axs[0].set_ylim(0, 15), axs[0].set_aspect("equal")
    axs[0].set_xlabel("true depth below the surface d_s [m]"), axs[0].set_ylabel("predicted d̂_s [m]")
    axs[0].legend(fontsize=8.5, loc="lower right"), axs[0].set_title("a  Predicted vs true depth")
    axs[1].axvline(0, color=INK2, lw=1)
    axs[1].set_xlabel("d̂_s − d_s [m]"), axs[1].set_ylabel("test scenes"), axs[1].legend(fontsize=8.5)
    axs[1].set_title("b  Depth error distribution")
    fig.tight_layout()
    save(fig, D16, "v2_depth")


def fig_v2_sunlight(P):
    et = P["true|E"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    for c, ch in enumerate("RGB"):
        ax = axs[c]
        ax.scatter(et[:, c], P["pred|baseline|E"][:, c], s=8, color=C["baseline"], alpha=0.35, lw=0, label="BetaNet (predicted E)")
        ax.scatter(et[:, c], P["pred|sun_v2|E"][:, c], s=9, color=C["sun_v2"], lw=0, label="v2 (computed E)")
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1.2)
        ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02), ax.set_aspect("equal")
        ax.set_xlabel(f"true E_{ch}"), ax.set_ylabel(f"estimated E_{ch}"), ax.set_title(f"{'abc'[c]}  {ch} channel")
    axs[0].legend(fontsize=8.5, loc="upper left")
    bins = np.linspace(0, 15, 46)
    for m in M3:
        a = angle(P[f"pred|{m}|E"], et)
        axs[3].hist(np.clip(a, 0, 15), bins, histtype="step", color=C[m], lw=2, label=f"{SHORT[m].replace(chr(10), ' ')}: mean {a.mean():.2f}°")
    axs[3].set_xlabel("sunlight colour error [°] (clipped at 15°)"), axs[3].set_ylabel("test scenes")
    axs[3].legend(fontsize=8.5)
    axs[3].set_title("d  Sunlight colour error")
    fig.tight_layout()
    save(fig, D16, "v2_sunlight")


def fig_v2_error_vs_image(P):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
    base = P["metric|sun_v2|psnr"]
    errs = [("K_d", np.abs(np.log(P["pred|sun_v2|K_d"] / P["true|K_d"])).mean(1), "mean |log K̂_d − log K_d|", "true K_d"),
            ("E0", angle(P["pred|sun_v2|E0"], P["true|E0"]), "E0 colour error [°]", "true E0"),
            ("d_s", np.abs(P["pred|sun_v2|d_s"][:, 0] - P["true|d_s"][:, 0]), "|d̂_s − d_s| [m]", "true d_s")]
    for ax, (k, e, xl, key), t in zip(axs, errs, "abc"):
        g = P[f"metric|sun_v2 | {key}|psnr"] - base
        ax.scatter(e, g, s=10, color=C["sun_v2"], lw=0.3, ec=SURFACE)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel(xl), ax.set_ylabel(f"PSNR gained with the {key} [dB]")
        ax.set_title(f"{t}  {k} error vs image cost")
        ax.text(0.97, 0.97, f"Spearman ρ = {spearmanr(e, g).statistic:.2f}\nmean gain {g.mean():+.2f} dB",
                transform=ax.transAxes, va="top", ha="right", fontsize=9.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=0.6))
    fig.tight_layout()
    save(fig, D16, "v2_error_vs_image")


# ----------------------------------------------------------------------------------------------- additional
def fig_individual(P):
    cols = [("beta_D", "β_D (mean of R, G, B) [1/m]"), ("B_inf", "B∞ (mean)"), ("E", "E (all channels)"),
            ("d_s", "d_s [m]"), ("K_d", "K_d (all channels) [1/m]")]
    fig, axs = plt.subplots(3, 5, figsize=(19, 11))
    for r, m in enumerate(M3):
        for c, (k, t) in enumerate(cols):
            ax = axs[r, c]
            tv = P[f"true|{k}"]
            pv = P.get(f"pred|{m}|{k}")
            if k == "K_d":
                pv = implied(P, m)[0]
            if pv is None:
                no_axes(ax), ax.text(0.5, 0.5, "not an output", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                continue
            if k in ("beta_D", "B_inf"):
                tv, pv = tv.mean(1), pv.mean(1)
            tv, pv = tv.ravel(), pv.ravel()
            ax.scatter(tv, pv, s=5, color=C[m], lw=0, alpha=0.6)
            lim = [min(tv.min(), pv.min()), max(tv.max(), pv.max())]
            ax.plot(lim, lim, color=INK2, ls="--", lw=1)
            if k in ("beta_D", "K_d"):
                ax.set_xscale("log"), ax.set_yscale("log")
            ax.text(0.04, 0.96, f"R² {r2(np.log(pv) if k in ('beta_D', 'K_d') else pv, np.log(tv) if k in ('beta_D', 'K_d') else tv):.2f}",
                    transform=ax.transAxes, va="top", fontsize=9, color=INK2)
            if r == 2:
                ax.set_xlabel(f"true {t}")
            if c == 0:
                ax.set_ylabel(f"{LBL[m]}\npredicted", fontsize=10)
            if r == 0:
                ax.set_title(t, fontsize=11)
    axs[1, 4].set_title("K_d (v1: 0.8 β̂_D)", fontsize=10)
    fig.tight_layout()
    save(fig, DF, "individual_parameter_scatter")


def fig_depth_meaning(P):
    """Does an independent K_d make the predicted depth more physical? Compare the two depth models directly."""
    dt = P["true|d_s"][:, 0]
    kt = P["true|K_d"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for m in ("sun_v1", "sun_v2"):
        dp = P[f"pred|{m}|d_s"][:, 0]
        kd, e0 = implied(P, m)
        # optical depth of the downwelling path, the quantity E actually constrains: tau_c = K_d,c d_s
        axs[0].scatter((kt * dt[:, None]).mean(1), (kd * dp[:, None]).mean(1), s=9, color=C[m], lw=0, alpha=0.6,
                       label=f"{SHORT[m].replace(chr(10), ' ')}")
        e = dp - dt
        axs[1].scatter(np.log(kd / kt).mean(1), e, s=9, color=C[m], lw=0, alpha=0.6,
                       label=f"{SHORT[m].replace(chr(10), ' ')}: Spearman {spearmanr(np.log(kd / kt).mean(1), e).statistic:.2f}")
        q = [np.abs(e[(dt >= a) & (dt < b)]).mean() / ((a + b) / 2) * 100 for a, b in DBINS]
        axs[2].plot(range(4), q, marker=MK[m], color=C[m], label=SHORT[m].replace("\n", " "))
    lim = [0, max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])]
    axs[0].plot(lim, lim, color=INK2, ls="--", lw=1)
    axs[0].set_xlabel("true mean K_d·d_s"), axs[0].set_ylabel("model's mean K̂_d·d̂_s"), axs[0].legend(fontsize=9)
    axs[0].set_title("a  Optical depth of the sunlight path")
    axs[1].axhline(0, color=INK2, lw=1), axs[1].axvline(0, color=INK2, lw=1)
    axs[1].set_xlabel("mean log(K̂_d / K_d)  (K_d over- or under-estimated)"), axs[1].set_ylabel("d̂_s − d_s [m]")
    axs[1].legend(fontsize=9), axs[1].set_title("b  Do K_d errors push the depth the other way?")
    axs[2].set_xticks(range(4), [f"{a}–{int(b)} m" for a, b in DBINS]), axs[2].set_ylabel("MAE / bin centre [%]")
    axs[2].set_xlabel("true d_s"), axs[2].legend(fontsize=9), axs[2].set_title("c  Relative depth error by depth")
    fig.tight_layout()
    save(fig, DF, "depth_meaning")


def fig_real_unvalidated(R):
    import torch
    sys.path.insert(0, str(HERE))
    from water_v2 import daylight_rgb
    loc = daylight_rgb(torch.linspace(4000, 10000, 60)).numpy()
    co = R["uieb"]["coefs"]
    fr = R["seathru"]["frames"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    e0u = np.array([c["E0"] for c in co["sun_v2"]])
    e0s = np.array([fr[f]["sun_v2"]["E0"] for f in fr])
    axs[0].plot(np.log(loc[:, 0] / loc[:, 2]), np.log(loc[:, 1] / loc[:, 2]), color=MUTED, lw=3,
                label="CIE daylight locus 4000–10000 K (training range)")
    axs[0].scatter(np.log(e0u[:, 0] / e0u[:, 2]), np.log(e0u[:, 1] / e0u[:, 2]), s=14, color=C["sun_v2"], alpha=0.7, label="UIEB photos")
    axs[0].scatter(np.log(e0s[:, 0] / e0s[:, 2]), np.log(e0s[:, 1] / e0s[:, 2]), s=30, color=INK, marker="x", label="Sea-thru frames")
    axs[0].set_xlabel("log(E0_R / E0_B)"), axs[0].set_ylabel("log(E0_G / E0_B)"), axs[0].legend(fontsize=8.5)
    axs[0].set_title("a  v2 predicted surface daylight E0")
    kdu = np.array([c["K_d"] for c in co["sun_v2"]])
    kds = np.array([fr[f]["sun_v2"]["K_d"] for f in fr])
    for c, ch in enumerate("RGB"):
        r = np.random.default_rng(c)
        axs[1].scatter(c * 2 + r.uniform(-0.25, 0.25, len(kdu)), kdu[:, c], s=12, color=CH[ch], alpha=0.6)
        axs[1].scatter(c * 2 + 1 + r.uniform(-0.2, 0.2, len(kds)), kds[:, c], s=20, color=CH[ch], marker="x")
    axs[1].set_yscale("log"), axs[1].set_xticks(range(6), ["R\nUIEB", "R\nSea-thru", "G\nUIEB", "G\nSea-thru", "B\nUIEB", "B\nSea-thru"])
    axs[1].set_ylabel("predicted K_d [1/m]"), axs[1].set_title("b  v2 predicted K_d")
    fig_real_depths(axs[2], R)
    fig.suptitle("Real photos: network predictions with NO ground truth (unvalidated)", x=0.01, y=1.04, ha="left",
                 color=INK2, fontsize=11)
    fig.tight_layout()
    save(fig, DF, "real_unvalidated_predictions")


def fig_real_training_water(R):
    """Does training with varying surface daylight (v2 water) help on real photos? Same architecture, old vs new
    training water, scored on the same real photos (real photos do not depend on the synthetic generator)."""
    U, ch = R["uieb"]["summary"], R["seathru"]["chart"]["D1/T_S03272"]
    pairs = [("baseline", "ref: baseline (old water)"), ("sun_v1", "ref: sun_v1 (old water)")]
    pairs = [(n, o) for n, o in pairs if n in U and o in U]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, t in ((axs[0], "", "a  UIEB, own E: CIEDE2000"), (axs[1], " + Day 1b", "b  UIEB, Day 1b (E unused)")):
        x = np.arange(len(pairs))
        old = [U[o + key]["de2000"] for _, o in pairs]
        new = [U[n + key]["de2000"] for n, _ in pairs]
        ax.bar(x - 0.2, old, 0.38, color=MUTED, label="trained on old water (white sun)")
        ax.bar(x + 0.2, new, 0.38, color=[C[n] for n, _ in pairs], label="trained on v2 water (daylight E0)")
        label_bars(ax, x - 0.2, old, "{:.2f}", fs=8.5), label_bars(ax, x + 0.2, new, "{:.2f}", fs=8.5)
        ax.set_xticks(x, [SHORT[n].replace("\n", " ") for n, _ in pairs]), ax.set_title(t + "\n(lower = better)", fontsize=11)
        ax.margins(y=0.2), ygrid(ax)
    x = np.arange(len(pairs))
    old, new = [ch[o] for _, o in pairs], [ch[n] for n, _ in pairs]
    axs[2].bar(x - 0.2, old, 0.38, color=MUTED), axs[2].bar(x + 0.2, new, 0.38, color=[C[n] for n, _ in pairs])
    label_bars(axs[2], x - 0.2, old, "{:.1f}°", fs=8.5), label_bars(axs[2], x + 0.2, new, "{:.1f}°", fs=8.5)
    axs[2].set_xticks(x, [SHORT[n].replace("\n", " ") for n, _ in pairs])
    axs[2].set_title("c  Sea-thru chart, own E\n(lower = better)", fontsize=11), axs[2].margins(y=0.2), ygrid(axs[2])
    fig.suptitle("Effect of the training water on real photos (same architecture, same real test photos)", x=0.01,
                 y=1.04, ha="left", color=INK2, fontsize=11)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=MUTED)] + [plt.Rectangle((0, 0), 1, 1, color=C[n]) for n, _ in pairs],
               ["trained on the old water (white sun, K_d = κ β_D)"] + [f"{SHORT[n].replace(chr(10), ' ')} trained on the v2 water"
                                                                      for n, _ in pairs],
               loc="upper right", ncol=3, bbox_to_anchor=(1, 1.0), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, DF, "real_generalisation_training_water")


def fig_real_depths(ax, R):
    fr = R["seathru"]["frames"]
    dives = sorted({f.split("/")[0] for f in fr})
    for j, m in enumerate(("sun_v1", "sun_v2")):
        for k, dv in enumerate(dives):
            ds = [fr[f][m]["d_s"][0] for f in fr if f.startswith(dv + "/")]
            ax.scatter(np.full(len(ds), k) + (j - 0.5) * 0.3, ds, color=C[m], marker=MK[m], s=24,
                       label=SHORT[m].replace("\n", " ") if k == 0 else None)
    ax.set_xticks(range(len(dives)), dives), ax.set_ylim(0, 15.5), ax.set_ylabel("predicted d̂_s [m]")
    ax.set_xlabel("Sea-thru dive"), ax.legend(fontsize=9), ax.set_title("c  Predicted depth per Sea-thru dive")


# ----------------------------------------------------------------------------------------------- galleries
def gallery(G, rows, cols, title_rows, name, metric=None, fs=11):
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.75 * len(cols), 2.2 * len(rows)), squeeze=False)
    for r, key in enumerate(rows):
        for c, (k, lab) in enumerate(cols):
            ax = axs[r, c]
            ax.imshow(np.clip(G[f"{key}|{k}"].astype(np.float32), 0, 1))
            no_axes(ax)
            if r == 0:
                ax.set_title(lab, fontsize=fs)
            if metric and k in metric:
                ax.text(0.02, 0.04, metric[k](key), transform=ax.transAxes, fontsize=8.5, color="white",
                        bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
        if title_rows:
            axs[r, 0].text(-0.04, 0.5, title_rows[r], transform=axs[r, 0].transAxes, rotation=90, ha="right",
                           va="center", fontsize=9.5, color=INK)
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    save(fig, DQ, name)


def galleries(S, P, R):
    G = dict(np.load(D16 / "synth_gallery.npz"))
    idx = S["gallery_idx"][::2][:6] if len(S["gallery_idx"]) >= 6 else S["gallery_idx"]
    m = lambda mm: (lambda key: f"{P[f'metric|{mm}|psnr'][int(key)]:.1f} dB  ΔE {P[f'metric|{mm}|de2000'][int(key)]:.1f}")
    rows_lab = [f"d_s {P['true|d_s'][i, 0]:.1f} m, β_D {P['true|beta_D'][i].mean():.2f}" for i in idx]
    cols3 = [("raw", "Underwater input"), ("baseline", "BetaNet (12)"), ("sun_v1", "v1 (10)"), ("sun_v2", "v2 (16)"),
             ("clean", "Ground truth")]
    gallery(G, [str(i) for i in idx], cols3, rows_lab, "gallery_3m_depth_turbidity",
            {k: m(k) for k in ("baseline", "sun_v1", "sun_v2")})
    gallery(G, [str(i) for i in idx], [cols3[0], cols3[1], cols3[3], cols3[4]], rows_lab, "gallery_b16",
            {k: m(k) for k in ("baseline", "sun_v2")})
    jg = S["jerlov_gallery"]
    Jn = dict(np.load(D16 / "synth_jerlov.npz"))
    jrows = [f"jerlov_{t}|{i}" for t, i in jg.items()]
    jlab = [f"Jerlov {t}, d_s {Jn[f'{t}|true|d_s'][i, 0]:.1f} m" for t, i in jg.items()]
    gallery(G, jrows, cols3, jlab, "gallery_3m_jerlov_types")
    if R is None:
        return
    O = dict(np.load(DQ / "real_outputs.npz"))
    keys = sorted({k.split("|")[0] for k in O if k.startswith("uieb")}, key=lambda s: int(s[4:]))
    pi = R["uieb"]["per_image"]
    de = lambda mm: (lambda key: f"ΔE {pi[mm]['de2000'][int(key[4:])]:.1f}")
    colsr = [("raw", "Real photo (UIEB)"), ("baseline", "BetaNet (12)"), ("sun_v1", "v1 (10)"), ("sun_v2", "v2 (16)"),
             ("reference", "UIEB reference")]
    colsr = [c for c in colsr if f"{keys[0]}|{c[0]}" in O]
    gallery(O, keys, colsr, None, "gallery_uieb_3m", {k: de(k) for k in ("baseline", "sun_v1", "sun_v2") if k in pi})
    gallery(O, keys, [c for c in colsr if c[0] != "sun_v1"], None, "gallery_uieb_b16",
            {k: de(k) for k in ("baseline", "sun_v2")})


def main():
    S, P, Jn, R = load()
    fig_3m_image_quality(S)
    fig_3m_physical(S, P)
    fig_3m_jerlov(S)
    fig_3m_depth_bins(P)
    fig_3m_attribution(S)
    fig_3m_image_quality(S, ["baseline", "sun_v2"], D2, "b16_image_quality")
    fig_b16_paired(S, P)
    fig_b16_by_condition(S, P, Jn)
    fig_v2_training()
    fig_v2_kd(P)
    fig_v2_e0(P)
    fig_v2_depth(P)
    fig_v2_sunlight(P)
    fig_v2_error_vs_image(P)
    fig_individual(P)
    fig_depth_meaning(P)
    if R:
        fig_3m_real(R)
        fig_b16_real(R)
        fig_real_unvalidated(R)
        fig_real_training_water(R)
    table_3m(S, R)
    table_b16(S, R)
    galleries(S, P, R)


if __name__ == "__main__":
    main()
