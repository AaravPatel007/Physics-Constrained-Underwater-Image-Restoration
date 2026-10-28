"""Before / after fine-tuning figures (run after finetune.py, eval_finetune.py, select_ft.py and
eval_real.py --ckpt <name> --tag _<name> for every fine-tuned model).

  python src/fig_finetune.py <selected> <collapsed> [other fine-tuned models ...]
  e.g.   python src/fig_finetune.py ft_nodive ft_seathru ft_teacher ft_seathru_noanchor ft_uieb ft_uieb_ref

fig15_ft_real.png      real Sea-thru frames: before, collapsed and selected fine-tune, with grey-patch errors
fig16_ft_metrics.png   chart error, UIEB, synthetic check, between-dive spread, held-out-dive physics losses
fig17_ft_curves.png    fine-tuning curves (held-out real losses, between-dive spread, synthetic accuracy vs step)
fig18_ft_sites.png     beta_D per dive: before / collapsed / selected (water-type read-out)
fig19_ft_uieb.png      UIEB before vs after
fig20_ft_tradeoff.png  teacher-weight sweep: chart / UIEB / held-out loss / dive spread
Writes results/finetune_figure_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

from fig_results import srgb
from vizstyle import CH, CH_MARK, INK2, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures" / "2_original_finetune"
BEFORE, BASE, BAD = "#86b6ef", "#c3c2b7", SERIES[1]
LABEL = {"beta_rgbd": "BetaNet before (synthetic only)",
         "ft_seathru": "Fine-tune v1: all losses (collapsed)",
         "ft_seathru_noanchor": "v1 without synthetic anchor",
         "ft_nodive": "Fine-tune: no same-dive loss",
         "ft_teacher": "+ same-dive loss + teacher 1.0",
         "ft_nodive_t01": "No same-dive loss + teacher 0.1",
         "ft_nodive_t03": "No same-dive loss + teacher 0.3",
         "ft_uieb": "v1 + UIEB (physics losses)",
         "ft_uieb_ref": "v1 + UIEB (+ reference loss)"}
GW, OWN = "BetaNet + gray-world [SfM range]", "BetaNet [SfM range]"
DIVES = ["D1", "D2", "D3", "D4", "D5"]


def label(n):
    return LABEL.get(n, n)


def colour(n, sel, collapsed):
    return BEFORE if n == "beta_rgbd" else SERIES[0] if n == sel else BAD if n == collapsed else "#9fb7d6"


def between(F, n):
    """std over dives of the dive-mean log beta_D, averaged over R, G, B (0 = every dive gets the same water)."""
    return float(np.mean(np.std([np.log(F[n]["sites"][d]["beta_D"]) for d in DIVES], 0)))


def real_grid(sel, collapsed, R):
    o = {n: np.load(RES / ("real_seathru_outputs.npz" if n == "beta_rgbd" else f"real_seathru_outputs_{n}.npz"))
         for n in ("beta_rgbd", collapsed, sel)}
    frames = [("D1/T_S03272", "chart frame (held out)"), ("D3/T_S04910", "dive D3 (never trained on)"),
              ("D4/RGT_0209", "turbid canyon (held-out frame)"), ("D5/LFT_3374", "dive D5 (held-out frame)")]
    cols = [("beta_rgbd", "raw", "Raw (linear, exposure only)"),
            ("beta_rgbd", "Sea-thru (classical, SfM range)", "Sea-thru classical"),
            ("beta_rgbd", GW, "BetaNet before\n(+ gray-world)"),
            (collapsed, GW, label(collapsed) + "\n(+ gray-world)"),
            (sel, GW, label(sel) + "\n(+ gray-world)"),
            (sel, OWN, label(sel) + "\n(own sunlight colour, no gray-world)")]
    fig, axs = plt.subplots(len(frames), len(cols), figsize=(3.9 * len(cols), 2.9 * len(frames)))
    for i, (fr, desc) in enumerate(frames):
        for j, (n, m, title) in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(srgb(o[n][f"{fr}|{m}"]))
            no_axes(ax)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.text(-0.04, 0.5, f"{fr}\n{desc}", transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=9, color=INK2)
            chart = R[n]["seathru"]["chart"].get(fr)
            if chart and m in chart:
                ax.text(0.02, 0.03, f"grey error {chart[m]:.1f}°", transform=ax.transAxes, fontsize=10,
                        color="white", bbox=dict(fc="#0b0b0b", alpha=0.6, lw=0, pad=2))
    fig.subplots_adjust(left=0.04, right=0.995, wspace=0.03, hspace=0.08)
    fig.suptitle("Real Sea-thru frames: before vs after fine-tuning on real dives (SfM range)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG / "fig15_ft_real.png", dpi=110)
    plt.close(fig)


def bars(ax, names, vals, colors, fmt="{:.1f}", lower_better=True):
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.68)
    ax.set_yticks(y, names, fontsize=9)
    for yy, v in zip(y, vals):
        ax.text(v, yy, " " + fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel(("lower" if lower_better else "higher") + " is better", fontsize=9)


def metrics(models, sel, collapsed, R, F):
    out = {}
    allm = ["beta_rgbd"] + models
    col = {n: colour(n, sel, collapsed) for n in allm}
    fig, axs = plt.subplots(2, 3, figsize=(23, 11))

    ch = lambda n, m: float(np.mean([v[m] for v in R[n]["seathru"]["chart"].values()]))
    rows = [("Raw", ch("beta_rgbd", "raw"), BASE), ("Gray-world", ch("beta_rgbd", "gray-world"), BASE),
            ("Sea-thru classical", ch("beta_rgbd", "Sea-thru (classical, SfM range)"), BASE)]
    rows += [(label(n), ch(n, GW), col[n]) for n in allm]
    rows += [("UNet black box", ch("beta_rgbd", "UNet (black box, SfM range)"), BASE)]
    out["chart_mean_deg"] = {r[0]: r[1] for r in rows}
    bars(axs[0, 0], *zip(*rows))
    axs[0, 0].set_title("Grey-patch error on the colour chart (°)\nmean of 2 D1 frames, BetaNet + gray-world", fontsize=11)

    for ax, k, t, lb in [(axs[0, 1], "psnr", "UIEB PSNR vs reference (dB)", False),
                         (axs[0, 2], "de2000", "UIEB colour error ΔE₀₀", True)]:
        rows = [("Raw", R["beta_rgbd"]["uieb"]["summary"]["raw"][k], BASE),
                ("Gray-world", R["beta_rgbd"]["uieb"]["summary"]["gray-world"][k], BASE)]
        rows += [(label(n), R[n]["uieb"]["summary"]["BetaNet + gray-world"][k], col[n]) for n in allm]
        rows += [("UNet black box", R["beta_rgbd"]["uieb"]["summary"]["UNet (black box)"][k], BASE)]
        out[f"uieb_{k}"] = {r[0]: r[1] for r in rows}
        bars(ax, *zip(*rows), lower_better=lb)
        ax.set_title(t + "\n90 validation images, BetaNet + gray-world", fontsize=11)

    rows = [(label(n), 100 * float(np.mean(F[n]["synth"]["coef_median_rel_err"]["beta_D"])), col[n]) for n in allm]
    out["synth_beta_D_err_pct"] = {r[0]: r[1] for r in rows}
    bars(axs[1, 0], *zip(*rows))
    axs[1, 0].set_title("Synthetic test (654 rooms): β_D median error (%)\nmust not get worse after real fine-tuning",
                        fontsize=11)

    rows = [(label(n), between(F, n), col[n]) for n in allm]
    out["between_dive_spread"] = {r[0]: r[1] for r in rows}
    bars(axs[1, 1], *zip(*rows), fmt="{:.2f}", lower_better=False)
    axs[1, 1].axvline(0.5 * between(F, "beta_rgbd"), color=INK2, lw=0.8, ls="--")
    axs[1, 1].set_title("Are the 5 dives still told apart?\nspread of per-dive log β_D (dashed = selection floor)",
                        fontsize=11)

    ax = axs[1, 2]
    keys = [("dark", "backscatter vs\ndarkest pixels"), ("range", "colour drift\nwith distance"), ("gray", "gray-world\nresidual")]
    show = ["beta_rgbd", collapsed, sel]
    w = 0.8 / len(show)
    for i, n in enumerate(show):
        rel = [F[n]["heldout_D3"][k] / F["beta_rgbd"]["heldout_D3"][k] for k, _ in keys]
        ax.bar(np.arange(3) + (i - (len(show) - 1) / 2) * w, rel, w, color=col[n], label=label(n))
        out.setdefault("heldout_D3_rel_to_before", {})[label(n)] = dict(zip([k for k, _ in keys], rel))
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(3), [t for _, t in keys], fontsize=9)
    ax.set_ylabel("loss relative to before (1 = no change)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Held-out dive D3 (never trained on):\nphysics-consistency losses, lower is better", fontsize=11)
    fig.subplots_adjust(wspace=0.75, hspace=0.42, left=0.11, right=0.98)
    fig.suptitle("Before vs after fine-tuning on real dives", x=0.01, ha="left", fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(FIG / "fig16_ft_metrics.png")
    plt.close(fig)
    return out


def curves(runs, sel, collapsed):
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.6))
    for n in runs:
        L = json.load(open(RES / n / "log.json"))
        s = [r["step"] for r in L]
        c = colour(n, sel, collapsed)
        lw = 2.5 if n in (sel, collapsed) else 1.2
        axs[0].plot(s, [(r["real_test"]["range"] + r["real_test"]["gray"]) / (L[0]["real_test"]["range"] + L[0]["real_test"]["gray"])
                        for r in L], color=c, lw=lw, label=label(n))
        sp = [float(np.std(np.log([r["real_test"]["beta_D_mean"][d][0] for d in DIVES]))) for r in L]
        axs[1].plot(s, sp, color=c, lw=lw, label=label(n))
        axs[2].plot(s, [r["synth"]["psnr"] for r in L], color=c, lw=lw, label=label(n))
    axs[0].set_title("Held-out real frames: colour drift + gray-world\nloss (relative to step 0)", fontsize=11)
    axs[1].set_title("Spread of β_D(red) across the 5 dives\n(held-out frames; falling to 0 = collapse)", fontsize=11)
    axs[2].set_title("Synthetic validation PSNR (dB)", fontsize=11)
    for ax in axs:
        ax.set_xlabel("fine-tuning step")
    axs[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.suptitle("Fine-tuning curves", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.06)
    fig.savefig(FIG / "fig17_ft_curves.png")
    plt.close(fig)


def sites(panels, F):
    desc = {"D1": "reef 10 m", "D2": "reef 10 m,\nhazy", "D3": "reef 4 m\n(held out)", "D4": "canyon,\nturbid", "D5": "reef 5 m"}
    fig, axs = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.6), sharey=True)
    for ax, n in zip(axs, panels):
        for ci, c in enumerate("RGB"):
            v = [F[n]["sites"][s]["beta_D"][ci] for s in DIVES]
            e = [F[n]["sites"][s]["beta_D"][ci] * F[n]["sites"][s]["beta_D_cv"][ci] for s in DIVES]
            ax.errorbar(np.arange(5) + (ci - 1) * 0.18, v, yerr=e, fmt=CH_MARK[c], color=CH[c], capsize=3,
                        label=["Red", "Green", "Blue"][ci])
        ax.set_xticks(range(5), [f"{s}\n{desc[s]}" for s in DIVES], fontsize=9)
        ax.set_title(f"{label(n)}\nspread across dives {between(F, n):.2f}", fontsize=11)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel(r"mean $\beta_D$ per dive (1/m), bar = ±1 std over frames")
    axs[0].legend(fontsize=9)
    fig.suptitle("Water-type read-out over all 658 downloaded Sea-thru frames (no labels used)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.savefig(FIG / "fig18_ft_sites.png")
    plt.close(fig)


def uieb_grid(sel):
    o = {"beta_rgbd": np.load(RES / "uieb_outputs.npz"), sel: np.load(RES / f"uieb_outputs_{sel}.npz")}
    idx = [0, 2, 4, 6]
    cols = [("beta_rgbd", "raw", "Raw"), ("beta_rgbd", "BetaNet + gray-world", label("beta_rgbd")),
            (sel, "BetaNet + gray-world", label(sel)), ("beta_rgbd", "reference", "UIEB reference")]
    fig, axs = plt.subplots(len(idx), len(cols), figsize=(3.6 * len(cols), 2.7 * len(idx)))
    for i, k in enumerate(idx):
        for j, (n, m, t) in enumerate(cols):
            axs[i, j].imshow(np.clip(o[n][f"{k}|{m}"].astype(np.float32), 0, 1))
            no_axes(axs[i, j])
            if i == 0:
                axs[i, j].set_title(t, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.995, wspace=0.03, hspace=0.05)
    fig.suptitle("UIEB validation images (DAv2 depth): before vs after fine-tuning", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", y=0.96)
    fig.savefig(FIG / "fig19_ft_uieb.png", dpi=120)
    plt.close(fig)


def tradeoff(R, F):
    """Teacher-weight sweep (no same-dive loss): what fine-tuning fixes vs what it keeps."""
    runs = [("beta_rgbd", None), ("ft_nodive", 0.0), ("ft_nodive_t01", 0.1), ("ft_nodive_t03", 0.3)]
    runs = [(n, w) for n, w in runs if n in F]
    xs = [w for n, w in runs if w is not None]
    ft = [n for n, w in runs if w is not None]
    ch = lambda n: float(np.mean([v[GW] for v in R[n]["seathru"]["chart"].values()]))
    panels = [("Chart grey-patch error (°)", ch, "lower is better"),
              ("UIEB PSNR (dB)", lambda n: R[n]["uieb"]["summary"]["BetaNet + gray-world"]["psnr"], "higher is better"),
              ("Held-out D3: colour drift with distance", lambda n: F[n]["heldout_D3"]["range"], "lower is better"),
              ("Spread of β_D across dives", lambda n: between(F, n), "higher = dives told apart")]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, (t, f, sub) in zip(axs, panels):
        ax.plot(xs, [f(n) for n in ft], "-o", color=SERIES[0], label="fine-tuned")
        ax.axhline(f("beta_rgbd"), color=BEFORE, ls="--", lw=1.5, label="before fine-tuning")
        ax.set_xticks(xs)
        ax.set_xlabel("teacher weight")
        ax.set_title(f"{t}\n{sub}", fontsize=11)
    axs[0].legend(fontsize=9)
    fig.suptitle("Teacher weight trades colour fixes on real dives against keeping the per-dive water read-out",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.08)
    fig.savefig(FIG / "fig20_ft_tradeoff.png")
    plt.close(fig)


def main():
    sel, collapsed, *others = sys.argv[1:]
    models = [collapsed] + [m for m in others if m != sel] + [sel]
    R = {n: json.load(open(RES / ("real_eval.json" if n == "beta_rgbd" else f"real_eval_{n}.json")))
         for n in ["beta_rgbd"] + models}
    F = json.load(open(RES / "finetune_eval.json"))
    real_grid(sel, collapsed, R)
    out = metrics(models, sel, collapsed, R, F)
    # only runs that start from the synthetic-only model (the UIEB runs start from the collapsed v1)
    curves([m for m in models if (RES / m / "log.json").exists() and not m.startswith("ft_uieb")], sel, collapsed)
    sites(["beta_rgbd", collapsed, sel], F)
    uieb_grid(sel)
    tradeoff(R, F)
    json.dump(out, open(RES / "finetune_figure_numbers.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
