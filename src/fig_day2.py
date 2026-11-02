"""Day 2 (review 2) figures for one retrained run vs the current baseline (baseline_v1_60ep_bigdata), from results/review2_day2/<run>/
(written by day2_queue.sh). Output: figures/6_review2_day2/<subfolder>/
  fins_compare.png    : the fins photo (UIEB val #2), base vs this run, current recipe and all Day 1 fixes
  uieb_examples.png   : the first 6 UIEB val images (fixed in advance), base vs this run, all Day 1 fixes
  summary.png         : dive separation + water read per dive, and the headline scores, base vs this run

  python fig_day2.py day2a_added_murky_water 1_more_murky_water
  python fig_day2.py day2b_added_deep_red_starved_water 2_low_red_water
  python fig_day2.py day2c_added_camera_flaws 3_camera_flaws
  python fig_day2.py day2d_added_murky_deep_red_camera_60ep 4_all_three_combined
"""
import json
import sys
from pathlib import Path

import numpy as np

from vizstyle import INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
BASE = "baseline_v1_60ep_bigdata"
OLD = "beta_rgbd"  # = baseline_v0_40ep; results/finetune_eval.json keeps the old key
LABEL = {OLD: "old baseline (40 ep)", BASE: "current baseline (60 ep)"}
FIX = "all Day 1 fixes"


def img(o, i, k):
    return np.clip(o[f"{i}|{k}"].astype(np.float32), 0, 1)


def tag(ax, per, k, i):
    ax.text(0.02, 0.03, f"ΔE {per[k]['de2000'][i]:.1f} · specks {per[k]['speck'][i]:.1f}", transform=ax.transAxes,
            fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))


def fins(res, ev, run, name, out, i=2):
    fig, axs = plt.subplots(2, 4, figsize=(14.4, 6.6))
    for r, m in enumerate([BASE, run]):
        o, per = np.load(res / f"uieb_outputs_{m}.npz"), ev[m]["per_image"]
        for c, (k, t) in enumerate([("raw", "Raw photo"), ("current (gray-world)", "Current recipe"),
                                    (FIX, "All Day 1 fixes"), ("reference", "UIEB reference")]):
            ax = axs[r, c]
            no_axes(ax), ax.imshow(img(o, i, k))
            if k in ("current (gray-world)", FIX):
                tag(ax, per, k, i)
            if r == 0:
                ax.set_title(t, fontsize=11)
        axs[r, 0].set_ylabel(LABEL.get(m, name), fontsize=11, color=INK2)
        axs[r, 0].yaxis.set_visible(True), axs[r, 0].set_yticks([])
    fig.suptitle(f"Fins photo (UIEB val #2): current baseline vs {name}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fins_compare.png")


def examples(res, ev, run, name, out, n=6):
    cols = [("raw", None, "Raw photo"), (FIX, BASE, "Current baseline (60 ep) + Day 1 fixes"),
            (FIX, run, f"{name} + Day 1 fixes"), ("reference", None, "UIEB reference")]
    o = {m: np.load(res / f"uieb_outputs_{m}.npz") for m in (BASE, run)}
    fig, axs = plt.subplots(n, 4, figsize=(15.2, 2.9 * n))
    for i in range(n):
        for c, (k, m, t) in enumerate(cols):
            ax = axs[i, c]
            no_axes(ax), ax.imshow(img(o[m or BASE], i, k))
            if m:
                tag(ax, ev[m]["per_image"], k, i)
            if i == 0:
                ax.set_title(t, fontsize=10)
    fig.suptitle(f"UIEB val (first 6, fixed in advance): current baseline vs {name}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "uieb_examples.png")


def bars(ax, names, vals, title, fmt, lower_better):
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, color=[SERIES[j] for j in range(len(names))], edgecolor="white", linewidth=2)
    for j, v in enumerate(vals):
        ax.text(j, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_title(title + ("\n(lower = better)" if lower_better else "\n(higher = better)"), fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="x", visible=False)


def summary(res, ev, run, name, out):
    sep = json.load(open(res / "dive_separation.json"))
    old = json.load(open(ROOT / "results" / "finetune_eval.json"))
    models = [OLD, BASE, run]
    sites = {m: (sep.get(m) or old[m])["sites"] for m in models}
    names = [LABEL.get(m, name).replace(" (", "\n(").replace(" water", "\nwater") for m in models]
    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, :3])
    dives = [d for d in sites[BASE] if d != "separation"]
    w = 0.26
    for j, m in enumerate(models):
        v = [np.mean(sites[m][d]["beta_D"]) for d in dives]
        ax.bar(np.arange(len(dives)) + (j - 1) * w, v, w, color=SERIES[j], edgecolor="white", linewidth=2,
               label=f"{LABEL.get(m, name)}  (separation {sites[m]['separation']:.2f})")
    ax.set_xticks(range(len(dives)), [d + (" (turbid)" if d == "D4" else "") for d in dives])
    ax.set_ylabel("mean β_D read by the model [1/m]")
    ax.set_title("Water read per real dive: D4 (turbid) should stand out as the murkiest")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[0, 3])
    bars(ax, names, [sites[m]["separation"] for m in models], "Dive separation", "{:.2f}", False)

    u = {m: ev["uieb"][m]["summary"][FIX] for m in (BASE, run)}
    syn = ev["synth_psnr"]
    two = [names[1], names[2]]
    panels = [("UIEB 90 photos: ΔE00", [u[m]["de2000"] for m in (BASE, run)], "{:.2f}", True),
              ("Fins photo: ΔE00", [ev["uieb"][m]["per_image"][FIX]["de2000"][2] for m in (BASE, run)], "{:.1f}", True),
              ("Fake test water: PSNR [dB]", [syn[m]["+ fog cap + smoothing"] for m in (BASE, run)], "{:.2f}", False)]
    if "synth_psnr_low_red" in ev:
        panels.append(("Deep red-starved fake: PSNR [dB]",
                       [ev["synth_psnr_low_red"][m]["+ fog cap + smoothing"] for m in (BASE, run)], "{:.2f}", False))
    else:
        panels.append(("Colour charts: angle [°]",
                       [np.mean([ev["charts"][m]["SfM range"][f][FIX] for f in ev["charts"][m]["SfM range"]])
                        for m in (BASE, run)], "{:.1f}", True))
    for c, (t, v, f, lb) in enumerate(panels):
        ax = fig.add_subplot(gs[1, c])
        bars(ax, two, v, t, f, lb)
        for p, col in zip(ax.patches, SERIES[1:3]):
            p.set_facecolor(col)
    fig.text(0.01, 0.005, "All scores with the Day 1 fixes on. Bottom row: current baseline vs this run only.",
             fontsize=9, color=MUTED)
    fig.suptitle(f"Day 2, {name}: does it help?", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "summary.png")


def main():
    run, sub = sys.argv[1], sys.argv[2]
    name = " ".join(sub.split("_")[1:])
    res = ROOT / "results" / "review2_day2" / run
    out = ROOT / "figures" / "6_review2_day2" / sub
    out.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(res / "eval.json"))
    fins(res, ev["uieb"], run, name, out)
    examples(res, ev["uieb"], run, name, out)
    summary(res, ev, run, name, out)
    print(out)


if __name__ == "__main__":
    main()
