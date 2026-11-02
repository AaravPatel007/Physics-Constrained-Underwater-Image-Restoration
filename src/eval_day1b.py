"""Day 1b (review 2): fix the two side effects of the Day 1 restore seen in the gallery. No retraining.

  - pink halo around objects (val #22): the red-from-green smoothing fits red = a*green + b in 15 px patches;
    a patch that straddles an object edge mixes both sides. Try smaller patches (r = 1, 3, 7 -> 3, 7, 15 px).
  - haze: the fog cap = darkest pixels + a safety margin (Day 1: 3 noise stds). A bigger margin lets more fog
    be removed (less haze, but too big brings the red specks back). Try margin 0, 1, 2, 3, 4, 6.
  tune : all 18 settings on 200 UIEB *train* photos (same ones as Day 1); white balance stays at the Day 1 values.
         Rule fixed before running: lowest mean dE00 to the reference; settings within 0.05 of it count as a tie,
         broken by the lowest edge dE00 (colour error next to edges = where halos show).
  val  : 90 UIEB val photos, once: no fixes / Day 1 / Day 1b. PSNR, SSIM, dE00, edge dE00, specks, haze.
Writes results/review2_day1b_smaller_smoothing_patch/eval.json; figures in figures/7_review2_day1b_smaller_smoothing_patch/ (tuning grid, gallery of
the first 25 val photos: raw | Day 1 | Day 1b | reference).

  python eval_day1b.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab

from eval_day1 import boot_ci, to_np, uieb_val
from eval_real import predict, t
from eval_synth import load
from metrics import full_reference
from physics import srgb_to_linear
from safe_restore import safe_restore, speck_score
from vizstyle import no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review2_day1b_smaller_smoothing_patch"
FIG = ROOT / "figures" / "7_review2_day1b_smaller_smoothing_patch"
MODEL = "baseline_v1_60ep_bigdata"
DAY1 = json.load(open(ROOT / "results" / "review2_day1" / "eval.json"))["tuning"]["beta_rgbd_big"]["chosen"]
RS, MS = (1, 3, 7), (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
GRID = [dict(r=r, margin=m) for r in RS for m in MS]
TIE = 0.05
NAMES = {"none": "No Day 1 fixes", "day1": "Day 1 (15 px, margin 3)", "day1b": "Day 1b"}


def patch_px(r):
    return 2 * r + 1


def haze(s):
    """Mean dark channel (min over RGB, 15 px min-filter) of an sRGB image: higher = hazier."""
    return float(cv2.erode(np.clip(s, 0, 1).min(2).astype(np.float32), np.ones((15, 15))).mean())


def edge_band(ref):
    g = (cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    return cv2.dilate(cv2.Canny(g, 50, 150), np.ones((9, 9))) > 0


def scores(o, ref, band):
    de = deltaE_ciede2000(rgb2lab(np.clip(o, 0, 1)), rgb2lab(ref))
    return {**full_reference(o, ref), "edge_de2000": float(de[band].mean()), "speck": speck_score(o), "haze": haze(o)}


def restore(I, z, p, which, best=None):
    if which == "none":
        J, _ = safe_restore(I, z, p, use_cap=False, use_smooth=False, wb="gray")
    elif which == "day1":
        J, _ = safe_restore(I, z, p, **DAY1)
    else:
        J, _ = safe_restore(I, z, p, **DAY1, **best)
    return to_np(J)


@torch.no_grad()
def tune(net, n=200):
    d = np.load(ROOT / "data" / "uieb_train.npz")
    res = {json.dumps(g): {"de2000": [], "edge_de2000": [], "haze": [], "speck": []} for g in GRID}
    ref_haze = []
    for i in range(n):
        I = srgb_to_linear(t(d[f"raw_{i}"] / 255.0))
        z = t(np.clip(d[f"z_{i}"].astype(np.float32), 0.1, 30)[..., None])
        ref = d[f"ref_{i}"] / 255.0
        band = edge_band(ref)
        ref_haze.append(haze(ref))
        p, _ = predict(net, I, z)
        for g in GRID:
            s = scores(restore(I, z, p, "day1b", g), ref, band)
            for k in res[json.dumps(g)]:
                res[json.dumps(g)][k].append(s[k])
    mean = {g: {k: float(np.mean(v)) for k, v in d_.items()} for g, d_ in res.items()}
    lo = min(m["de2000"] for m in mean.values())
    ties = [g for g, m in mean.items() if m["de2000"] <= lo + TIE]
    best = min(ties, key=lambda g: mean[g]["edge_de2000"])
    return json.loads(best), mean, float(np.mean(ref_haze)), ties


@torch.no_grad()
def eval_val(net, best, data):
    raws, refs, depths = data
    res, keep, starved = {k: {} for k in NAMES}, {}, []
    ref_haze = []
    for i, (a, b, zz) in enumerate(zip(raws, refs, depths)):
        I = srgb_to_linear(t(a / 255.0))
        z = t(np.clip(zz, 0.1, 30)[..., None])
        ref = b / 255.0
        band = edge_band(ref)
        ref_haze.append(haze(ref))
        starved.append(bool(np.median(a[..., 0]) < 10))
        p, _ = predict(net, I, z)
        for k in NAMES:
            o = restore(I, z, p, k, best)
            for m, x in scores(o, ref, band).items():
                res[k].setdefault(m, []).append(float(x))
            if i < 25:
                keep[(i, k)] = o
    st = np.array(starved)
    mets = ("psnr", "ssim", "de2000", "edge_de2000", "speck", "haze")
    out = {
        "summary": {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in res.items()},
        "red_starved": {"n": int(st.sum()),
                        "summary": {k: {m: float(np.mean(np.array(v)[st])) for m, v in d.items()} for k, d in res.items()}},
        "reference_haze": float(np.mean(ref_haze)),
        "ci_day1b_minus_day1": {m: boot_ci(res["day1"][m], res["day1b"][m]) for m in mets},
        "n_day1b_better_de2000": int(np.sum(np.array(res["day1b"]["de2000"]) < np.array(res["day1"]["de2000"]))),
        "fins_val2": {k: {m: res[k][m][2] for m in mets} for k in NAMES},
        "per_image": res,
    }
    return out, keep


def fig_grid(mean, ref_haze, best):
    rs, ms = RS, MS
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.2))
    for ax, (k, title) in zip(axs.flat, [("de2000", "Colour error ΔE (lower = better)"),
                                         ("edge_de2000", "Colour error next to edges (halo)"),
                                         ("haze", f"Haze (reference {ref_haze:.3f})"),
                                         ("speck", "Red specks (lower = better)")]):
        a = np.array([[mean[json.dumps(dict(r=r, margin=m))][k] for m in ms] for r in rs])
        ax.imshow(a, cmap="Blues", aspect="auto")
        for y in range(len(rs)):
            for x in range(len(ms)):
                star = " ★" if (rs[y], ms[x]) == (best["r"], best["margin"]) else ""
                ax.text(x, y, f"{a[y, x]:.3f}{star}" if k == "haze" else f"{a[y, x]:.2f}{star}", ha="center",
                        va="center", fontsize=10, color="white" if a[y, x] > a.mean() else "black")
        ax.set_xticks(range(len(ms)), [f"{m:g}" for m in ms])
        ax.set_yticks(range(len(rs)), [f"{patch_px(r)} px" for r in rs])
        ax.set_xlabel("fog-cap safety margin (noise levels)")
        ax.set_ylabel("smoothing patch")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Day 1b tuning on 200 UIEB training photos (★ = chosen; Day 1 was 15 px, margin 3)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "tuning_grid_train200.png")
    plt.close(fig)


def fig_gallery(keep, data, best, res, per=5, n=25):
    raws, refs, _ = data
    out = FIG / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    b = f"Day 1b ({patch_px(best['r'])} px, margin {best['margin']:g})"
    for pg in range(n // per):
        fig, axs = plt.subplots(per, 4, figsize=(16, 2.75 * per))
        for k in range(per):
            i = pg * per + k
            cols = [(raws[i] / 255.0, "Raw photo", None), (keep[(i, "day1")], NAMES["day1"], "day1"),
                    (keep[(i, "day1b")], b, "day1b"), (refs[i] / 255.0, "UIEB reference", None)]
            for c, (im, ti, key) in enumerate(cols):
                ax = axs[k, c]
                no_axes(ax), ax.imshow(np.clip(im, 0, 1))
                if k == 0:
                    ax.set_title(ti, fontsize=11)
                if key:
                    ax.text(0.02, 0.03, f"ΔE {res[key]['de2000'][i]:.1f} · haze {res[key]['haze'][i]:.2f}",
                            transform=ax.transAxes, fontsize=9, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
            axs[k, 0].text(0.02, 0.03, f"val #{i}", transform=axs[k, 0].transAxes, fontsize=9, color="white",
                           bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
        fig.suptitle(f"UIEB val photos {pg * per}–{pg * per + per - 1} (first 25, fixed in advance): Day 1 vs Day 1b",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"page_{pg + 1}.png")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    net = load(MODEL)[0]
    best, mean, ref_haze_train, ties = tune(net)
    print("chosen", best, "ties", ties, flush=True)
    fig_grid(mean, ref_haze_train, best)
    data = uieb_val(90)
    val, keep = eval_val(net, best, data)
    json.dump({"model": MODEL, "day1_settings": DAY1, "tie_tolerance_de2000": TIE,
               "tuning": {"chosen": best, "grid_mean": mean, "ties": ties, "reference_haze_train": ref_haze_train},
               "uieb_val": val}, open(OUT / "eval.json", "w"), indent=1)
    fig_gallery(keep, data, best, val["per_image"])
    s = val["summary"]
    print(f"{'':26s} {'PSNR':>6s} {'ΔE':>6s} {'edgeΔE':>7s} {'specks':>7s} {'haze':>6s}")
    for k, name in NAMES.items():
        print(f"{name:26s} {s[k]['psnr']:6.2f} {s[k]['de2000']:6.2f} {s[k]['edge_de2000']:7.2f} "
              f"{s[k]['speck']:7.2f} {s[k]['haze']:6.3f}")
    print("reference haze", round(val["reference_haze"], 3), "| CI day1b-day1", val["ci_day1b_minus_day1"])
    print("fins", val["fins_val2"])


if __name__ == "__main__":
    main()
