"""Day 1 (review 2): evaluate the safer restoration (safe_restore.py) against the current recipe. No retraining.

  tune   : pick the white-balance gain cap / shades-of-gray power / guided-filter eps on 200 UIEB *train* images
           (data/uieb_train.npz), by mean CIEDE2000 to the reference. The val images are never used to choose.
  uieb   : 90 UIEB val images (DAv2 depth, cached in results/uieb_val_dav2.npz), every ablation variant:
           PSNR / SSIM / dE00 to the reference, red speck score, fog-over-signal fraction; paired bootstrap CIs;
           also the red-starved subset (median raw red < 10 of 255 levels).
  charts : the 2 Sea-thru grey-chart frames (SfM range and DAv2 + 1 range reading), grey-patch angular error.
  synth  : 654 NYU test rooms, same random water as eval_synth_big.py; restoration divides by E (no white balance),
           so this checks the fog cap and smoothing do not hurt when the coefficients are right (model and oracle).
Writes results/review2_day1/eval.json and results/review2_day1/uieb_outputs_<model>.npz (first 8 val images,
all variants). Figures: fig_day1.py.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from eval_real import CHARTS, chart_error, predict, t
from eval_synth import load
from metrics import full_reference
from physics import add_water, linear_to_srgb, sample_water, srgb_to_linear
from safe_restore import safe_restore, speck_score
from train import load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review2_day1"
DEV = "cuda"
MODELS = ["beta_rgbd", "beta_rgbd_big"]
VARIANTS = {  # name -> (fog cap, smoothing, white balance)
    "current (gray-world)": (False, False, "gray"),
    "+ fog cap": (True, False, "gray"),
    "+ fog cap + smoothing": (True, True, "gray"),
    "robust WB only": (False, False, "robust"),
    "all Day 1 fixes": (True, True, "robust"),
}
BASE, FULL = "current (gray-world)", "all Day 1 fixes"


def to_np(J):
    return linear_to_srgb(J)[0].permute(1, 2, 0).cpu().numpy()


def run_variants(I, z, p, opts, variants=VARIANTS):
    outs, diags = {}, {}
    for v, (cap, sm, wb) in variants.items():
        J, d = safe_restore(I, z, p, use_cap=cap, use_smooth=sm, wb=wb, **opts)
        outs[v], diags[v] = J, d
    return outs, diags


def boot_ci(a, b, n=2000, seed=0):
    d = np.asarray(b) - np.asarray(a)
    r = np.random.default_rng(seed)
    m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


@torch.no_grad()
def tune(net, n=200):
    d = np.load(ROOT / "data" / "uieb_train.npz")
    grid = [dict(wb_cap=c, wb_p=p, eps=e) for c in (4.0, 8.0, 16.0) for p in (1, 4) for e in (1e-4, 1e-3)]
    scores = {json.dumps(g): [] for g in grid}
    for i in range(n):
        I = srgb_to_linear(t(d[f"raw_{i}"] / 255.0))
        z = t(np.clip(d[f"z_{i}"].astype(np.float32), 0.1, 30)[..., None])
        ref = d[f"ref_{i}"] / 255.0
        p, _ = predict(net, I, z)
        for g in grid:
            J, _ = safe_restore(I, z, p, **g)
            scores[json.dumps(g)].append(full_reference(to_np(J), ref)["de2000"])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(mean, key=mean.get)
    return json.loads(best), mean


def uieb_val(n=90):
    cache = ROOT / "results" / "review2_day1" / "uieb_val_dav2.npz"  # shared by later days
    f = sorted((ROOT / "data" / "uieb" / "data").glob("val-*.parquet"))[0]
    rows = pq.read_table(f).to_pylist()[:n]
    raws, refs = [], []
    for r in rows:
        a = np.array(Image.open(io.BytesIO(r["raw"]["bytes"])).convert("RGB"))
        b = np.array(Image.open(io.BytesIO(r["gt"]["bytes"])).convert("RGB"))
        s = 400 / max(a.shape[:2])
        size = (int(a.shape[1] * s) // 8 * 8, int(a.shape[0] * s) // 8 * 8)
        raws.append(cv2.resize(a, size, interpolation=cv2.INTER_AREA))
        refs.append(cv2.resize(b, size, interpolation=cv2.INTER_AREA))
    if cache.exists():
        c = np.load(cache)
        depths = [c[f"z_{i}"].astype(np.float32) for i in range(n)]
    else:
        import depth as dav2
        depths = dav2.predict(raws, "metric-indoor", bs=1)
        np.savez_compressed(cache, **{f"z_{i}": z.astype(np.float16) for i, z in enumerate(depths)})
        torch.cuda.empty_cache()
    return raws, refs, depths


@torch.no_grad()
def eval_uieb(net, opts, data, tag):
    raws, refs, depths = data
    res, keep = {}, {}
    starved = []
    for i, (a, b, z) in enumerate(zip(raws, refs, depths)):
        I = srgb_to_linear(t(a / 255.0))
        zt = t(np.clip(z, 0.1, 30)[..., None])
        ref = b / 255.0
        p, _ = predict(net, I, zt)
        outs, diags = run_variants(I, zt, p, opts)
        starved.append(bool(np.median(a[..., 0]) < 10))
        for v, J in outs.items():
            o = to_np(J)
            r = {**full_reference(o, ref), "speck": speck_score(o),
                 "fog_over_signal_red": diags[v]["fog_over_signal_after"][0]}
            for k, x in r.items():
                res.setdefault(v, {}).setdefault(k, []).append(float(x))
            if i < 8:
                keep[f"{i}|{v}"] = o.astype(np.float16)
        res.setdefault("reference", {}).setdefault("speck", []).append(speck_score(ref))
        res.setdefault("raw fog over signal (red)", []).append(diags[BASE]["fog_over_signal"][0])
        if i < 8:
            keep[f"{i}|raw"], keep[f"{i}|reference"] = (a / 255.0).astype(np.float16), ref.astype(np.float16)
    np.savez_compressed(OUT / f"uieb_outputs_{tag}.npz", **keep)
    st = np.array(starved)
    summ = {v: {k: float(np.mean(x)) for k, x in d.items()} for v, d in res.items() if isinstance(d, dict)}
    hard = {v: {k: float(np.mean(np.array(x)[st])) for k, x in d.items()} for v, d in res.items()
            if isinstance(d, dict) and v != "reference"}
    ci = {k: boot_ci(res[BASE][k], res[FULL][k]) for k in ("psnr", "ssim", "de2000", "speck")}
    ci_hard = {k: boot_ci(np.array(res[BASE][k])[st], np.array(res[FULL][k])[st]) for k in ("psnr", "de2000", "speck")}
    better = {k: int(np.sum(np.array(res[FULL][k]) > np.array(res[BASE][k]))) for k in ("psnr", "ssim")}
    better["de2000"] = int(np.sum(np.array(res[FULL]["de2000"]) < np.array(res[BASE]["de2000"])))
    return {"summary": summ, "red_starved": {"n": int(st.sum()), "summary": hard, "ci_full_minus_current": ci_hard},
            "ci_full_minus_current": ci, "n_better_full_vs_current": better, "per_image": res}


@torch.no_grad()
def eval_charts(net, opts):
    d = np.load(ROOT / "data" / "seathru_small.npz")
    names = list(d["names"])
    out = {}
    for name in CHARTS:
        i = names.index(name)
        lin = d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5)
        gt = d[f"gt_{i}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        dm = np.clip(d[f"dav2_metric-indoor_{i}"], 0.1, 30)
        I = t(np.clip(lin, 0, 1))
        for dn, z in {"SfM range": gt, "DAv2 + 1 range reading": dm * np.median(gt) / np.median(dm)}.items():
            zt = t(z[..., None])
            p, _ = predict(net, I, zt)
            outs, _ = run_variants(I, zt, p, opts)
            out.setdefault(dn, {})[name] = {v: chart_error(J[0].permute(1, 2, 0).cpu().numpy(), CHARTS[name])
                                            for v, J in outs.items()}
    return out


@torch.no_grad()
def eval_synth(nets, low_red=0.0):
    rgb, depth = load_split("test")
    seed = 2026 if not low_red else 2027  # 2027 = deep red-starved test water (day 2), never used in training
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(seed), low_red=low_red)
    gen = torch.Generator().manual_seed(7)
    variants = {"no fixes": (False, False, "E"), "+ fog cap": (True, False, "E"), "+ fog cap + smoothing": (True, True, "E")}
    res = {}
    for i in range(len(rgb)):
        J = srgb_to_linear(rgb[i:i + 1].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[i:i + 1].to(DEV).float()[:, None]
        w = {k: v[i:i + 1] for k, v in water.items()}
        I, _, _ = add_water(J, z, w)
        I = (I + 0.005 * torch.randn(I.shape, generator=gen).to(DEV)).clamp(0, 1)
        Jt = to_np(J)
        coefs = {"oracle (true coefs)": w, **{n: net(I, z) for n, net in nets.items()}}
        for cn, p in coefs.items():
            for v, (cap, sm, wb) in variants.items():
                Jh, _ = safe_restore(I, z, p, use_cap=cap, use_smooth=sm, wb=wb)
                res.setdefault(cn, {}).setdefault(v, []).append(full_reference(to_np(Jh.clamp(0, 1.5)), Jt)["psnr"])
    return {cn: {v: float(np.mean(x)) for v, x in d.items()} for cn, d in res.items()}


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review2_day1", help="results/<out>/ (e.g. review2_day2 for retrained models)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--synth-low-red", action="store_true", help="also test on all-deep, red-starved synthetic water")
    a = ap.parse_args()
    OUT = ROOT / "results" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    nets = {m: load(m)[0] for m in a.models}
    data = uieb_val()
    out = {"tuning": {}, "uieb": {}, "charts": {}}
    for m, net in nets.items():
        best, grid = tune(net)
        out["tuning"][m] = {"chosen": best, "grid_mean_de2000": grid}
        print(m, "chosen", best, flush=True)
        out["uieb"][m] = eval_uieb(net, best, data, m)
        out["charts"][m] = eval_charts(net, best)
        print(json.dumps(out["uieb"][m]["summary"], indent=1), flush=True)
        print(json.dumps(out["charts"][m], indent=1), flush=True)
    out["synth_psnr"] = eval_synth(nets)
    print(json.dumps(out["synth_psnr"], indent=1))
    if a.synth_low_red:
        out["synth_psnr_low_red"] = eval_synth(nets, low_red=1.0)
        print(json.dumps(out["synth_psnr_low_red"], indent=1))
    json.dump(out, open(OUT / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
