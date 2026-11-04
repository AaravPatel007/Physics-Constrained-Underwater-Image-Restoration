"""Synthetic evaluation of the three models on the v2 water (+ the two old-water checkpoints as a labelled reference).

Protocol (as sunlight_phy/eval_sun.py / src/eval_synth.py, water from water_v2.sample_water_v2):
  random : 654 NYU test rooms, one v2 water each (seed 2026), sensor noise 0.005 (seed 7), batch 16
  jerlov : first 200 test rooms under each of the 10 exact Jerlov types (seed 1), E0 and sun angle still random
Models (all trained on the v2 water with the same recipe; names used everywhere):
  baseline  BetaNet, 12 outputs, predicts E                                  runs/baseline_v2data
  sun_v1    SunDepthBetaNet v1, 10 outputs, d_s; K_d = 0.8 beta_D, E0 white  runs/sun_v1_v2data
  sun_v2    SunDepthBetaNet v2, 16 outputs, E0, K_d, d_s                     runs/sun_v2
  ref: baseline (old water), ref: sun_v1 (old water)   the sunlight_phy checkpoints, trained on the old water
Restorations (PSNR, SSIM, CIEDE2000 on sRGB, metrics.full_reference):
  raw, gray-world, oracle (true coefs), true coefs | v1 sunlight (E from true d_s with K_d = 0.8 beta_D, E0 white)
  <model>, <model> | true E; sun_v1 | true d_s; sun_v2 | true E0 / true K_d / true d_s (one quantity replaced)

  python sunlight_depth_v2/eval_v2.py [--quick]
Writes three_model_comparisons/synth_eval.json and results_16_output/synth_per_image.npz, synth_jerlov.npz,
synth_gallery.npz (images of the gallery scenes).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "sunlight_phy"), str(HERE)]
from metrics import full_reference  # noqa: E402
from model import BetaNet  # noqa: E402
from physics import JERLOV_N, add_water, gray_world, restore, srgb_to_linear  # noqa: E402
from train import load_split  # noqa: E402

import eval_sun as ES  # noqa: E402  (to_np, angle_deg, depth_stats, timing, boot_ci, gallery_indices)
from model_sun import SunDepthBetaNet  # noqa: E402
from model_v2 import SunDepthBetaNetV2  # noqa: E402
from water_v2 import sample_water_v2, sunlight_v2  # noqa: E402

DEV = "cuda"
MODELS = {"baseline": HERE / "runs/baseline_v2data", "sun_v1": HERE / "runs/sun_v1_v2data", "sun_v2": HERE / "runs/sun_v2",
          "ref: baseline (old water)": ROOT / "sunlight_phy/runs/baseline_v1_repro",
          "ref: sun_v1 (old water)": ROOT / "sunlight_phy/runs/sun_v1"}
WKEYS = ("beta_D", "beta_B", "B_inf", "E")
JER = list(JERLOV_N)
OUT3, OUT16 = HERE / "three_model_comparisons", HERE / "results_16_output"


def load(path):
    ck = torch.load(Path(path) / "best.pt", map_location=DEV)
    a = ck["args"]
    if a["model"] == "sun_v2":
        net = SunDepthBetaNetV2(pretrained=False)
    elif a["model"] == "sun":
        net = SunDepthBetaNet(pretrained=False, kappa0=a["kappa0"], detach_kd=not a["no_detach_kd"])
    else:
        net = BetaNet(not a.get("no_depth", False), pretrained=False)
    net.load_state_dict(ck["state"])
    return net.to(DEV).eval()


def fwd(net, I, z, **kw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = net(I, z, **kw)
    return {k: v.float() for k, v in p.items()}


@torch.no_grad()
def run(rgb, depth, water, nets, bs=16, seed=7, keep=(), full=True):
    gen = torch.Generator().manual_seed(seed)
    res, images = {}, {}
    pred, true = {n: [] for n in nets}, {k: [] for k in water}
    for b in range(0, len(rgb), bs):
        sl = slice(b, b + bs)
        J = srgb_to_linear(rgb[sl].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[sl].to(DEV).float()[:, None]
        w = {k: v[sl] for k, v in water.items()}
        wc = {k: w[k] for k in WKEYS}
        I, _, _ = add_water(J, z, wc)
        I = (I + 0.005 * torch.randn(I.shape, generator=gen).to(DEV)).clamp(0, 1)
        for k in true:
            true[k].append(w[k].cpu())
        outs = {"raw": I, "oracle (true coefs)": restore(I, z, wc)}
        if full:
            outs["gray-world"] = gray_world(I)
            E1 = sunlight_v2(torch.ones_like(w["E0"]), 0.8 * w["beta_D"], w["d_s"])
            outs["true coefs | v1 sunlight (K_d = 0.8 beta_D, E0 white)"] = restore(I, z, {**wc, "E": E1})
        for name, net in nets.items():
            p = fwd(net, I, z)
            pred[name].append({k: v.cpu() for k, v in p.items()})
            outs[name] = restore(I, z, p)
            if name.startswith("ref:") or not full:
                continue
            outs[f"{name} | true E"] = restore(I, z, {**p, "E": w["E"]})
            if isinstance(net, SunDepthBetaNetV2):
                for q, lab in (("E0", "true E0"), ("K_d", "true K_d"), ("d_s", "true d_s")):
                    outs[f"{name} | {lab}"] = restore(I, z, fwd(net, I, z, override={q: w[q]}))
            elif isinstance(net, SunDepthBetaNet):
                outs[f"{name} | true d_s"] = restore(I, z, fwd(net, I, z, d_override=w["d_s"]))
        Jt = ES.to_np(J)
        for name, Jh in outs.items():
            Jh = ES.to_np(Jh)
            for i in range(len(Jh)):
                for m, v in full_reference(Jh[i], Jt[i]).items():
                    res.setdefault(name, {}).setdefault(m, []).append(v)
                if b + i in keep:
                    images[f"{b + i}|{name}"] = Jh[i].astype(np.float16)
        for i in range(len(Jt)):
            if b + i in keep:
                images[f"{b + i}|clean"] = Jt[i].astype(np.float16)
    cat = lambda L: {k: torch.cat([d[k] for d in L]).numpy() for k in L[0]}
    return ({m: {k: np.array(v) for k, v in d.items()} for m, d in res.items()},
            {n: cat(v) for n, v in pred.items()}, {k: torch.cat(v).numpy() for k, v in true.items()}, images)


def rel_err(p, t):
    return (np.median(np.abs(p - t) / t, 0) * 100).round(2).tolist()


def angle(a, b):
    return ES.angle_deg(torch.from_numpy(np.asarray(a, np.float32)), torch.from_numpy(np.asarray(b, np.float32))).numpy()


def implied(name, p):
    """K_d and E0 a model uses (explicit for v2, implied for v1: 0.8 beta_D and white; none for the baseline)."""
    if "K_d" in p and "E0" in p:
        return p["K_d"], p["E0"]
    if "d_s" in p:
        return 0.8 * p["beta_D"], np.ones_like(p["beta_D"])
    return None, None


def summarise(res, pred, true):
    S = {"metrics": {m: {k: float(v.mean()) for k, v in d.items()} for m, d in res.items()}, "coef_median_rel_err_pct": {},
         "sunlight_angle_deg": {}, "E0_angle_deg": {}, "Kd_median_rel_err_pct": {}, "depth": {}}
    for n, p in pred.items():
        S["coef_median_rel_err_pct"][n] = {k: rel_err(p[k], true[k]) for k in WKEYS}
        a = angle(p["E"], true["E"])
        S["sunlight_angle_deg"][n] = {"mean": float(a.mean()), "median": float(np.median(a))}
        kd, e0 = implied(n, p)
        if kd is not None:
            S["Kd_median_rel_err_pct"][n] = rel_err(kd, true["K_d"])
            a0 = angle(e0, true["E0"])
            S["E0_angle_deg"][n] = {"mean": float(a0.mean()), "median": float(np.median(a0))}
        if "d_s" in p:
            S["depth"][n] = ES.depth_stats(p["d_s"][:, 0], true["d_s"][:, 0])
    return S


def jerlov_gallery_rooms(true_by_type, types=("I", "II", "1C", "3C", "5C", "7C")):
    """One room per Jerlov type, going from shallow to deep across the types (labels only, fixed before evaluation)."""
    bins = np.linspace(0.5, 12, len(types) + 1)
    pick = {}
    for t, lo, hi in zip(types, bins[:-1], bins[1:]):
        d = true_by_type[t]["d_s"][:, 0]
        pick[t] = int(np.flatnonzero((d >= lo) & (d < hi))[0])
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 32 rooms, 16 per Jerlov type")
    ap.add_argument("--models", nargs="*", help="override as name=run_dir")
    a = ap.parse_args()
    models = dict(MODELS)
    for s in a.models or []:
        k, v = s.split("=", 1)
        models[k] = (ROOT / v).resolve()
    models = {k: v for k, v in models.items() if (Path(v) / "best.pt").exists()}
    print("models:", {k: str(v) for k, v in models.items()}, flush=True)
    nets = {k: load(v) for k, v in models.items()}
    main3 = [n for n in ("baseline", "sun_v1", "sun_v2") if n in nets]
    rgb, depth = load_split("test")
    n = 32 if a.quick else len(rgb)
    rgb, depth = rgb[:n], depth[:n]
    OUT3.mkdir(exist_ok=True), OUT16.mkdir(exist_ok=True)

    water = sample_water_v2(n, DEV, torch.Generator().manual_seed(2026))
    lab = {k: v.cpu().numpy() for k, v in water.items()}
    keep = ES.gallery_indices(lab) if not a.quick else [0, 1, 2]
    res, pred, true, images = run(rgb, depth, water, nets, keep=keep)
    S = {"models": {k: str(Path(v).relative_to(ROOT)) for k, v in models.items()}, "n_test": n, "gallery_idx": keep}
    S |= summarise(res, pred, true)
    S["paired"] = {}
    for x, y in [("sun_v2", "baseline"), ("sun_v2", "sun_v1"), ("sun_v1", "baseline")]:
        if x in res and y in res:
            S["paired"][f"{x} - {y}"] = {m: {"mean": float((res[x][m] - res[y][m]).mean()),
                                             "ci95": ES.boot_ci(res[x][m] - res[y][m]),
                                             "share_x_better": float(((res[x][m] - res[y][m]) * (-1 if m == "de2000" else 1) > 0).mean())}
                                         for m in ("psnr", "ssim", "de2000")}
    S["timing_ms"] = ES.timing({k: nets[k] for k in main3})

    # Jerlov types
    nj = 16 if a.quick else 200
    J, true_j, jpred = {}, {}, {}
    for t in JER:
        wt = sample_water_v2(nj, DEV, torch.Generator().manual_seed(1), water_type=t)
        r, pj, tj, _ = run(rgb[:nj], depth[:nj], wt, {k: nets[k] for k in main3}, full=False)
        true_j[t], jpred[t] = tj, pj
        J[t] = {f"{m}|{name}": float(v[m].mean()) for name, v in r.items() for m in ("psnr", "de2000")}
        for k in main3:
            J[t][f"E_angle|{k}"] = float(angle(pj[k]["E"], tj["E"]).mean())
            if "d_s" in pj[k]:
                J[t][f"depth_MAE|{k}"] = float(np.abs(pj[k]["d_s"] - tj["d_s"]).mean())
            kd, _ = implied(k, pj[k])
            if kd is not None:
                J[t][f"Kd_rel_err_pct|{k}"] = float((np.abs(kd - tj["K_d"]) / tj["K_d"]).mean() * 100)
        print(t, {k: round(v, 2) for k, v in J[t].items() if k.startswith("psnr")}, flush=True)
    S["jerlov"] = J
    gal = jerlov_gallery_rooms(true_j) if not a.quick else {"I": 0, "5C": 1}
    S["jerlov_gallery"] = gal
    for t, i in gal.items():  # images of the type gallery (one room per type)
        wt = sample_water_v2(nj, DEV, torch.Generator().manual_seed(1), water_type=t)
        _, _, _, im = run(rgb[:nj], depth[:nj], wt, {k: nets[k] for k in main3}, full=False, keep=(i,))
        images |= {f"jerlov_{t}|{k}": v for k, v in im.items()}

    json.dump(S, open(OUT3 / "synth_eval.json", "w"), indent=1)
    flat = {f"metric|{m}|{k}": v for m, d in res.items() for k, v in d.items()}
    flat |= {f"pred|{nm}|{k}": v for nm, d in pred.items() for k, v in d.items()}
    flat |= {f"true|{k}": v for k, v in true.items()}
    np.savez_compressed(OUT16 / "synth_per_image.npz", **flat)
    jflat = {f"{t}|true|{k}": v for t, d in true_j.items() for k, v in d.items()}
    jflat |= {f"{t}|pred|{nm}|{k}": v for t, d in jpred.items() for nm, dd in d.items() for k, v in dd.items()}
    np.savez_compressed(OUT16 / "synth_jerlov.npz", **jflat)
    np.savez_compressed(OUT16 / "synth_gallery.npz", **images)
    print(json.dumps({m: {k: round(v, 3) for k, v in d.items()} for m, d in S["metrics"].items()}, indent=1))


if __name__ == "__main__":
    main()
