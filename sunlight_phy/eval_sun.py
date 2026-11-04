"""Synthetic evaluation: baseline BetaNet (v1 recipe) vs SunDepthBetaNet, on exactly the eval_synth.py test.

Test protocol copied from src/eval_synth.py (so the raw / oracle numbers must equal results/synth_eval_big.json):
  random : 654 NYU test rooms, one water each from sample_water seed 2026, sensor noise 0.005 from seed 7, batch 16
  jerlov : the first 200 test rooms under each of the 10 exact Jerlov types (sample_water seed 1)
The waters come from sun_physics.sample_water_sun, which returns the same waters plus their sunlight depth d_s.

Restorations scored (PSNR, SSIM, CIEDE2000 on sRGB, metrics.full_reference):
  raw, gray-world, oracle (true coefficients)                       as in eval_synth.py
  baseline, sun                                                     the two models, plain physics.restore
  baseline | true E, sun | true E                                   each model's water estimate, true sunlight
  sun | true d_s                                                    the new model given the true depth
  true coefs | E(true d_s, K_d = 0.8 beta_D)                       floor of the option-A sunlight model itself
Also: coefficient errors, sunlight error, depth errors, latency.

  python sunlight_phy/eval_sun.py --baseline baseline_v1_repro --sun sun_v1
Writes sunlight_phy/results/synth_eval_sun.json and synth_per_image_sun.npz (+ gallery images).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from metrics import full_reference  # noqa: E402
from model import BetaNet  # noqa: E402
from physics import JERLOV_N, add_water, gray_world, linear_to_srgb, restore, srgb_to_linear  # noqa: E402
from train import load_split  # noqa: E402

from model_sun import SunDepthBetaNet  # noqa: E402
from sun_physics import KAPPA0, coefs_only, sample_water_sun, sunlight_from_depth  # noqa: E402

DEV = "cuda"
RUNS, RES = HERE / "runs", HERE / "results"


def load(name):
    ck = torch.load(RUNS / name / "best.pt", map_location=DEV)
    a = ck["args"]
    if a["model"] == "sun":
        net = SunDepthBetaNet(pretrained=False, kappa0=a["kappa0"], detach_kd=not a["no_detach_kd"])
    else:
        net = BetaNet(not a["no_depth"], pretrained=False)
    net.load_state_dict(ck["state"])
    return net.to(DEV).eval()


def to_np(x):
    return linear_to_srgb(x).permute(0, 2, 3, 1).cpu().numpy()


def angle_deg(a, b):
    """Angle between two RGB colours (B,3): the colour error of an illuminant estimate (0 = right colour)."""
    c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def gallery_indices(true, per_cell=1):
    """Representative test images chosen from the *labels only* (before looking at any prediction): one image
    for every (depth bin x turbidity tercile) cell, the first one in a fixed random order."""
    d, t = true["d_s"][:, 0], true["beta_D"].mean(1)
    tb = np.quantile(t, [1 / 3, 2 / 3])
    order = np.random.default_rng(0).permutation(len(d))
    pick = []
    for lo, hi in [(0.5, 3), (3, 6), (6, 9), (9, 12.01)]:
        for tlo, thi in [(-1, tb[0]), (tb[0], tb[1]), (tb[1], 99)]:
            cell = [i for i in order if lo <= d[i] < hi and tlo <= t[i] < thi]
            pick += [int(i) for i in cell[:per_cell]]
    return pick


@torch.no_grad()
def run(rgb, depth, water, nets, bs=16, seed=7, keep=(), full=True):
    gen = torch.Generator().manual_seed(seed)
    res, pred, images = {}, {n: [] for n in nets}, {}
    true = {k: [] for k in water}
    for b in range(0, len(rgb), bs):
        sl = slice(b, b + bs)
        J = srgb_to_linear(rgb[sl].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[sl].to(DEV).float()[:, None]
        w = {k: v[sl] for k, v in water.items()}
        I, _, _ = add_water(J, z, coefs_only(w))
        I = (I + 0.005 * torch.randn(I.shape, generator=gen).to(DEV)).clamp(0, 1)
        for k in true:
            true[k].append(w[k].cpu())
        outs = {"raw": I, "oracle (true coefs)": restore(I, z, coefs_only(w))}
        if full:
            outs["gray-world"] = gray_world(I)
            Ef = sunlight_from_depth(KAPPA0 * w["beta_D"], w["d_s"])
            outs["true coefs | E(true d_s, K_d=0.8 beta_D)"] = restore(I, z, {**coefs_only(w), "E": Ef})
        for name, net in nets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            p = {k: v.float() for k, v in p.items()}
            pred[name].append({k: v.cpu() for k, v in p.items()})
            outs[name] = restore(I, z, p)
            outs[f"{name} | true E"] = restore(I, z, {**p, "E": w["E"]})
            if isinstance(net, SunDepthBetaNet):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    po = net(I, z, d_override=w["d_s"])
                outs[f"{name} | true d_s"] = restore(I, z, {k: v.float() for k, v in po.items()})
        Jt = to_np(J)
        for name, Jh in outs.items():
            Jh = to_np(Jh)
            for i in range(len(Jh)):
                for m, v in full_reference(Jh[i], Jt[i]).items():
                    res.setdefault(name, {}).setdefault(m, []).append(v)
                if b + i in keep:
                    images[f"{b + i}|{name}"] = Jh[i].astype(np.float16)
        for i in range(len(Jt)):
            if b + i in keep:
                images[f"{b + i}|clean"] = Jt[i].astype(np.float16)
        print(b, flush=True)
    cat = lambda L: {k: torch.cat([d[k] for d in L]).numpy() for k in L[0]}
    return ({m: {k: np.array(v) for k, v in d.items()} for m, d in res.items()},
            {n: cat(v) for n, v in pred.items()}, {k: torch.cat(v).numpy() for k, v in true.items()}, images)


def coef_rel_err(p, t):
    return {k: (np.median(np.abs(p[k] - t[k]) / t[k], 0) * 100).round(2).tolist()
            for k in ("beta_D", "beta_B", "B_inf", "E")}


def sun_err(p, t):
    """Sunlight colour error: angle between estimated and true E [deg], and mean |log E_hat - log E| (training form)."""
    a = angle_deg(torch.from_numpy(p["E"]), torch.from_numpy(t["E"])).numpy()
    lg = np.abs(np.log(p["E"]) - np.log(t["E"])).mean(1)
    return a, lg


def depth_stats(dp, dt):
    e = dp - dt
    return {"MAE_m": float(np.abs(e).mean()), "RMSE_m": float(np.sqrt((e ** 2).mean())),
            "median_abs_m": float(np.median(np.abs(e))), "bias_m": float(e.mean()),
            "rel_err_pct": float((np.abs(e) / dt).mean() * 100), "log_l1": float(np.abs(np.log(dp) - np.log(dt)).mean()),
            "R2": float(1 - (e ** 2).sum() / ((dt - dt.mean()) ** 2).sum()), "pearson_r": float(np.corrcoef(dp, dt)[0, 1])}


def depth_floor(true):
    """Best depth any estimator can reach from the sunlight colour, for this generator.

    The generator's E depends on d_s only through tau = kappa * d_s (kappa ~ U(0.6, 1), d_s ~ U(0.5, 12)). Even if
    tau were read perfectly, d_s | tau has density ~ 1/d_s on [max(0.5, tau), min(12, tau / 0.6)], whose median
    (the log-L1-optimal estimate) is the geometric mean of the two ends."""
    tau = true["kappa"][:, 0] * true["d_s"][:, 0]
    lo, hi = np.maximum(0.5, tau / 1.0), np.minimum(12.0, tau / 0.6)
    est = np.sqrt(lo * hi)
    return est, depth_stats(est, true["d_s"][:, 0])


@torch.no_grad()
def timing(nets):
    I = torch.rand(1, 3, 240, 320, device=DEV)
    z = torch.rand(1, 1, 240, 320, device=DEV) * 5
    out = {}
    for n, net in nets.items():
        for with_restore in (False, True):
            for _ in range(10):
                net(I, z)
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(50):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = net(I, z)
                    if with_restore:
                        restore(I, z, {k: v.float() for k, v in p.items()})
                torch.cuda.synchronize()
                ts.append((time.time() - t) / 50 * 1000)
            out[f"{n}{' + restore' if with_restore else ''}"] = {"median_ms": float(np.median(ts)), "all_ms": ts}
    return out


def boot_ci(x, n=5000, seed=0):
    r = np.random.default_rng(seed)
    m = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    global RUNS, RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_v1_repro")
    ap.add_argument("--sun", default="sun_v1")
    ap.add_argument("--runs", default=str(RUNS), help="folder with <name>/best.pt")
    ap.add_argument("--res", default=str(RES), help="output folder")
    ap.add_argument("--quick", action="store_true", help="smoke test: 32 test rooms, 16 per Jerlov type")
    a = ap.parse_args()
    RUNS, RES = Path(a.runs), Path(a.res)
    RES.mkdir(exist_ok=True)
    nets = {"baseline": load(a.baseline), "sun": load(a.sun)}
    rgb, depth = load_split("test")
    if a.quick:
        rgb, depth = rgb[:32], depth[:32]
    tim = timing(nets)
    print(json.dumps({k: round(v["median_ms"], 2) for k, v in tim.items()}), flush=True)

    water = sample_water_sun(len(rgb), DEV, torch.Generator().manual_seed(2026))
    true0 = {k: v.cpu().numpy() for k, v in water.items()}
    keep = gallery_indices(true0)
    res, pred, true, images = run(rgb, depth, water, nets, keep=set(keep))
    fl_est, fl_stats = depth_floor(true)

    out = {"models": {"baseline": a.baseline, "sun": a.sun}, "n_test": len(rgb), "timing_ms": tim,
           "random": {m: {k: float(v.mean()) for k, v in d.items()} for m, d in res.items()},
           "coef_median_rel_err_pct": {n: coef_rel_err(pred[n], true) for n in nets}, "gallery_idx": keep}
    for n in nets:
        ang, lg = sun_err(pred[n], true)
        out.setdefault("sunlight", {})[n] = {"angle_deg_mean": float(ang.mean()), "angle_deg_median": float(np.median(ang)),
                                             "log_l1": float(lg.mean())}
    dt, dp = true["d_s"][:, 0], pred["sun"]["d_s"][:, 0]
    d_E = true["kappa"][:, 0] * dt / KAPPA0  # depth that reproduces the true E under K_d = 0.8 beta_D_true
    out["depth"] = {"vs_true_d_s": depth_stats(dp, dt), "vs_E_equivalent_depth": depth_stats(dp, d_E),
                    "floor_from_perfect_tau": fl_stats,
                    "E_equivalent_depth_as_estimate": depth_stats(np.clip(d_E, 1e-3, None), dt)}
    diff = res["sun"]["psnr"] - res["baseline"]["psnr"]
    out["paired_sun_minus_baseline"] = {
        m: {"mean": float((res["sun"][m] - res["baseline"][m]).mean()), "ci95": boot_ci(res["sun"][m] - res["baseline"][m]),
            "share_sun_better": float(((res["sun"][m] > res["baseline"][m]) if m != "de2000" else (res["sun"][m] < res["baseline"][m])).mean())}
        for m in ("psnr", "ssim", "de2000")}
    out["paired_sun_minus_baseline"]["psnr"]["share_differs_by_more_than_1dB"] = float((np.abs(diff) > 1).mean())
    np.savez(RES / "synth_per_image_sun.npz",
             **{f"metric|{m}|{k}": v for m, d in res.items() for k, v in d.items()},
             **{f"pred|{n}|{k}": v for n, d in pred.items() for k, v in d.items()},
             **{f"true|{k}": v for k, v in true.items()}, floor_est=fl_est)
    np.savez_compressed(RES / "synth_gallery_sun.npz", **images)

    # per exact Jerlov type: first 200 test rooms (as eval_synth.py)
    nj = 16 if a.quick else 200
    sub = slice(0, nj)
    out["jerlov"], jer = {}, {}
    for t in JERLOV_N:
        w = sample_water_sun(nj, DEV, torch.Generator().manual_seed(1), water_type=t)
        r, p, tr, _ = run(rgb[sub], depth[sub], w, nets, full=False)
        ang = {n: sun_err(p[n], tr)[0] for n in nets}
        out["jerlov"][t] = {**{f"psnr|{m}": float(v["psnr"].mean()) for m, v in r.items()},
                            **{f"de2000|{m}": float(v["de2000"].mean()) for m, v in r.items()},
                            **{f"E_angle_deg|{n}": float(ang[n].mean()) for n in nets},
                            "depth": depth_stats(p["sun"]["d_s"][:, 0], tr["d_s"][:, 0]),
                            "beta_D_true": tr["beta_D"][0].round(4).tolist(),
                            **{f"beta_D_pred|{n}": p[n]["beta_D"].mean(0).round(4).tolist() for n in nets}}
        jer.update({f"{t}|psnr|{m}": v["psnr"] for m, v in r.items()})
        jer.update({f"{t}|de2000|{m}": v["de2000"] for m, v in r.items()})
        jer.update({f"{t}|E_angle|{n}": ang[n] for n in nets})
        jer[f"{t}|d_pred"], jer[f"{t}|d_true"] = p["sun"]["d_s"][:, 0], tr["d_s"][:, 0]
    np.savez(RES / "synth_jerlov_sun.npz", **jer)
    json.dump(out, open(RES / "synth_eval_sun.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("random", "coef_median_rel_err_pct", "sunlight", "depth",
                                          "paired_sun_minus_baseline")}, indent=1))


if __name__ == "__main__":
    main()
