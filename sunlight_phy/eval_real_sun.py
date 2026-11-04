"""Real underwater photos: baseline BetaNet (v1 recipe) vs SunDepthBetaNet. Same protocol as src/eval_real.py.

UIEB (90 val pairs, first val parquet, longest side 400 px, Depth Anything V2 metric-indoor range, coefficients from a
240-px-high copy, restoration at full size): PSNR / SSIM / CIEDE2000 to the UIEB reference.
  <model>                 physics.restore (divides by the model's own E)   <- the sunlight estimate matters here
  <model> + gray-world    remove water, then gray-world (E not used)       <- control: same for both if betas agree
  <model> + Day 1b        safe_restore(..., **CURRENT_RECIPE): the method in use for review 2 (E not used)
  <model> + Day 1b, own E safe_restore(..., wb="E"): Day 1b fog cap + red smoothing, then divide by own E
Sea-thru (24 frames D1-D5, linear RAW, SfM range): grey-patch angular error on the 2 D1 chart frames with the
model's own E (the most direct real test of a sunlight estimate) and with gray-world; predicted d_s per dive.
No ground-truth depth below the surface exists for these photos.

  python sunlight_phy/eval_real_sun.py --baseline baseline_v1_repro --sun sun_v1
Writes sunlight_phy/results/real_eval_sun.json, real_outputs_sun.npz.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import depth as dav2  # noqa: E402
from eval_real import CHARTS, chart_error, predict, t  # noqa: E402
from metrics import full_reference, uciqe  # noqa: E402
from physics import gray_world, linear_to_srgb, remove_water, srgb_to_linear  # noqa: E402
from safe_restore import CURRENT_RECIPE, safe_restore  # noqa: E402

import eval_sun as ES  # noqa: E402
from eval_sun import DEV  # noqa: E402

ROOT = HERE.parent
SEATHRU_SMALL = ROOT / "data" / "seathru_small_sun.npz"


def as_np(x):
    return x[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def uieb(nets, n=90, keep_idx=(0, 15, 30, 45, 60, 75)):
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
    depths = dav2.predict(raws, "metric-indoor", bs=1)
    res, keep, coefs = {}, {}, {n_: [] for n_ in nets}
    for i, (a, b, z) in enumerate(zip(raws, refs, depths)):
        I = srgb_to_linear(t(a / 255.0))
        zt = t(np.clip(z, 0.1, 30)[..., None])
        outs = {"raw": I, "gray-world": gray_world(I)}
        for name, net in nets.items():
            p, J = predict(net, I, zt)
            p = {k: v.float() for k, v in p.items()}
            coefs[name].append({k: v[0].tolist() for k, v in p.items()})
            outs[name] = J.clamp(0, 1.5)
            outs[f"{name} + gray-world"] = gray_world(remove_water(I, zt, p).clamp(0, 1.5))
            outs[f"{name} + Day 1b"] = safe_restore(I, zt, p, **CURRENT_RECIPE)[0]
            outs[f"{name} + Day 1b, own E"] = safe_restore(I, zt, p, **{**CURRENT_RECIPE, "wb": "E"})[0].clamp(0, 1.5)
        ref = b / 255.0
        for m, o in outs.items():
            o = as_np(linear_to_srgb(o))
            for k, v in {**full_reference(o, ref), "uciqe": uciqe(o)}.items():
                res.setdefault(m, {}).setdefault(k, []).append(float(v))
            if i in keep_idx and m in ("raw", "baseline", "sun", "baseline + Day 1b", "sun + Day 1b"):
                keep[f"uieb{i}|{m}"] = cv2.resize(o, (o.shape[1] // 2, o.shape[0] // 2)).astype(np.float16)
        if i in keep_idx:
            keep[f"uieb{i}|reference"] = cv2.resize(ref, (ref.shape[1] // 2, ref.shape[0] // 2)).astype(np.float16)
    summ = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}
    return {"summary": summ, "per_image": res, "coefs": coefs}, keep


@torch.no_grad()
def seathru(nets):
    d = np.load(SEATHRU_SMALL)
    names = list(d["names"])
    res = {"chart": {}, "frames": {}}
    keep = {}
    for i, name in enumerate(names):
        lin = d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5)
        gt = d[f"gt_{i}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        I, z = t(np.clip(lin, 0, 1)), t(gt[..., None])
        outs, fr = {"raw": I, "gray-world": gray_world(I)}, {}
        for n_, net in nets.items():
            p, J = predict(net, I, z)
            p = {k: v.float() for k, v in p.items()}
            fr[n_] = {k: v[0].tolist() for k, v in p.items()}
            outs[n_] = J.clamp(0, 1.5)
            outs[f"{n_} + gray-world"] = gray_world(remove_water(I, z, p).clamp(0, 1.5))
        res["frames"][name] = fr
        if name in CHARTS:
            res["chart"][name] = {m: chart_error(as_np(o), CHARTS[name]) for m, o in outs.items()}
        for m in ("raw", "baseline", "sun"):
            o = as_np(linear_to_srgb(outs[m]))
            keep[f"{name}|{m}"] = cv2.resize(o, (o.shape[1] // 2, o.shape[0] // 2)).astype(np.float16)
    return res, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_v1_repro")
    ap.add_argument("--sun", default="sun_v1")
    ap.add_argument("--runs", default=str(ES.RUNS))
    ap.add_argument("--res", default=str(ES.RES))
    ap.add_argument("--quick", action="store_true", help="smoke test: 8 UIEB photos")
    a = ap.parse_args()
    ES.RUNS, ES.RES = Path(a.runs), Path(a.res)
    nets = {"baseline": ES.load(a.baseline), "sun": ES.load(a.sun)}
    out, keep = {}, {}
    if SEATHRU_SMALL.exists():
        out["seathru"], k = seathru(nets)
        keep.update(k)
        print(json.dumps(out["seathru"]["chart"], indent=1), flush=True)
    out["uieb"], k = uieb(nets, n=8 if a.quick else 90)
    keep.update(k)
    print(json.dumps(out["uieb"]["summary"], indent=1))
    json.dump(out, open(ES.RES / "real_eval_sun.json", "w"), indent=1)
    np.savez_compressed(ES.RES / "real_outputs_sun.npz", **keep)


if __name__ == "__main__":
    main()
