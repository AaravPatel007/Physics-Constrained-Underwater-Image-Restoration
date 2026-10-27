"""Prepare the downloaded Sea-thru frames (linear 16-bit PNG + SfM metric range map) at 1/4 resolution,
run Depth Anything V2 variants on them, and score each against the SfM ground truth.

Writes data/seathru_small.npz and results/depth_eval.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import depth as dav2
from metrics import depth_errors

ROOT = Path(__file__).resolve().parents[1]
SCALE = 4


def to_display(lin):
    """Linear RAW -> exposure-normalised sRGB uint8 (what a depth network expects to see)."""
    x = np.clip(lin / np.percentile(lin, 99.5), 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (x * 255).round().astype(np.uint8)


def main():
    pngs = sorted((ROOT / "data" / "seathru").glob("D*/D*/linearPNG/*.png"))
    names, lins, gts, disp = [], [], [], []
    for p in pngs:
        tifs = list(p.parents[1].glob(f"depth*/depth{p.stem}.tif"))
        if not tifs:
            continue
        lin = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
        h, w = lin.shape[0] // SCALE, lin.shape[1] // SCALE
        lin = cv2.resize(lin, (w, h), interpolation=cv2.INTER_AREA)
        gt = np.array(Image.open(tifs[0]), dtype=np.float32)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        names.append(f"{p.parts[-4]}/{p.stem}")
        lins.append(lin), gts.append(gt), disp.append(to_display(lin))
        print(names[-1], lin.shape, f"range {np.nanmin(gt[gt > 0]):.1f}-{np.nanmax(gt):.1f} m, valid {np.mean(gt > 0):.2f}", flush=True)

    preds = {v: dav2.predict(disp, v) for v in ["metric-indoor", "metric-outdoor", "relative"]}
    res = {}
    for v, P in preds.items():
        for name, p, g in zip(names, P, gts):
            if v == "relative":
                e = depth_errors(p, g, align="affine-inv")
            else:
                e = depth_errors(p, g)
                e_s = depth_errors(p, g, align="scale")
                res.setdefault(v + " (median-scaled)", {}).setdefault(name, e_s)
            res.setdefault(v, {})[name] = e
    summary = {v: {m: float(np.mean([e[m] for e in d.values()])) for m in ["absrel", "rmse", "delta1"]}
               for v, d in res.items()}
    json.dump({"summary": summary, "per_image": res}, open(ROOT / "results" / "depth_eval.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    # frames differ in size between dive sites -> one key per frame
    arrays = {"names": np.array(names)}
    for i in range(len(names)):
        arrays[f"lin_{i}"], arrays[f"gt_{i}"] = lins[i], gts[i]
        for v, P in preds.items():
            arrays[f"dav2_{v}_{i}"] = P[i]
    np.savez_compressed(ROOT / "data" / "seathru_small.npz", **arrays)


if __name__ == "__main__":
    main()
