"""Real underwater photos for the three models (+ old-water references). Protocol of sunlight_phy/eval_real_sun.py.

UIEB (90 val pairs, Depth Anything V2 metric-indoor range): CIEDE2000 / PSNR / SSIM to the UIEB reference for
  <model>               physics.restore with the model's own E           <- the sunlight estimate matters
  <model> + gray-world  remove water, then gray-world (E not used)       <- control
  <model> + Day 1b      safe_restore(..., **CURRENT_RECIPE) (E not used) <- the recipe in use
Sea-thru: grey-patch angular error on the D1/T_S03272 colour chart (own E, and gray-world); per-frame predictions.
Ground truth on real photos: the UIEB reference images (a human-chosen enhancement, not a measurement) and the
Sea-thru colour chart (a measurement of the true grey). There is NO ground truth for d_s, K_d or E0 on real photos:
those predictions are reported as unvalidated.

  cd src && python ../sunlight_depth_v2/eval_real_v2.py
Writes three_model_comparisons/real_eval.json and qualitative_images/real_outputs.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[0] / "src"), str(HERE.parents[0] / "sunlight_phy"), str(HERE)]
import eval_real_sun as ERS  # noqa: E402
import eval_v2 as EV  # noqa: E402

KEEP = ("raw", "baseline", "sun_v1", "sun_v2", "baseline + Day 1b", "sun_v2 + Day 1b")


def main():
    quick = "--quick" in sys.argv
    models = {k: v for k, v in EV.MODELS.items() if (Path(v) / "best.pt").exists()}
    nets = {k: EV.load(v) for k, v in models.items()}
    print("models:", list(nets), flush=True)
    out, keep = {"models": {k: str(Path(v).relative_to(EV.ROOT)) for k, v in models.items()}}, {}
    if ERS.SEATHRU_SMALL.exists():
        out["seathru"], k = seathru(nets)
        keep.update(k)
        print(json.dumps(out["seathru"]["chart"], indent=1), flush=True)
    out["uieb"], k = uieb(nets, n=8 if quick else 90)
    keep.update(k)
    print(json.dumps({m: {kk: round(v, 3) for kk, v in d.items()} for m, d in out["uieb"]["summary"].items()}, indent=1))
    (HERE / "three_model_comparisons").mkdir(exist_ok=True), (HERE / "qualitative_images").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "three_model_comparisons" / "real_eval.json", "w"), indent=1)
    np.savez_compressed(HERE / "qualitative_images" / "real_outputs.npz", **keep)


def seathru(nets):
    """eval_real_sun.seathru with this experiment's model names."""
    import cv2
    import torch
    from physics import gray_world, linear_to_srgb, remove_water

    d = np.load(ERS.SEATHRU_SMALL)
    res, keep = {"chart": {}, "frames": {}}, {}
    with torch.no_grad():
        for i, name in enumerate(d["names"]):
            lin = d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5)
            gt = d[f"gt_{i}"].copy()
            gt[~(gt > 0.1)] = np.nanmax(gt)
            I, z = ERS.t(np.clip(lin, 0, 1)), ERS.t(gt[..., None])
            outs, fr = {"raw": I, "gray-world": gray_world(I)}, {}
            for n_, net in nets.items():
                p, J = ERS.predict(net, I, z)
                p = {k: v.float() for k, v in p.items()}
                fr[n_] = {k: v[0].tolist() for k, v in p.items()}
                outs[n_] = J.clamp(0, 1.5)
                outs[f"{n_} + gray-world"] = gray_world(remove_water(I, z, p).clamp(0, 1.5))
            res["frames"][str(name)] = fr
            if name in ERS.CHARTS:
                res["chart"][str(name)] = {m: ERS.chart_error(ERS.as_np(o), ERS.CHARTS[name]) for m, o in outs.items()}
                for m in ("raw", "baseline", "sun_v1", "sun_v2"):
                    if m in outs:
                        o = ERS.as_np(linear_to_srgb(outs[m]))
                        keep[f"{name}|{m}"] = cv2.resize(o, (o.shape[1] // 2, o.shape[0] // 2)).astype(np.float16)
    return res, keep


def uieb(nets, n=90, keep_idx=(0, 15, 30, 45, 60, 75)):
    """eval_real_sun.uieb with this experiment's model names in the saved-image list."""
    import io

    import cv2
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from metrics import full_reference, uciqe
    from physics import gray_world, linear_to_srgb, remove_water, srgb_to_linear
    from safe_restore import CURRENT_RECIPE, safe_restore

    f = sorted((EV.ROOT / "data" / "uieb" / "data").glob("val-*.parquet"))[0]
    rows = pq.read_table(f).to_pylist()[:n]
    raws, refs = [], []
    for r in rows:
        a = np.array(Image.open(io.BytesIO(r["raw"]["bytes"])).convert("RGB"))
        b = np.array(Image.open(io.BytesIO(r["gt"]["bytes"])).convert("RGB"))
        s = 400 / max(a.shape[:2])
        size = (int(a.shape[1] * s) // 8 * 8, int(a.shape[0] * s) // 8 * 8)
        raws.append(cv2.resize(a, size, interpolation=cv2.INTER_AREA))
        refs.append(cv2.resize(b, size, interpolation=cv2.INTER_AREA))
    depths = ERS.dav2.predict(raws, "metric-indoor", bs=1)
    res, keep, coefs = {}, {}, {k: [] for k in nets}
    with torch.no_grad():
        for i, (a, b, z) in enumerate(zip(raws, refs, depths)):
            I = srgb_to_linear(ERS.t(a / 255.0))
            zt = ERS.t(np.clip(z, 0.1, 30)[..., None])
            outs = {"raw": I, "gray-world": gray_world(I)}
            for name, net in nets.items():
                p, J = ERS.predict(net, I, zt)
                p = {k: v.float() for k, v in p.items()}
                coefs[name].append({k: v[0].tolist() for k, v in p.items()})
                outs[name] = J.clamp(0, 1.5)
                outs[f"{name} + gray-world"] = gray_world(remove_water(I, zt, p).clamp(0, 1.5))
                outs[f"{name} + Day 1b"] = safe_restore(I, zt, p, **CURRENT_RECIPE)[0]
            ref = b / 255.0
            for m, o in outs.items():
                o = ERS.as_np(linear_to_srgb(o))
                for k, v in {**full_reference(o, ref), "uciqe": uciqe(o)}.items():
                    res.setdefault(m, {}).setdefault(k, []).append(float(v))
                if i in keep_idx and m in KEEP:
                    keep[f"uieb{i}|{m}"] = cv2.resize(o, (o.shape[1] // 2, o.shape[0] // 2)).astype(np.float16)
            if i in keep_idx:
                keep[f"uieb{i}|reference"] = cv2.resize(ref, (ref.shape[1] // 2, ref.shape[0] // 2)).astype(np.float16)
    summ = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}
    return {"summary": summ, "per_image": res, "coefs": coefs}, keep


if __name__ == "__main__":
    main()
