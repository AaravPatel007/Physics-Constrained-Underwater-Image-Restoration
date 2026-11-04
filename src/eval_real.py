"""Real underwater images.

Sea-thru (24 frames, D1-D5, linear RAW + SfM range):
  - BetaNet coefficients per frame with (a) SfM range, (b) DAv2 metric-indoor range (no anchor),
    (c) DAv2 rescaled by ONE number (median SfM range) = what an altimeter / DVL reading would give
  - colour-chart grey-patch angular error on the frames where the chart is large enough
  - consistency of predicted coefficients within a dive site (water does not change between frames)
UIEB (90 val pairs): DAv2 depth, metrics against the UIEB reference images + UCIQE.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

import depth as dav2
from eval_synth import load
from metrics import angular_error_deg, full_reference, uciqe
from physics import gray_world, linear_to_srgb, remove_water, restore, srgb_to_linear
from device import DEV

ROOT = Path(__file__).resolve().parents[1]
# grey-patch centres (x, y) at 1/4 resolution, marked by hand on the zoomed charts (6 patches each)
CHARTS = {"D1/T_S03272": [(462, 305), (485, 305), (508, 305), (532, 305), (557, 305), (581, 305)],
          "D1/T_S03425": [(307, 33), (327, 30), (347, 27), (367, 24), (387, 21), (407, 18)]}


def chart_error(lin, pts, r=3):
    rgb = np.array([np.median(lin[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3), 0) for x, y in pts])
    return float(angular_error_deg(rgb).mean())


def t(x):  # (H,W,3) numpy -> (1,3,H,W) cuda
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().to(DEV)


@torch.no_grad()
def predict(net, I, z):
    """Coefficients from a 240-px-high copy (the training scale); restoration at full size."""
    h = 240
    w = int(round(I.shape[-1] * h / I.shape[-2]))
    Is = torch.nn.functional.interpolate(I, (h, w), mode="area")
    zs = torch.nn.functional.interpolate(z, (h, w), mode="nearest")
    p = net(Is, zs)
    return p, restore(I, z, p)


def seathru(net, unet=None, tag=""):
    d = np.load(ROOT / "data" / "seathru_small.npz")
    cl = np.load(ROOT / "results" / "seathru_classical_real.npz")
    names = list(d["names"])
    res = {"frames": {}, "chart": {}}
    images = {}
    for i, name in enumerate(names):
        lin = d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5)
        gt = d[f"gt_{i}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        dm = np.clip(d[f"dav2_metric-indoor_{i}"], 0.1, 30)
        depths = {"SfM range": gt, "DAv2 (no anchor)": dm, "DAv2 + 1 range reading": dm * np.median(gt) / np.median(dm)}
        I = t(np.clip(lin, 0, 1))
        frame = {}
        outs = {"raw": lin, "gray-world": gray_world(I)[0].permute(1, 2, 0).cpu().numpy(),
                "Sea-thru (classical, SfM range)": cl[f"st_{i}"]}
        for dn, z in depths.items():
            p, J = predict(net, I, t(z[..., None]))
            frame[dn] = {k: v[0].tolist() for k, v in p.items()}
            outs[f"BetaNet [{dn}]"] = J.clamp(0, 1.5)[0].permute(1, 2, 0).cpu().numpy()
            # Sea-thru's own recipe: remove water along the line of sight, then gray-world for the illuminant
            Jw = gray_world(remove_water(I, t(z[..., None]), p).clamp(0, 1.5))
            outs[f"BetaNet + gray-world [{dn}]"] = Jw[0].permute(1, 2, 0).cpu().numpy()
        if unet is not None:
            H, W = I.shape[-2:]
            pad = (0, -W % 8, 0, -H % 8)  # UNet needs sides divisible by 8
            with torch.no_grad():
                Ju = unet(torch.nn.functional.pad(I, pad, mode="reflect"),
                          torch.nn.functional.pad(t(gt[..., None]), pad, mode="reflect"))[..., :H, :W]
            outs["UNet (black box, SfM range)"] = Ju[0].permute(1, 2, 0).cpu().numpy()
        res["frames"][name] = frame
        images[name] = outs
        if name in CHARTS:
            res["chart"][name] = {m: chart_error(o, CHARTS[name]) for m, o in outs.items()}
    np.savez_compressed(ROOT / "results" / f"real_seathru_outputs{tag}.npz",
                        **{f"{n}|{m}": o.astype(np.float16) for n, outs in images.items() for m, o in outs.items()})
    return res


def uieb(net, unet=None, n=90, tag=""):
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
    depths = dav2.predict(raws, "metric-indoor", bs=1)  # sizes differ
    res, keep = {}, {}
    with torch.no_grad():
        for i, (a, b, z) in enumerate(zip(raws, refs, depths)):
            I = srgb_to_linear(t(a / 255.0))
            zt = t(np.clip(z, 0.1, 30)[..., None])
            p, J = predict(net, I, zt)
            outs = {"raw": I, "gray-world": gray_world(I), "BetaNet": J.clamp(0, 1.5),
                    "BetaNet + gray-world": gray_world(remove_water(I, zt, p).clamp(0, 1.5))}
            if unet is not None:
                outs["UNet (black box)"] = unet(I, zt)
            ref = b / 255.0
            for m, o in outs.items():
                o = linear_to_srgb(o)[0].permute(1, 2, 0).cpu().numpy()
                r = {**full_reference(o, ref), "uciqe": uciqe(o)}
                for k, v in r.items():
                    res.setdefault(m, {}).setdefault(k, []).append(float(v))
                if i < 8:
                    keep[f"{i}|{m}"] = o.astype(np.float16)
            res.setdefault("reference", {}).setdefault("uciqe", []).append(float(uciqe(ref)))
            res.setdefault("coefs", []).append({k: v[0].tolist() for k, v in p.items()})
            if i < 8:
                keep[f"{i}|reference"] = ref.astype(np.float16)
                keep[f"{i}|depth"] = z.astype(np.float16)
    np.savez_compressed(ROOT / "results" / f"uieb_outputs{tag}.npz", **keep)
    summ = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items() if m != "coefs"}
    return {"summary": summ, "per_image": res}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="beta_rgbd")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _ft (default: the main results)")
    a = ap.parse_args()
    net, _ = load(a.ckpt)
    unet = load("unet_rgbd")[0] if (ROOT / "results/unet_rgbd/best.pt").exists() else None
    out = {"seathru": seathru(net, unet, a.tag)}
    if (ROOT / "data" / "uieb" / "data").exists():
        out["uieb"] = uieb(net, unet, tag=a.tag)
    json.dump(out, open(ROOT / "results" / f"real_eval{a.tag}.json", "w"), indent=1)
    print(json.dumps(out["seathru"]["chart"], indent=1))
    if "uieb" in out:
        print(json.dumps(out["uieb"]["summary"], indent=1))
