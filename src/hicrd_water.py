"""HICRD: BetaNet's water read-out against the water that was actually measured, site by site.

K_d(wavelength) is the slope of the published exponential light-with-depth fits (HI_RAMSES_Ed_profiles.xlsx), then
averaged over each camera channel with the Ximea sensor response, 400-700 nm. BetaNet's beta_D comes from photos of
the same site with DAv2 metric-indoor depth, which HICRD does not measure, so the run repeats with depth x0.5 and x2.

beta_D (along the line of sight, one camera band) and K_d (vertical, diffuse light) are not the same quantity: expect
them to rank together, not to be equal. The sensor weighting ignores the light's own spectrum.

  python src/hicrd_water.py     -> results/hicrd/water.json, figures/hicrd/beta_d_vs_kd.png
"""
import json

import numpy as np
import openpyxl
import torch
from PIL import Image
from scipy.stats import spearmanr

from device import autocast
from eval_real import t
from eval_synth import load
from physics import srgb_to_linear
from prep_uieb import resize_pair
from stress_test import with_depth
from train import ROOT

H = ROOT / "data" / "external" / "hicrd"
OUT, FIG = ROOT / "results" / "hicrd", ROOT / "figures" / "hicrd"
SITES = (1, 2, 4, 5, 7, 8)  # the sites with a usable profile
PER_SITE = 100
CH = ("R", "G", "B")


def kd_spectrum(ws):
    rows = list(ws.iter_rows(values_only=True))
    z = np.array([float(c[:-1]) for c in rows[0][1:15]])
    wl, kd = [], []
    for r in rows[1:]:
        p = np.array([np.nan if v is None else v for v in r[1:15]], float)
        ok = np.isfinite(p) & (p > 0)
        if r[0] is not None and ok.sum() >= 2:
            wl.append(r[0]), kd.append(-np.polyfit(z[ok], np.log(p[ok]), 1)[0])
    return np.array(wl, float), np.array(kd)


def sensor():
    rows = list(openpyxl.load_workbook(H / "metadata" / "Ximea_RGB_camera_sensor_responses_CMV2000-QE.xlsx",
                                       read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    a = np.array([r[:5] for r in rows[1:] if r[0] is not None], float)
    a = a[(a[:, 0] >= 400) & (a[:, 0] <= 700)]
    return a[:, 0], {"R": a[:, 4].clip(0), "G": ((a[:, 2] + a[:, 3]) / 2).clip(0), "B": a[:, 1].clip(0)}


def kd_bands():
    wb = openpyxl.load_workbook(H / "metadata" / "HI_RAMSES_Ed_profiles.xlsx", read_only=True, data_only=True)
    sw, resp = sensor()
    out = {}
    for s in SITES:
        wl, kd = kd_spectrum(wb[f"prop Site {s:02d}"])
        k = np.interp(sw, wl, kd)
        out[s] = [float((k * resp[c]).sum() / resp[c].sum()) for c in CH]
    return out


def site_photos(s):
    ps = sorted((H / "HICRD_paired" / "trainA_paired").glob(f"site{s}_good_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    return ps[::max(1, len(ps) // PER_SITE)][:PER_SITE]


@torch.no_grad()
def read_water(base, raws, zs, scale):
    out = []
    for a, z in zip(raws, zs):
        with autocast():
            p = base(srgb_to_linear(t(a / 255.0)), t(np.clip(z * scale, 0.1, 30)[..., None]))
        out.append(p["beta_D"][0].float().cpu().numpy())
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True), FIG.mkdir(parents=True, exist_ok=True)
    kd = kd_bands()
    assert all(k[0] > k[1] > 0 and k[0] > k[2] > 0 for k in kd.values()), kd  # red dies first in any water
    base = load("baseline_v1_60ep_bigdata")[0].requires_grad_(False).eval()
    res = {"sites": {}, "note": "beta_D and K_d differ in definition; DAv2 depth is unmeasured, see x0.5 and x2"}
    for s in SITES:
        ps = site_photos(s)
        raws = [resize_pair(a, a, upscale=False)[0] for a in (np.array(Image.open(p).convert("RGB")) for p in ps)]
        zs = with_depth(f"hicrd_site{s}", raws)
        res["sites"][s] = {"n": len(ps), "kd": kd[s], "z_median": float(np.median([np.median(z) for z in zs]))}
        for scale in (1.0, 0.5, 2.0):
            b = read_water(base, raws, zs, scale)
            res["sites"][s][f"beta_d_x{scale}"] = b.mean(0).tolist()
            if scale == 1.0:
                res["sites"][s]["beta_d_std"] = b.std(0).tolist()
        print(s, json.dumps(res["sites"][s]), flush=True)
    K = np.array([kd[s] for s in SITES])
    res["rank_corr_x1"] = {}
    for scale in (1.0, 0.5, 2.0):
        B = np.array([res["sites"][s][f"beta_d_x{scale}"] for s in SITES])
        res[f"mean_ratio_beta_over_kd_x{scale}"] = dict(zip(CH, (B / K).mean(0).tolist()))
        if scale == 1.0:
            res["rank_corr_x1"] = {c: float(spearmanr(K[:, i], B[:, i])[0]) for i, c in enumerate(CH)}
            res["rank_corr_all_points"] = float(spearmanr(K.ravel(), B.ravel())[0])
    json.dump(res, open(OUT / "water.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "sites"}, indent=1))

    from vizstyle import plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    B = np.array([res["sites"][s]["beta_d_x1.0"] for s in SITES])
    for i, (c, col, mk) in enumerate(zip(CH, ("#d03b3b", "#0ca30c", "#2a78d6"), "o^s")):
        ax.scatter(K[:, i], B[:, i], c=col, marker=mk, s=46, label=c, edgecolors="white", linewidths=1)
    top = max(K.max(), B.max()) * 1.08
    ax.plot([0, top], [0, top], color="#999", lw=1, ls="--", label="equal")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="measured K_d (1/m)", ylabel="BetaNet β_D (1/m)", title="HICRD: 6 sites, 3 camera channels")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "beta_d_vs_kd.png", dpi=140)


if __name__ == "__main__":
    main()
