"""Stress test (session 5): every system on every test set we have, one table + figures.

Systems (all on the same frozen BetaNet --base): raw, gray-world, BetaNet (plain restore), BetaNet + Day 1b
(safe_restore, CURRENT_RECIPE), and BetaNet + RefineNet for each --refiners checkpoint. RefineNet gets the plain
restore() output at full size, exactly as in finetune_refine.py (its training input).

Full-reference sets (PSNR / SSIM / CIEDE2000 colour error vs a reference):
  synth   654 NYU test rooms, one random water each (refine.test protocol)      reference = true clean image
  uieb160 the 160 held-out uieb_train photos (finetune_refine.real_split)        reference = UIEB enhanced photo
  uieb90  the official UIEB val split (90)                                       reference = UIEB enhanced photo
  lsui    LSUI held-out 400 (prep_extra_pairs.TEST_SPLIT)                        reference = LSUI enhanced photo
  euvp    EUVP-Dark held-out 500                                                 reference = EUVP enhanced photo
  tub     TUB (measured turbidity, 10 stages per camera), test-split experiments reference = the same camera's
          stage-1 (clearest) raw photo; also "drift" = colour error to the method's OWN stage-1 output
          (a perfect restorer gives the same image at every turbidity), and BetaNet's beta_B vs NTU.
Charts: Sea-thru D1 chart frames, SfM range, grey-patch angular error (deg) as in eval_real.py.
No-reference sets (colour cast = chroma of the mean Lab colour, lower = more neutral; UCIQE; red-speck score):
  RUIE UCCS blue / green / blue-green (100 each), RUIE UIQS (40 per visibility level A-E), U45 (45), TankImage-I.
Real photos have no true clean image: UIEB/LSUI/EUVP references are human-preferred enhancements, so those
numbers measure agreement with a preferred look, not physical correctness. Real-photo depth = DAv2 metric-indoor
(cached in data/stress_cache/), as in training.

  python src/stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_v1,refine_ft,refine_mix
  python src/stress_test.py --figures-only
"""
import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.stats import spearmanr
from skimage.color import rgb2lab

import depth as dav2
from device import DEV, autocast
from eval_day1 import boot_ci
from eval_real import CHARTS, chart_error, t
from eval_synth import load
from finetune_refine import extra_split, real_split
from metrics import full_reference, uciqe
from model import RefineNet
from physics import add_water, gray_world, linear_to_srgb, restore, sample_water, srgb_to_linear
from prep_uieb import resize_pair
from safe_restore import CURRENT_RECIPE, safe_restore, speck_score
from train import ROOT, load_split

EXT = ROOT / "data" / "external"
CACHE = ROOT / "data" / "stress_cache"
OUT = ROOT / "results" / "stress_test"
FIG = ROOT / "figures" / "9_stress_test"
TUB_EXPS = (9, 14, 15, 18, 19, 20, 22, 24)  # TUB's own test split
N_GALLERY = 4
FULL_REF = ["synth", "uieb160", "uieb90", "lsui", "euvp", "tub"]
NO_REF = ["uccs_blue", "uccs_green", "uccs_blue-green", "uiqs", "u45", "tank"]


def cast(s):
    lab = rgb2lab(np.clip(s, 0, 1))
    return float(np.hypot(lab[..., 1].mean(), lab[..., 2].mean()))


def srgb_np(x):
    return np.clip(linear_to_srgb(x.float().clamp(0, 1.5))[0].permute(1, 2, 0).cpu().numpy(), 0, 1)


def small(s, h=160):
    w = int(round(s.shape[1] * h / s.shape[0]))
    return np.asarray(Image.fromarray((s * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))


@torch.no_grad()
def systems(base, refs, I, z):
    with autocast():
        p = {k: v.float() for k, v in base(I, z).items()}
    Jb = restore(I, z, p).clamp(0, 1.5)
    outs = {"raw": I, "gray-world": gray_world(I), "BetaNet": Jb,
            "BetaNet + Day 1b": safe_restore(I, z, p, **CURRENT_RECIPE)[0]}
    h, w = I.shape[-2:]
    pad = lambda x: torch.nn.functional.pad(x, (0, -w % 8, 0, -h % 8), mode="replicate")  # U-Net needs sides divisible by 8
    for name, r in refs.items():
        with autocast():
            outs[f"RefineNet [{name}]"] = r(pad(Jb), pad(I), pad(z)).float()[..., :h, :w]
    return p, outs


# ---------- test sets: each yields (raw uint8, ref uint8 or None, z float32, meta) ----------
def with_depth(name, raws):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        d = np.load(f)
        return [d[f"z_{i}"].astype(np.float32) for i in range(len(raws))]
    zs = [np.clip(z, 0.1, 30) for z in dav2.predict(raws, "metric-indoor", bs=1)]
    np.savez_compressed(f, **{f"z_{i}": z.astype(np.float16) for i, z in enumerate(zs)})
    return zs


def pairs_npz(d, idx):
    for i in idx:
        yield d[f"raw_{i}"], d[f"ref_{i}"], d[f"z_{i}"].astype(np.float32), {"i": int(i)}


def files(name, paths):
    raws = [resize_pair(a, a, upscale=False)[0] for a in (np.array(Image.open(p).convert("RGB")) for p in paths)]
    for p, a, z in zip(paths, raws, with_depth(name, raws)):
        yield a, None, z, {"file": str(p.relative_to(EXT))}


def load_set(name):
    if name == "uieb160":
        d, _, va = real_split()
        return pairs_npz(d, va)
    if name in ("lsui", "euvp"):
        d, _, te = extra_split(name)
        return pairs_npz(d, te)
    if name == "uieb90":
        rows = pq.read_table(sorted((ROOT / "data" / "uieb" / "data").glob("val-*.parquet"))[0]).to_pylist()
        prs = [resize_pair(np.array(Image.open(io.BytesIO(r["raw"]["bytes"])).convert("RGB")),
                           np.array(Image.open(io.BytesIO(r["gt"]["bytes"])).convert("RGB"))) for r in rows]
        zs = with_depth(name, [a for a, _ in prs])
        return ((a, b, z, {"i": i}) for i, ((a, b), z) in enumerate(zip(prs, zs)))
    if name == "tub":
        ntu = {(int(r[0][3:]), int(r[1])): float(r[2]) for r in
               (l.split(",") for l in (EXT / "tub" / "av_turbidity.csv").read_text().split()[1:])}
        ps = sorted((p for p in (EXT / "tub" / "images").glob("Exp*_Cam*_Stage*.jpg")
                     if int(re.match(r"Exp(\d+)", p.name)[1]) in TUB_EXPS),
                    key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.name)))
        def gen():
            stage1 = {}
            for a, _, z, m in files(name, ps):
                e, c, s = (int(x) for x in re.findall(r"\d+", Path(m["file"]).name))
                if s == 1:
                    stage1[(e, c)] = a
                ref = stage1.get((e, c))
                if ref is None or ref.shape != a.shape:
                    continue
                yield a, ref, z, {**m, "exp": e, "cam": c, "stage": s, "ntu": ntu.get((e, s))}
        return gen()
    if name.startswith("uccs_"):
        return files(name, sorted((EXT / "ruie" / "UCCS" / name[5:]).glob("*")))
    if name == "uiqs":
        return files(name, [p for lvl in "ABCDE" for p in sorted((EXT / "ruie" / "UIQS" / lvl).glob("*"))[:40]])
    if name == "u45":
        return files(name, sorted((EXT / "u45" / "upload" / "U45" / "U45").glob("*.png")))
    if name == "tank":
        return files(name, sorted((EXT / "tankimage_i").glob("*/*/*.bmp")))
    raise ValueError(name)


@torch.no_grad()
def run_real(name, base, refs):
    res, gal, coefs, metas, first = {}, [], [], [], {}
    for k, (a, ref, z, meta) in enumerate(load_set(name)):
        I, zt = srgb_to_linear(t(a / 255.0)), t(z[..., None])
        p, outs = systems(base, refs, I, zt)
        imgs = {m: srgb_np(o) for m, o in outs.items()}
        for m, s in imgs.items():
            r = {"uciqe": uciqe(s), "cast": cast(s), "speck": speck_score(s)}
            if ref is not None:
                r.update(full_reference(s, ref / 255.0))
            if name == "tub":  # drift to the same method's stage-1 output
                key = (meta["exp"], meta["cam"], m)
                if meta["stage"] == 1:
                    first[key] = s
                r["drift_de2000"] = full_reference(s, first[key])["de2000"] if key in first else float("nan")
            for kk, v in r.items():
                res.setdefault(m, {}).setdefault(kk, []).append(float(v))
        coefs.append({kk: v[0].tolist() for kk, v in p.items()})
        metas.append(meta)
        if k < N_GALLERY or (name == "tub" and meta.get("stage") == 10 and len(gal) < 2 * N_GALLERY):
            gal.append({**{m: small(s) for m, s in imgs.items()}, **({"reference": small(ref / 255.0)} if ref is not None else {})})
        if k % 100 == 0:
            print(f"  {name} {k}", flush=True)
    return res, gal, coefs, metas


@torch.no_grad()
def run_synth(base, refs, n=654):
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    gen, res, gal = torch.Generator().manual_seed(7), {}, []
    for i in range(min(n, len(rgb))):
        J = srgb_to_linear(rgb[i:i + 1].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[i:i + 1].to(DEV).float()[:, None]
        w = {k: v[i:i + 1] for k, v in water.items()}
        I = (add_water(J, z, w)[0] + 0.005 * torch.randn(J.shape, generator=gen).to(DEV)).clamp(0, 1)
        _, outs = systems(base, refs, I, z)
        outs["oracle (true coefs)"] = restore(I, z, w)
        tgt = srgb_np(J)
        imgs = {m: srgb_np(o) for m, o in outs.items()}
        for m, s in imgs.items():
            for kk, v in {**full_reference(s, tgt), "cast": cast(s)}.items():
                res.setdefault(m, {}).setdefault(kk, []).append(float(v))
        if i < N_GALLERY:
            gal.append({**{m: small(s) for m, s in imgs.items()}, "reference": small(tgt)})
    return res, gal, [], []


@torch.no_grad()
def run_charts(base, refs):
    d = np.load(ROOT / "data" / "seathru_small.npz")
    res = {}
    for i, name in enumerate(map(str, d["names"])):
        if name not in CHARTS:
            continue
        lin = np.clip(d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5), 0, 1)
        gt = d[f"gt_{i}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        _, outs = systems(base, refs, t(lin), t(gt[..., None]))
        for m, o in outs.items():
            res.setdefault(m, {})[name] = chart_error(o[0].permute(1, 2, 0).float().cpu().numpy(), CHARTS[name])
    return res


def summarise(raw, refiners):
    s = {"sets": {}, "refiners": refiners}
    for name, (res, _, coefs, metas) in raw.items():
        if name == "charts":
            s["sets"][name] = {m: {"grey_patch_deg": float(np.mean(list(v.values()))), **v} for m, v in res.items()}
            continue
        e = {m: {k: float(np.nanmean(v)) for k, v in d.items()} for m, d in res.items()}
        e["n"] = len(next(iter(res.values()))["cast"])
        if "psnr" in res["BetaNet"]:  # paired bootstrap CIs (b - a): RefineNets vs BetaNet, and refiner vs refiner
            names = [f"RefineNet [{r}]" for r in refiners]
            pairs = [("BetaNet", n) for n in names] + [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
            e["ci"] = {f"{b} - {a}": {k: boot_ci(res[a][k], res[b][k]) for k in ("psnr", "de2000")} for a, b in pairs}
        if name == "tub":
            ntu = np.array([m["ntu"] for m in metas], dtype=float)
            bb = np.array([np.mean(c["beta_B"]) for c in coefs])
            bd = np.array([np.mean(c["beta_D"]) for c in coefs])
            ok = np.isfinite(ntu)
            e["coef_vs_ntu_spearman"] = {"beta_B": float(spearmanr(ntu[ok], bb[ok])[0]), "beta_D": float(spearmanr(ntu[ok], bd[ok])[0])}
            e["per_stage"] = {}
            for st in range(1, 11):
                idx = [k for k, m in enumerate(metas) if m["stage"] == st]
                if idx:
                    e["per_stage"][st] = {"ntu": float(np.nanmean(ntu[idx])), "beta_B": float(bb[idx].mean()),
                                          **{m: {k: float(np.nanmean(np.array(res[m][k])[idx])) for k in ("de2000", "drift_de2000", "psnr")}
                                             for m in res}}
        s["sets"][name] = e
    return s


# ---------- figures ----------
def short(m):
    return m.replace("BetaNet + RefineNet", "RefineNet").replace("RefineNet [", "RN [")


def bars(summary, sets, metrics, path, title):
    from vizstyle import INK2, MUTED, SERIES, plt
    sets = [x for x in sets if x in summary["sets"]]
    systems_ = [m for m, v in summary["sets"][sets[0]].items()
                if isinstance(v, dict) and m not in ("ci", "per_stage", "coef_vs_ntu_spearman", "oracle (true coefs)")]
    fig, axs = plt.subplots(len(metrics), 1, figsize=(1.6 * len(sets) * len(systems_) / 3 + 3, 3.2 * len(metrics)))
    axs = np.atleast_1d(axs)
    wdt = 0.8 / len(systems_)
    for ax, (key, label) in zip(axs, metrics):
        for j, m in enumerate(systems_):
            vals = [summary["sets"][x].get(m, {}).get(key, np.nan) for x in sets]
            ax.bar(np.arange(len(sets)) + (j - len(systems_) / 2 + 0.5) * wdt, vals, wdt, color=SERIES[j % len(SERIES)], label=short(m))
        ax.set_xticks(range(len(sets)), sets)
        ax.set_ylabel(label, color=INK2)
    axs[0].set_title(title)
    axs[0].legend(ncol=min(3, len(systems_)), fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08 - 1.2 * (len(metrics) - 1)))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def tub_figure(summary, path):
    from vizstyle import SERIES, plt
    e = summary["sets"].get("tub")
    if not e:
        return
    st = e["per_stage"]
    ntu = [st[k]["ntu"] for k in st]
    systems_ = [m for m in st[next(iter(st))] if isinstance(st[next(iter(st))][m], dict)]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, m in enumerate(systems_):
        axs[0].plot(ntu, [st[k][m]["drift_de2000"] for k in st], "o-", color=SERIES[j % len(SERIES)], label=short(m), ms=4)
        axs[1].plot(ntu, [st[k][m]["de2000"] for k in st], "o-", color=SERIES[j % len(SERIES)], ms=4)
    axs[2].plot(ntu, [st[k]["beta_B"] for k in st], "o-", color=SERIES[0])
    axs[0].set(title="Drift from own clearest-stage output ↓", xlabel="turbidity (NTU)", ylabel="ΔE00")
    axs[1].set(title="Colour error vs clearest raw photo (favours raw)", xlabel="turbidity (NTU)", ylabel="ΔE00")
    axs[2].set(title=f"BetaNet β_B vs turbidity (ρ={e['coef_vs_ntu_spearman']['beta_B']:.2f})", xlabel="turbidity (NTU)", ylabel="mean β_B (1/m)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def galleries(gal_file, path_fmt):
    from vizstyle import no_axes, plt
    g = np.load(gal_file, allow_pickle=True)["gal"].item()
    for name, rows in g.items():
        if not rows:
            continue
        cols = [c for c in rows[0] if c not in ("gray-world",)]
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.2 * len(rows)))
        axs = np.atleast_2d(axs)
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                no_axes(axs[r, c])
                if col in row:
                    axs[r, c].imshow(row[col])
                if r == 0:
                    axs[r, c].set_title(short(col), fontsize=9)
        fig.tight_layout()
        fig.savefig(path_fmt.format(name), dpi=110)
        plt.close(fig)
        print("wrote", path_fmt.format(name))


def figures():
    summary = json.load(open(OUT / "summary.json"))
    FIG.mkdir(parents=True, exist_ok=True)
    bars(summary, FULL_REF, [("psnr", "PSNR (dB) ↑"), ("de2000", "ΔE00 ↓"), ("ssim", "SSIM ↑")],
         FIG / "full_reference_bars.png", "Full-reference sets (synthetic = true clean image; real = enhanced reference)")
    bars(summary, NO_REF, [("cast", "colour cast ↓"), ("uciqe", "UCIQE ↑"), ("speck", "red specks ↓")],
         FIG / "no_reference_bars.png", "No-reference sets (colour cast, UCIQE, red specks)")
    tub_figure(summary, FIG / "tub_turbidity_curves.png")
    galleries(OUT / "galleries.npz", str(FIG / "gallery_{}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="baseline_v1_60ep_bigdata")
    ap.add_argument("--refiners", default="refine_v1,refine_ft,refine_mix")
    ap.add_argument("--only", default="", help="comma list of sets (default: all)")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if a.figures_only:
        return figures()
    base = load(a.base)[0].requires_grad_(False).eval()
    refs = {}
    for n in a.refiners.split(","):
        r = RefineNet().to(DEV).eval()
        r.load_state_dict(torch.load(ROOT / "results" / n / "best.pt", map_location=DEV)["state"])
        refs[n] = r
    sets = a.only.split(",") if a.only else FULL_REF + NO_REF + ["charts"]
    OUT.mkdir(parents=True, exist_ok=True)
    raw, gal = {}, {}
    for name in sets:
        print(name, flush=True)
        if name == "charts":
            raw[name] = (run_charts(base, refs), [], [], [])
            continue
        res, g, coefs, metas = run_synth(base, refs) if name == "synth" else run_real(name, base, refs)
        raw[name], gal[name] = (res, g, coefs, metas), g
        json.dump({"per_image": res, "meta": metas, "coefs": coefs}, open(OUT / f"per_image_{name}.json", "w"))
    summary = summarise(raw, a.refiners.split(","))
    summary["base"] = a.base
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    np.savez_compressed(OUT / "galleries.npz", gal=np.array(gal, dtype=object))
    for name, e in summary["sets"].items():
        print(f"\n== {name} (n={e.get('n', '')})")
        for m, v in e.items():
            if isinstance(v, dict) and m not in ("ci", "per_stage", "coef_vs_ntu_spearman"):
                print(f"  {m:34s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items() if isinstance(x, float)))
    figures()


if __name__ == "__main__":
    main()
