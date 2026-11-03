"""Day 3 (review 2): a small learned "touch-up" on top of the physics restore (baseline_v1_60ep_bigdata + Day 1b).

Physics removes the water using the colours the camera recorded; the touch-up learns, from the 800 UIEB train
photos and their references, the colour correction that is still missing. Two sizes:
  global : a small CNN looks at a 128 px thumbnail and outputs one colour matrix + one tone curve per channel for
           the whole photo. It can shift colours but cannot paint new detail or halos.
  local  : global + a small U-Net that adds a per-pixel correction (zero at start).
Control: the same touch-up trained on the RAW photo (no physics), to show what physics adds.

Split: UIEB train 800 -> 680 to train, 120 held out to pick the epoch and global vs local (fixed seed).
The 90 UIEB val photos are used once, at the end. Also the 2 Sea-thru grey-chart frames (never seen in training:
checks whether it learned true colour or only UIEB's editing style) and the fins photo (val #2).

  python day3_touchup.py cache     # Day 1b outputs of all 890 photos -> data/day3_cache.npz
  python day3_touchup.py train <variant>   # day3a_touchup_global | day3b_touchup_global_plus_local | day3c_control_touchup_on_raw
  python day3_touchup.py eval      # all trained variants -> results/review2_day3_touchup/eval.json + figures
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab

from eval_day1 import boot_ci, uieb_val
from eval_day1b import edge_band, haze
from eval_real import CHARTS, chart_error, predict, t
from eval_synth import load
from metrics import full_reference
from physics import linear_to_srgb, srgb_to_linear
from safe_restore import CURRENT_RECIPE, safe_restore, speck_score
from vizstyle import no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review2_day3_touchup"
FIG = ROOT / "figures" / "8_review2_day3_touchup"
CACHE = ROOT / "data" / "day3_cache.npz"
BASE = "baseline_v1_60ep_bigdata"
DEV = "cuda"
VARIANTS = {"day3a_touchup_global": ("d1b", False),
            "day3b_touchup_global_plus_local": ("d1b", True),
            "day3c_control_touchup_on_raw": ("raw", None)}  # control uses the winner's size
K, THUMB, CROP = 16, 128, 176


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def restore_srgb(net, raw_u8, z):
    I = srgb_to_linear(t(raw_u8 / 255.0))
    zt = t(np.clip(z.astype(np.float32), 0.1, 30)[..., None])
    p, _ = predict(net, I, zt)
    J, _ = safe_restore(I, zt, p, **CURRENT_RECIPE)
    return linear_to_srgb(J)[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def make_cache():
    net = load(BASE)[0]
    d = np.load(ROOT / "data" / "uieb_train.npz")
    out = {}
    for i in range(int(d["n"])):
        out[f"tr_raw_{i}"], out[f"tr_ref_{i}"] = d[f"raw_{i}"], d[f"ref_{i}"]
        out[f"tr_d1b_{i}"] = u8(restore_srgb(net, d[f"raw_{i}"], d[f"z_{i}"]))
    raws, refs, depths = uieb_val(90)
    for i in range(90):
        out[f"va_raw_{i}"], out[f"va_ref_{i}"] = raws[i], refs[i]
        out[f"va_d1b_{i}"] = u8(restore_srgb(net, raws[i], depths[i]))
    np.savez(CACHE, n_tr=int(d["n"]), **out)
    print("cache", CACHE)


class TouchUp(nn.Module):
    """Global colour matrix + per-channel monotone tone curve from a thumbnail; optional local residual U-Net.
    Starts as the identity (all last layers zero), so at epoch 0 it returns its input unchanged."""

    def __init__(self, local):
        super().__init__()
        c = [6, 16, 32, 64, 128]
        self.enc = nn.Sequential(*[m for a, b in zip(c, c[1:]) for m in (nn.Conv2d(a, b, 3, 2, 1), nn.ReLU())],
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, 9 + 3 + 3 * K)
        nn.init.zeros_(self.head.weight), nn.init.zeros_(self.head.bias)
        self.local = local
        if local:
            def blk(a, b):
                return nn.Sequential(nn.Conv2d(a, b, 3, 1, 1), nn.ReLU(), nn.Conv2d(b, b, 3, 1, 1), nn.ReLU())
            self.d1, self.d2, self.d3 = blk(6, 16), blk(16, 32), blk(32, 64)
            self.u2, self.u1 = blk(64 + 32, 32), blk(32 + 16, 16)
            self.last = nn.Conv2d(16, 3, 1)
            nn.init.zeros_(self.last.weight), nn.init.zeros_(self.last.bias)

    def forward(self, x, thumb):
        h = self.head(self.enc(thumb))
        B = x.shape[0]
        M = torch.eye(3, device=x.device) + 0.5 * h[:, :9].view(B, 3, 3)
        y = (torch.einsum("bij,bjhw->bihw", M, x) + 0.2 * h[:, 9:12, None, None]).clamp(0, 1)
        inc = F.softplus(h[:, 12:].view(B, 3, K) + 0.5413) / K  # softplus(0.5413) = 1 -> identity curve
        ys = torch.cat([torch.zeros_like(inc[..., :1]), inc.cumsum(-1)], -1)  # (B, 3, K+1)
        pos = y.flatten(2) * K
        i = pos.floor().clamp(0, K - 1).long()
        f = pos - i
        y = (ys.gather(2, i) * (1 - f) + ys.gather(2, i + 1) * f).view_as(x)
        if self.local:
            a = torch.cat([y, x], 1)
            e1 = self.d1(a)
            e2 = self.d2(F.avg_pool2d(e1, 2))
            e3 = self.d3(F.avg_pool2d(e2, 2))
            u2 = self.u2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear"), e2], 1))
            u1 = self.u1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="bilinear"), e1], 1))
            y = y + self.last(u1)
        return y.clamp(0, 1)


def tens(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255)


def thumb_of(inp, raw):
    x = torch.cat([tens(inp), tens(raw)], 0)[None]
    return F.interpolate(x, (THUMB, THUMB), mode="area")[0]


def load_split():
    c = np.load(CACHE)
    n = int(c["n_tr"])
    perm = np.random.default_rng(0).permutation(n)
    return c, perm[120:], perm[:120]


@torch.no_grad()
def run_full(model, inp_u8, raw_u8):
    x = tens(inp_u8)[None].to(DEV)
    th = thumb_of(inp_u8, raw_u8)[None].to(DEV)
    return model(x, th)[0].permute(1, 2, 0).cpu().numpy()


def held_out_de(model, c, idx, src):
    model.eval()
    de = [full_reference(run_full(model, c[f"tr_{src}_{i}"], c[f"tr_raw_{i}"]), c[f"tr_ref_{i}"] / 255.0)["de2000"]
          for i in idx]
    model.train()
    return float(np.mean(de))


def train(name, local, epochs=120, bs=8, lr=1e-3):
    src = VARIANTS[name][0]
    c, tr, ho = load_split()
    inp = [c[f"tr_{src}_{i}"] for i in range(int(c["n_tr"]))]
    raw = [c[f"tr_raw_{i}"] for i in range(int(c["n_tr"]))]
    ref = [c[f"tr_ref_{i}"] for i in range(int(c["n_tr"]))]
    th = [thumb_of(a, b) for a, b in zip(inp, raw)]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TouchUp(local).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    start = held_out_de(model, c, ho, src)
    log = {"held_out_de2000_epoch0": start, "epochs": [], "args": dict(local=local, src=src, epochs=epochs, bs=bs, lr=lr)}
    best = start
    print(name, "params", sum(p.numel() for p in model.parameters()), "held-out ΔE at start", round(start, 3), flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        order = rng.permutation(tr)
        tot = 0.0
        for k in range(0, len(order) - bs + 1, bs):
            xs, ys, ts = [], [], []
            for i in order[k:k + bs]:
                h, w = inp[i].shape[:2]
                y0, x0 = rng.integers(0, h - CROP + 1), rng.integers(0, w - CROP + 1)
                a, b = tens(inp[i][y0:y0 + CROP, x0:x0 + CROP]), tens(ref[i][y0:y0 + CROP, x0:x0 + CROP])
                if rng.random() < 0.5:
                    a, b = a.flip(-1), b.flip(-1)
                xs.append(a), ys.append(b), ts.append(th[i])
            x, y, tt = torch.stack(xs).to(DEV), torch.stack(ys).to(DEV), torch.stack(ts).to(DEV)
            loss = (model(x, tt) - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if ep % 5 == 0:
            de = held_out_de(model, c, ho, src)
            log["epochs"].append({"epoch": ep, "train_l1": tot / (len(order) // bs), "held_out_de2000": de})
            if de < best:
                best = de
                torch.save({"state": model.state_dict(), "local": local, "src": src, "epoch": ep}, d / "best.pt")
            print(f"{name} ep {ep} L1 {tot / (len(order) // bs):.4f} held-out ΔE {de:.3f} best {best:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not (d / "best.pt").exists():  # never beat the identity
        torch.save({"state": model.state_dict(), "local": local, "src": src, "epoch": 0}, d / "best.pt")
    log["best_held_out_de2000"] = best
    json.dump(log, open(d / "train_log.json", "w"), indent=1)


def load_touchup(name):
    ck = torch.load(OUT / name / "best.pt", map_location=DEV)
    m = TouchUp(ck["local"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    return m, ck["src"], ck["epoch"]


def scores(o, ref):
    band = edge_band(ref)
    de = deltaE_ciede2000(rgb2lab(np.clip(o, 0, 1)), rgb2lab(ref))
    return {**full_reference(o, ref), "edge_de2000": float(de[band].mean()), "speck": speck_score(o), "haze": haze(o)}


@torch.no_grad()
def eval_charts(models):
    """Grey-chart error (degrees, lower = better) on the 2 Sea-thru chart frames, SfM range and DAv2 + 1 reading."""
    net = load(BASE)[0]
    d = np.load(ROOT / "data" / "seathru_small.npz")
    names = list(d["names"])
    out = {}
    for name in CHARTS:
        i = names.index(name)
        lin = np.clip(d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5), 0, 1)
        gt = d[f"gt_{i}"].copy()
        gt[~(gt > 0.1)] = np.nanmax(gt)
        dm = np.clip(d[f"dav2_metric-indoor_{i}"], 0.1, 30)
        raw_u8 = u8(linear_to_srgb(t(lin))[0].permute(1, 2, 0).cpu().numpy())
        for dn, z in {"SfM range": gt, "DAv2 + 1 range reading": dm * np.median(gt) / np.median(dm)}.items():
            I, zt = t(lin), t(z[..., None])
            p, _ = predict(net, I, zt)
            J, _ = safe_restore(I, zt, p, **CURRENT_RECIPE)
            d1b = linear_to_srgb(J)[0].permute(1, 2, 0).cpu().numpy()
            res = {"Day 1b": chart_error(srgb_to_linear(t(d1b))[0].permute(1, 2, 0).cpu().numpy(), CHARTS[name])}
            for m, (model, src, _) in models.items():
                o = run_full(model, u8(d1b) if src == "d1b" else raw_u8, raw_u8)
                res[m] = chart_error(srgb_to_linear(t(o))[0].permute(1, 2, 0).cpu().numpy(), CHARTS[name])
            out.setdefault(dn, {})[name] = res
    return out


def main_eval():
    c = np.load(CACHE)
    models = {m: load_touchup(m) for m in VARIANTS if (OUT / m / "best.pt").exists()}
    res = {"Day 1b": {}, **{m: {} for m in models}}
    keep, starved = {}, []
    for i in range(90):
        raw, ref, d1b = c[f"va_raw_{i}"], c[f"va_ref_{i}"] / 255.0, c[f"va_d1b_{i}"]
        starved.append(bool(np.median(raw[..., 0]) < 10))
        outs = {"Day 1b": d1b / 255.0}
        for m, (model, src, _) in models.items():
            outs[m] = run_full(model, d1b if src == "d1b" else raw, raw)
        for k, o in outs.items():
            for s, x in scores(o, ref).items():
                res[k].setdefault(s, []).append(float(x))
            if i < 25:
                keep[(i, k)] = o
    st = np.array(starved)
    mets = ("psnr", "ssim", "de2000", "edge_de2000", "speck", "haze")
    out = {"summary": {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in res.items()},
           "red_starved": {"n": int(st.sum()), "summary": {k: {s: float(np.mean(np.array(v)[st])) for s, v in d.items()}
                                                           for k, d in res.items()}},
           "ci_vs_day1b": {m: {s: boot_ci(res["Day 1b"][s], res[m][s]) for s in mets} for m in models},
           "n_better_de2000_vs_day1b": {m: int(np.sum(np.array(res[m]["de2000"]) < np.array(res["Day 1b"]["de2000"])))
                                        for m in models},
           "fins_val2": {k: {s: res[k][s][2] for s in mets} for k in res},
           "best_epoch": {m: v[2] for m, v in models.items()},
           "held_out_de2000": {m: json.load(open(OUT / m / "train_log.json"))["best_held_out_de2000"] for m in models},
           "per_image": res}
    out["charts"] = eval_charts(models)
    json.dump(out, open(OUT / "eval.json", "w"), indent=1)
    gallery(keep, c, res, list(models))
    s = out["summary"]
    for k in res:
        print(f"{k:34s} PSNR {s[k]['psnr']:.2f} ΔE {s[k]['de2000']:.2f} edgeΔE {s[k]['edge_de2000']:.2f} "
              f"specks {s[k]['speck']:.2f} haze {s[k]['haze']:.3f}")
    print(json.dumps({k: out[k] for k in ("ci_vs_day1b", "n_better_de2000_vs_day1b", "fins_val2", "charts",
                                          "held_out_de2000", "best_epoch")}, indent=0))


def gallery(keep, c, res, models, per=5, n=25):
    out = FIG / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["raw", "Day 1b"] + models + ["reference"]
    title = {"raw": "Raw photo", "Day 1b": "Physics (v1 + Day 1b)", "reference": "UIEB reference",
             "day3a_touchup_global": "Day 1b + touch-up (global)",
             "day3b_touchup_global_plus_local": "Day 1b + touch-up (global + local)",
             "day3c_control_touchup_on_raw": "Control: touch-up on RAW (no physics)"}
    for pg in range(n // per):
        fig, axs = plt.subplots(per, len(cols), figsize=(3.6 * len(cols), 2.75 * per))
        for k in range(per):
            i = pg * per + k
            for j, col in enumerate(cols):
                im = (c[f"va_raw_{i}"] / 255.0 if col == "raw" else c[f"va_ref_{i}"] / 255.0 if col == "reference"
                      else keep[(i, col)])
                ax = axs[k, j]
                no_axes(ax), ax.imshow(np.clip(im, 0, 1))
                if k == 0:
                    ax.set_title(title[col], fontsize=10)
                lab = f"val #{i}" if col == "raw" else (f"ΔE {res[col]['de2000'][i]:.1f}" if col in res else None)
                if lab:
                    ax.text(0.02, 0.03, lab, transform=ax.transAxes, fontsize=9, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
        fig.suptitle(f"UIEB val photos {pg * per}–{pg * per + per - 1} (first 25, fixed in advance): Day 3 touch-up",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"page_{pg + 1}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "cache":
        make_cache()
    elif cmd == "train":
        name = sys.argv[2]
        local = VARIANTS[name][1]
        if local is None:
            local = sys.argv[3] == "local"
        train(name, local)
    elif cmd == "eval":
        main_eval()
