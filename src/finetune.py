"""Fine-tune BetaNet on real Sea-thru frames (photo + SfM metric range, no labels) with self-supervised
physics losses, while every step also keeps the supervised synthetic NYU loss so the coefficients stay
anchored to physical units.

Real-image losses (all on linear RGB, valid-depth pixels only; the cues classical Sea-thru uses):
  dark   : the darkest 1 % of pixels in each range bin lie on (or just above) the predicted backscatter
           curve B_inf (1 - e^{-beta_B z}). Asymmetric: predicting more backscatter than the darkest
           pixels is physically impossible, so it costs more than predicting less.
  range  : after removing the water (before dividing by E) a scene's average colour must not depend on
           distance. Equal-count range bins; chromaticity (and, weighted lower, brightness) of each bin's
           mean vs the image mean. This is the cue that pins beta_D.
  gray   : gray-world on the fully restored image (pins the illuminant E).
  dive   : frames of the same dive share one water, so coefficients are pulled to their dive's batch mean.
  teacher: (--w-teacher) L1 on log beta_D, beta_B, B_inf to the frozen starting model's prediction for the
           same crop. Without it the real-image losses are all satisfied by one "average water" and every
           dive collapses to the same coefficients (observed in ft_seathru); E stays free.

Optional UIEB stage (--w-uieb, after the Sea-thru stage): UIEB train photos (sRGB -> linear) with Depth
Anything V2 depth get the same dark / range / gray losses; --w-ref adds an L1 loss between the gray-world
balanced restoration and the UIEB reference image (subjective "preferred" images, so a low weight).

Split: dive D3 is held out entirely (unseen-water test), as are the 24 frames of data/seathru_small.npz
and every D1 frame within +-25 of the two colour-chart frames (same scene as the chart test).

  python src/finetune.py --name ft_seathru [--w-dark 1 --w-range 1 --w-gray 0.5 --w-dive 0.5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_synth import load
from physics import gray_world, linear_to_srgb, remove_water, restore, sample_water, srgb_to_linear
from train import coef_loss, img_loss, load_split, make_batch

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda"
HOLDOUT_SITES = {"D3"}
CHART_FRAMES = {("D1", 3272), ("D1", 3425)}


def frame_number(stem):
    return int("".join(ch for ch in stem if ch.isdigit()))


def split_of(site, stem, eval_names):
    if site in HOLDOUT_SITES or f"{site}/{stem}" in eval_names:
        return "test"
    if any(site == s and abs(frame_number(stem) - n) <= 25 for s, n in CHART_FRAMES):
        return "test"
    return "train"


def load_real(split):
    eval_names = set(np.load(ROOT / "data" / "seathru_small.npz")["names"].tolist())
    frames = []
    for f in sorted((ROOT / "data" / "seathru_train").glob("D*/*.npz")):
        site, stem = f.parent.name, f.stem
        if split_of(site, stem, eval_names) != split:
            continue
        d = np.load(f)
        lin, z = d["lin"].astype(np.float32), d["z"].astype(np.float32)
        valid = z > 0.1
        if valid.mean() < 0.3:
            continue
        lin = np.clip(lin / np.percentile(lin, 99.5), 0, 1)
        zf = np.where(valid, z, z[valid].max())       # network input: no-depth pixels = far (as eval_real)
        frames.append({"name": f"{site}/{stem}", "site": site, "I": torch.from_numpy(lin).permute(2, 0, 1),
                       "z": torch.from_numpy(zf)[None], "m": torch.from_numpy(valid)[None]})
    return frames


def load_uieb():
    d = np.load(ROOT / "data" / "uieb_train.npz")
    return [(torch.from_numpy(d[f"raw_{i}"]), torch.from_numpy(d[f"ref_{i}"]), torch.from_numpy(d[f"z_{i}"].astype(np.float32)))
            for i in range(int(d["n"])) if min(d[f"z_{i}"].shape) >= 192]  # a few tiny images cannot be cropped


def uieb_batch(data, n, gen, s=192):
    I, R, z = [], [], []
    for j in torch.randint(len(data), (n,), generator=gen).tolist():
        a, b, d = data[j]
        H, W = d.shape
        y, x = torch.randint(H - s + 1, (1,), generator=gen).item(), torch.randint(W - s + 1, (1,), generator=gen).item()
        I.append(a[y:y + s, x:x + s]), R.append(b[y:y + s, x:x + s]), z.append(d[y:y + s, x:x + s])
    I = srgb_to_linear(torch.stack(I).to(DEV).permute(0, 3, 1, 2).float() / 255)
    R = torch.stack(R).to(DEV).permute(0, 3, 1, 2).float() / 255
    z = torch.stack(z).to(DEV)[:, None]
    return I, R, z


def real_batch(frames, by_site, n_sites, per_site, gen, h=240, w=320):
    """per_site random crops from each of n_sites random dives (so the dive loss has pairs)."""
    sites = list(by_site)
    pick = [sites[i] for i in torch.randperm(len(sites), generator=gen)[:n_sites]]
    I, z, m, site_id = [], [], [], []
    for k, s in enumerate(pick):
        for j in torch.randint(len(by_site[s]), (per_site,), generator=gen).tolist():
            f = frames[by_site[s][j]]
            H, W = f["I"].shape[-2:]
            y, x = (torch.randint(H - h + 1, (1,), generator=gen).item(), torch.randint(W - w + 1, (1,), generator=gen).item())
            crop = [t[:, y:y + h, x:x + w] for t in (f["I"], f["z"], f["m"])]
            if torch.rand(1, generator=gen).item() < 0.5:
                crop = [t.flip(-1) for t in crop]
            I.append(crop[0]), z.append(crop[1]), m.append(crop[2]), site_id.append(k)
    return (torch.stack(I).to(DEV), torch.stack(z).to(DEV), torch.stack(m).to(DEV),
            torch.tensor(site_id, device=DEV))


def range_bins(z, m, nb):
    """Equal-count range bins over valid pixels -> list of boolean masks, each (P,)."""
    zv = z[m]
    edges = torch.quantile(zv, torch.linspace(0, 1, nb + 1, device=z.device))
    return zv, [(zv >= edges[i]) & (zv <= edges[i + 1]) for i in range(nb)]


def selfsup_losses(I, z, m, p, nb=10, dark_q=0.01):
    """Per-image physics losses, averaged over the batch."""
    out = {"dark": [], "range": [], "gray": []}
    bd, bb, binf, E = (p[k].float() for k in ("beta_D", "beta_B", "B_inf", "E"))
    for i in range(len(I)):
        mi = m[i, 0]
        if mi.sum() < 500:
            continue
        Iv = I[i][:, mi].T                                           # (P, 3)
        zv, bins = range_bins(z[i, 0], mi, nb)
        back = binf[i] * (1 - torch.exp(-bb[i] * zv[:, None]))       # (P, 3)
        gain = torch.exp((bd[i] * zv[:, None]).clamp(max=float(np.log(50.0))))
        Jw = ((Iv - back) * gain).clamp(-0.5, 2.0)                   # water removed, not yet / E
        dk, chrom, bright, w = [], [], [], []
        for b in bins:
            if b.sum() < 50:
                continue
            Ib, Bb = Iv[b], back[b]
            k = max(5, int(dark_q * len(Ib)))
            idx = Ib.sum(1).topk(k, largest=False).indices
            r = Ib[idx] - Bb[idx]                                    # >= 0 if backscatter is right
            dk.append((torch.relu(-r) + 0.3 * torch.relu(r)).mean())
            mean = Jw[b].clamp_min(0).mean(0).clamp_min(0.02)   # floor: far bins can be backscatter only
            chrom.append(mean.log() - mean.log().mean())
            bright.append(mean.log().mean())
            w.append(b.sum().float())
        if len(chrom) < 3:
            continue
        chrom, bright, w = torch.stack(chrom), torch.stack(bright), torch.stack(w)
        w = w / w.sum()
        c0, b0 = (w[:, None] * chrom).sum(0), (w * bright).sum()
        out["dark"].append(torch.stack(dk).mean())
        out["range"].append((w[:, None] * (chrom - c0).abs()).sum() + 0.25 * (w * (bright - b0).abs()).sum())
        g = (Jw / E[i]).clamp(0, 2).mean(0).clamp_min(0.02).log()
        out["gray"].append((g - g.mean()).abs().mean())
    return {k: torch.stack(v).mean() if v else torch.zeros((), device=I.device) for k, v in out.items()}


def dive_loss(p, site_id):
    loss = 0.0
    for key in ("beta_D", "beta_B", "B_inf"):
        v = p[key].float().log()
        for s in site_id.unique():
            vs = v[site_id == s]
            if len(vs) > 1:
                loss = loss + (vs - vs.mean(0, keepdim=True)).abs().mean()
    return loss / max(1, len(site_id.unique()))


@torch.no_grad()
def evaluate_real(net, frames):
    """Self-supervised losses on full held-out frames (240-high copies) + coefficient spread per dive."""
    net.eval()
    agg, coefs = {"dark": [], "range": [], "gray": []}, {}
    for f in frames:
        I, z, m = (t[None].to(DEV) for t in (f["I"], f["z"], f["m"]))
        H, W = I.shape[-2:]
        size = (240, int(round(W * 240 / H)))
        I, z = (torch.nn.functional.interpolate(t, size, mode="area") for t in (I, z))
        m = torch.nn.functional.interpolate(m.float(), size, mode="nearest") > 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(I, z)
        for k, v in selfsup_losses(I, z, m, p).items():
            agg[k].append(v.item())
        coefs.setdefault(f["site"], []).append(torch.cat([p["beta_D"], p["beta_B"]], 1).float()[0].cpu().numpy())
    spread = {s: float(np.mean(np.std(np.log(np.array(c)), 0))) for s, c in coefs.items()}
    return {**{k: float(np.mean(v)) for k, v in agg.items()}, "dive_log_std": spread,
            "beta_D_mean": {s: np.exp(np.log(np.array(c))[:, :3].mean(0)).round(3).tolist() for s, c in coefs.items()}}


@torch.no_grad()
def evaluate_synth(net, rgb, depth, idx, water):
    net.eval()
    gen = torch.Generator().manual_seed(123)
    cl, psnr = [], []
    for b in range(0, len(idx), 16):
        sl = slice(b, b + 16)
        I, z, J, w = make_batch(rgb, depth, idx[sl], gen, DEV, augment=False, water={k: v[sl] for k, v in water.items()})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(I, z)
        cl.append(coef_loss(p, w).item())
        mse = ((restore(I, z, p).clamp(0, 1) - J) ** 2).mean((1, 2, 3))
        psnr += (-10 * torch.log10(mse + 1e-10)).tolist()
    return {"coef": float(np.mean(cl)), "psnr": float(np.mean(psnr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="beta_rgbd")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs-synth", type=int, default=12)
    ap.add_argument("--sites-per-batch", type=int, default=2)
    ap.add_argument("--per-site", type=int, default=4)
    ap.add_argument("--w-synth", type=float, default=1.0)
    ap.add_argument("--w-dark", type=float, default=1.0)
    ap.add_argument("--w-range", type=float, default=1.0)
    ap.add_argument("--w-gray", type=float, default=0.5)
    ap.add_argument("--w-dive", type=float, default=0.5)
    ap.add_argument("--w-teacher", type=float, default=0.0)
    ap.add_argument("--w-uieb", type=float, default=0.0)
    ap.add_argument("--w-ref", type=float, default=0.0)
    ap.add_argument("--bs-uieb", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, _ = load(a.init)
    teacher = load(a.init)[0].eval() if a.w_teacher else None
    train, test = load_real("train"), load_real("test")
    by_site = {}
    for i, f in enumerate(train):
        by_site.setdefault(f["site"], []).append(i)
    print(f"{a.name}: real train {len(train)} frames {({s: len(v) for s, v in by_site.items()})}, "
          f"real test {len(test)}", flush=True)

    uieb = load_uieb() if a.w_uieb or a.w_ref else None
    rgb, depth = load_split("train")
    nval = max(64, len(rgb) // 10)
    tr_idx, va_idx = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    val_water = sample_water(nval, DEV, torch.Generator().manual_seed(123))

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    log = [{"step": 0, "synth": evaluate_synth(net, rgb, depth, va_idx, val_water), "real_test": evaluate_real(net, test)}]
    print(json.dumps(log[-1]), flush=True)
    t0, run = time.time(), {}
    for step in range(1, a.steps + 1):
        net.train()
        for mod in net.modules():  # frozen BN statistics: real batches must not shift the synthetic ones
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        I_s, z_s, J_s, w_s = make_batch(rgb, depth, tr_idx[torch.randint(len(tr_idx), (a.bs_synth,), generator=gen)], gen, DEV)
        I_r, z_r, m_r, site_id = real_batch(train, by_site, a.sites_per_batch, a.per_site, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_s, p_r = net(I_s, z_s), net(I_r, z_r)
        L = {"synth": coef_loss(p_s, w_s) + img_loss(restore(I_s, z_s, p_s), J_s, z_s, w_s),
             **selfsup_losses(I_r, z_r, m_r, p_r), "dive": dive_loss(p_r, site_id)}
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_t = teacher(I_r, z_r)
            L["teacher"] = sum((p_r[k].float().log() - p_t[k].float().log()).abs().mean() for k in ("beta_D", "beta_B", "B_inf"))
        loss = sum(getattr(a, f"w_{k}") * v for k, v in L.items())
        if uieb is not None:
            I_u, R_u, z_u = uieb_batch(uieb, a.bs_uieb, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p_u = net(I_u, z_u)
            Lu = selfsup_losses(I_u, z_u, torch.ones_like(z_u, dtype=torch.bool), p_u)
            L["uieb"] = Lu["dark"] + Lu["range"] + 0.5 * Lu["gray"]
            L["ref"] = (linear_to_srgb(gray_world(remove_water(I_u, z_u, p_u).clamp(0, 1.5))) - R_u).abs().mean()
            loss = loss + a.w_uieb * L["uieb"] + a.w_ref * L["ref"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in L.items():
            run.setdefault(k, []).append(float(v.detach()) if torch.is_tensor(v) else float(v))
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "time": time.time() - t0, "train": {k: float(np.mean(v)) for k, v in run.items()},
                   "synth": evaluate_synth(net, rgb, depth, va_idx, val_water), "real_test": evaluate_real(net, test)}
            run = {}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(out / "log.json", "w"), indent=1)
    # keep the same checkpoint format as train.py so every eval script can load it
    torch.save({"state": net.state_dict(), "args": {**torch.load(ROOT / "results" / a.init / "best.pt", map_location="cpu")["args"],
                                                    "finetune": vars(a)}}, out / "best.pt")


if __name__ == "__main__":
    main()
