"""Train SunDepthBetaNet v2 (16 outputs) on the v2 water, with the baseline_v1_60ep_bigdata recipe.

Recipe as src/train.py and sunlight_phy/train_sun.py: data/nyu_train_big.npz (last 288 frames = validation),
on-the-fly water from the seeded generator, AdamW lr 5e-4 wd 1e-4, one-cycle, batch 24, bf16, grad clip 1,
60 epochs, murky_keep 0.4, best checkpoint by val_coef + 10 * val_img_l1 (val_coef = train.coef_loss, the same
4-term number for every model).

Loss (all weights 1; no tuning):
  L = L_optical + L_E0 + L_Kd + L_ds + L_image + L_E
  L_optical  |log b_D - log b_D*| + |log b_B - log b_B*| + 3 |B_inf - B_inf*|    (train.coef_loss without E)
  L_E0       |r - mean_c(r)| with r = log E0_hat - log E0*                       (chromaticity only: scale-free)
  L_Kd       |log K_d - log K_d*|
  L_ds       |log d_s - log d_s*|                                                (as v1)
  L_image    train.img_loss (unchanged)
  L_E        |log E - log E*| of the computed E                                  (train.coef_loss's E term)

  python sunlight_depth_v2/train_v2.py --name sun_v2 --data train_big --nval 288 --epochs 60 --murky-keep 0.4
Outputs: sunlight_depth_v2/runs/<name>/best.pt, log.json, config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]
import train as T  # noqa: E402
from physics import restore  # noqa: E402

from model_v2 import SunDepthBetaNetV2  # noqa: E402
from water_v2 import sample_water_v2  # noqa: E402

T.sample_water = sample_water_v2  # train.make_batch draws its water from here; extra label keys ride along


def logl1(a, b):
    return (a.clamp_min(1e-6).log() - b.log()).abs().mean()


def losses(p, w):
    r = p["E0"].clamp_min(1e-6).log() - w["E0"].log()
    return {"optical": logl1(p["beta_D"], w["beta_D"]) + logl1(p["beta_B"], w["beta_B"])
                       + 3 * (p["B_inf"] - w["B_inf"]).abs().mean(),
            "E0": (r - r.mean(1, keepdim=True)).abs().mean(),
            "Kd": logl1(p["K_d"], w["K_d"]),
            "ds": logl1(p["d_s"], w["d_s"]),
            "E": logl1(p["E"], w["E"])}


def angle_deg(a, b):
    c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser()
    for k in ("optical", "E0", "Kd", "ds", "img", "E"):
        ap.add_argument(f"--lambda-{k}", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="train_big")
    ap.add_argument("--nval", type=int, default=288)
    ap.add_argument("--murky-keep", type=float, default=0.4)
    a = ap.parse_args()
    lam = {k: getattr(a, f"lambda_{k}") for k in ("optical", "E0", "Kd", "ds", "img", "E")}

    dev = "cuda"
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rgb, depth = T.load_split(a.data)
    nval = a.nval or max(64, len(rgb) // 10)
    tr_idx, va_idx = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    val_water = sample_water_v2(nval, dev, vgen)

    net = SunDepthBetaNetV2().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(tr_idx) // a.bs), pct_start=0.05)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(a), "model": "sun_v2", "water": "v2", "train_frames": len(tr_idx), "val_frames": nval,
           "params_M": sum(p.numel() for p in net.parameters()) / 1e6, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name()}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    log, best = [], float("inf")
    print(f"{a.name}: {len(tr_idx)} train / {nval} val scenes, {cfg['params_M']:.2f}M params", flush=True)

    for ep in range(a.epochs):
        net.train()
        t0, tl, parts = time.time(), [], {k: [] for k in ("optical", "E0", "Kd", "ds", "img", "E")}
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for b in range(len(perm) // a.bs):
            I, z, J, w = T.make_batch(rgb, depth, perm[b * a.bs:(b + 1) * a.bs], gen, dev, murky_keep=a.murky_keep)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            L = losses(p, w) | {"img": T.img_loss(restore(I, z, p), J, z, w)}
            loss = sum(lam[k] * v for k, v in L.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
            for k, v in L.items():
                parts[k].append(v.item())

        net.eval()
        vl = {k: [] for k in ("coef", "img_l1", "psnr", "depth_abs", "Kd_logl1", "E0_angle", "E_angle", "E_logl1")}
        with torch.no_grad():
            for b in range(0, nval, a.bs):
                sl = slice(b, b + a.bs)
                I, z, J, w = T.make_batch(rgb, depth, va_idx[sl], vgen, dev, augment=False,
                                          water={k: v[sl] for k, v in val_water.items()})
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = net(I, z)
                p = {k: v.float() for k, v in p.items()}
                Jc = restore(I, z, p).clamp(0, 1)
                vl["coef"].append(T.coef_loss(p, w).item())
                vl["img_l1"].append((Jc - J).abs().mean().item())
                vl["psnr"] += (-10 * torch.log10(((Jc - J) ** 2).mean((1, 2, 3)) + 1e-10)).tolist()
                vl["depth_abs"] += (p["d_s"] - w["d_s"]).abs().flatten().tolist()
                vl["Kd_logl1"].append(logl1(p["K_d"], w["K_d"]).item())
                vl["E0_angle"] += angle_deg(p["E0"], w["E0"]).tolist()
                vl["E_angle"] += angle_deg(p["E"], w["E"]).tolist()
                vl["E_logl1"].append(logl1(p["E"], w["E"]).item())
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), **{f"train_{k}": float(np.mean(v)) for k, v in parts.items()},
               "time": time.time() - t0, "lr": opt.param_groups[0]["lr"], **{f"val_{k}": float(np.mean(v)) for k, v in vl.items()}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec["val_coef"] + rec["val_img_l1"] * 10
        if score < best:
            best = score
            torch.save({"state": net.state_dict(), "args": {**vars(a), "model": "sun_v2"}, "epoch": ep + 1}, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
