"""Train BetaNet (coefficient estimator) or the black-box UNet on synthetic underwater NYU-v2.

Each batch: clean in-air linear image J + metric depth z  ->  random Jerlov-based water
            ->  I = J e^{-beta_D z} + B_inf (1 - e^{-beta_B z}), exposure jitter, sensor noise.

  python src/train.py --model beta --name beta_rgbd
  python src/train.py --model beta --no-depth --name beta_rgb
  python src/train.py --model beta --lambda-img 0 --name beta_coefonly
  python src/train.py --model unet --name unet_rgbd
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import BetaNet, UNetDirect
from physics import add_water, restore, sample_water, srgb_to_linear

ROOT = Path(__file__).resolve().parents[1]


def load_split(name):
    d = np.load(ROOT / "data" / f"nyu_{name}.npz")
    return torch.from_numpy(d["rgb"]), torch.from_numpy(d["depth"])


def make_batch(rgb, depth, idx, gen, device, augment=True, water=None):
    J = srgb_to_linear(rgb[idx].to(device).permute(0, 3, 1, 2).float() / 255)
    z = depth[idx].to(device).float()[:, None]
    n = len(idx)
    if augment:
        flip = torch.rand(n, generator=gen) < 0.5
        J[flip], z[flip] = J[flip].flip(-1), z[flip].flip(-1)
        z = z * torch.exp(torch.empty(n, 1, 1, 1).uniform_(math.log(0.7), math.log(2.0), generator=gen)).to(device)
    w = water if water is not None else sample_water(n, device, gen)
    k = torch.empty(n, 1).uniform_(0.7, 1.3, generator=gen).to(device) if augment else torch.ones(n, 1, device=device)
    w = {**w, "B_inf": w["B_inf"] * k}
    J = J * k[:, :, None, None]
    I, _, _ = add_water(J, z, w)
    sigma = torch.empty(n, 1, 1, 1).uniform_(0.002, 0.01, generator=gen).to(device)
    I = (I + sigma * torch.randn(I.shape, generator=gen).to(device)).clamp(0, 1)
    return I, z, J, w


def coef_loss(p, w):
    return ((p["beta_D"].log() - w["beta_D"].log()).abs().mean()
            + (p["beta_B"].log() - w["beta_B"].log()).abs().mean()
            + 3 * (p["B_inf"] - w["B_inf"]).abs().mean()
            + (p["E"].log() - w["E"].log()).abs().mean())


def img_loss(Jhat, J, z, w):
    # weight pixels by how much signal actually survives the water (true transmission x illuminant)
    t = (w["E"][:, :, None, None] * torch.exp(-w["beta_D"][:, :, None, None] * z)).mean(1, keepdim=True)
    return (t * (Jhat.clamp(0, 1.5) - J).abs()).sum() / (3 * t.sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["beta", "unet"], default="beta")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--lambda-img", type=float, default=1.0)
    ap.add_argument("--lambda-coef", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rgb, depth = load_split("train")
    nval = max(64, len(rgb) // 10)          # last frames = last shard = rooms not in train
    tr_idx, va_idx = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    val_water = sample_water(nval, dev, vgen)

    use_depth = not a.no_depth
    net = (BetaNet(use_depth) if a.model == "beta" else UNetDirect(use_depth)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    print(f"{a.name}: {len(tr_idx)} train / {nval} val scenes, {sum(p.numel() for p in net.parameters())/1e6:.1f}M params", flush=True)

    for ep in range(a.epochs):
        net.train()
        t0, tl = time.time(), []
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for b in range(len(perm) // a.bs):
            I, z, J, w = make_batch(rgb, depth, perm[b * a.bs:(b + 1) * a.bs], gen, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            if a.model == "beta":
                Jhat = restore(I, z, p)
                loss = a.lambda_coef * coef_loss(p, w) + a.lambda_img * img_loss(Jhat, J, z, w)
            else:
                loss = (p - J).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())

        # validation: fixed scenes x fixed waters, no augmentation
        net.eval()
        vl = {"coef": [], "img_l1": [], "psnr": []}
        with torch.no_grad():
            for b in range(0, nval, a.bs):
                sl = slice(b, b + a.bs)
                wv = {k: v[sl] for k, v in val_water.items()}
                I, z, J, w = make_batch(rgb, depth, va_idx[sl], vgen, dev, augment=False, water=wv)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = net(I, z)
                Jhat = restore(I, z, p) if a.model == "beta" else p
                if a.model == "beta":
                    vl["coef"].append(coef_loss(p, w).item())
                Jc = Jhat.clamp(0, 1)
                vl["img_l1"].append((Jc - J).abs().mean().item())
                mse = ((Jc - J) ** 2).mean((1, 2, 3))
                vl["psnr"] += (-10 * torch.log10(mse + 1e-10)).tolist()
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "time": time.time() - t0,
               **{f"val_{k}": float(np.mean(v)) for k, v in vl.items() if v}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("val_coef", 0) + rec["val_img_l1"] * 10
        if score < best:
            best = score
            torch.save({"state": net.state_dict(), "args": vars(a)}, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
