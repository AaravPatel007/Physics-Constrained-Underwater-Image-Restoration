"""Train BetaNet (coefficient estimator) or the black-box UNet on synthetic underwater NYU-v2.

Each batch: clean in-air linear image J + metric depth z  ->  random Jerlov-based water
            ->  I = J e^{-beta_D z} + B_inf (1 - e^{-beta_B z}), exposure jitter, sensor noise.

  python src/train.py --model beta --name beta_rgbd
  python src/train.py --model beta --no-depth --name beta_rgb
  python src/train.py --model beta --lambda-img 0 --name beta_coefonly
  python src/train.py --model unet --name unet_rgbd
  --stored-water: fixed waters per room from data/nyu_train_big_dav2.npz (prep_dav2_train.py); with
  --depth-input dav2 the network gets (and restores with) DAv2's depth guess instead of the true depth.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import BetaNet, UNetDirect
from physics import add_water, camera_flaws, restore, sample_water, srgb_to_linear

ROOT = Path(__file__).resolve().parents[1]


def load_split(name):
    d = np.load(ROOT / "data" / f"nyu_{name}.npz")
    return torch.from_numpy(d["rgb"]), torch.from_numpy(d["depth"])


def make_batch(rgb, depth, idx, gen, device, augment=True, water=None, murky_keep=1.0, low_red=0.0, camera=0.0):
    J = srgb_to_linear(rgb[idx].to(device).permute(0, 3, 1, 2).float() / 255)
    z = depth[idx].to(device).float()[:, None]
    n = len(idx)
    if augment:
        flip = torch.rand(n, generator=gen) < 0.5
        J[flip], z[flip] = J[flip].flip(-1), z[flip].flip(-1)
        z = z * torch.exp(torch.empty(n, 1, 1, 1).uniform_(math.log(0.7), math.log(2.0), generator=gen)).to(device)
    w = water if water is not None else sample_water(n, device, gen, murky_keep=murky_keep, low_red=low_red)
    k = torch.empty(n, 1).uniform_(0.7, 1.3, generator=gen).to(device) if augment else torch.ones(n, 1, device=device)
    w = {**w, "B_inf": w["B_inf"] * k}
    J = J * k[:, :, None, None]
    I, _, _ = add_water(J, z, w)
    sigma = torch.empty(n, 1, 1, 1).uniform_(0.002, 0.01, generator=gen).to(device)
    I = (I + sigma * torch.randn(I.shape, generator=gen).to(device)).clamp(0, 1)
    if camera > 0:
        I = camera_flaws(I, gen, p=camera)
    return I, z, J, w


def load_stored():
    d = np.load(ROOT / "data" / "nyu_train_big_dav2.npz")
    return {k: torch.from_numpy(d[k]) for k in ("zd", "k", "s", "nver", "beta_D", "beta_B", "B_inf", "E")}


def make_batch_stored(rgb, depth, st, idx, gen, device, augment=True, dav2=False):
    """Like make_batch, but the water / exposure / depth scale are one of the stored versions of each room.
    Returns I, z_in (what the network sees), z (true depth, for the loss weights), J, w."""
    n = len(idx)
    v = (torch.rand(n, generator=gen) * st["nver"][idx].float()).long() if augment else torch.zeros(n, dtype=torch.long)
    w = {k: st[k][idx, v].to(device) for k in ("beta_D", "beta_B", "B_inf", "E")}
    kk, ss = st["k"][idx, v].to(device), st["s"][idx, v].to(device)
    J = srgb_to_linear(rgb[idx].to(device).permute(0, 3, 1, 2).float() / 255) * kk[:, None, None, None]
    z = depth[idx].to(device).float()[:, None] * ss[:, None, None, None]
    zd = torch.nn.functional.interpolate(st["zd"][idx, v].to(device).float()[:, None], z.shape[-2:], mode="bilinear")
    if augment:
        flip = torch.rand(n, generator=gen) < 0.5
        J[flip], z[flip], zd[flip] = J[flip].flip(-1), z[flip].flip(-1), zd[flip].flip(-1)
    I, _, _ = add_water(J, z, w)
    sigma = torch.empty(n, 1, 1, 1).uniform_(0.002, 0.01, generator=gen).to(device)
    I = (I + sigma * torch.randn(I.shape, generator=gen).to(device)).clamp(0, 1)
    return I, (zd.clamp(0.1, 30) if dav2 else z), z, J, w


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
    ap.add_argument("--data", default="train", help="data/nyu_<data>.npz (train_big = 49 shards, prep_nyu_more.py)")
    ap.add_argument("--nval", type=int, default=0, help="validation frames at the end (0 = 10%%)")
    ap.add_argument("--murky-keep", type=float, default=1.0, help="fraction of near-hopeless waters kept")
    ap.add_argument("--low-red", type=float, default=0.0,
                    help="fraction of deep, red-starved waters (physics.sample_water); validation water is unchanged")
    ap.add_argument("--camera", type=float, default=0.0,
                    help="fraction of training images given camera flaws (shot noise, 8-bit, JPEG; physics.camera_flaws)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: constant LR, halved after 3 epochs without a better validation score, stop "
                         "after this many epochs without one (--epochs is then only the maximum). 0 = one-cycle schedule")
    ap.add_argument("--stored-water", action="store_true", help="fixed waters per room (prep_dav2_train.py)")
    ap.add_argument("--depth-input", choices=["true", "dav2"], default="true", help="with --stored-water")
    ap.add_argument("--init", default="", help="start from the weights of results/<init>/best.pt (e.g. baseline_v1_60ep_bigdata)")
    ap.add_argument("--resume-epoch", type=int, default=0,
                    help="continue from results/<name>/best.pt saved at this epoch (LR schedule fast-forwarded; "
                         "optimizer moments restart because best.pt holds weights only)")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rgb, depth = load_split(a.data)
    nval = a.nval or max(64, len(rgb) // 10)  # last frames = last shards = rooms not in train
    tr_idx, va_idx = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    val_water = sample_water(nval, dev, vgen)

    st = load_stored() if a.stored_water else None
    use_depth = not a.no_depth
    net = (BetaNet(use_depth) if a.model == "beta" else UNetDirect(use_depth)).to(dev)
    if a.init:
        net.load_state_dict(torch.load(ROOT / "results" / a.init / "best.pt", map_location=dev)["state"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.bs)
    if a.patience:
        sched, plateau = None, torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    else:
        sched, plateau = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05), None
    since_best = 0
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    if a.resume_epoch:
        net.load_state_dict(torch.load(out / "best.pt", map_location=dev)["state"])
        log = [r for r in json.load(open(out / "log.json")) if r["epoch"] <= a.resume_epoch]
        best = min(r.get("val_coef", 0) + r["val_img_l1"] * 10 for r in log)
        for _ in range(a.resume_epoch * (len(tr_idx) // a.bs) if sched else 0):
            sched.step()
    print(f"{a.name}: {len(tr_idx)} train / {nval} val scenes, {sum(p.numel() for p in net.parameters())/1e6:.1f}M params", flush=True)

    for ep in range(a.resume_epoch, a.epochs):
        net.train()
        t0, tl = time.time(), []
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for b in range(len(perm) // a.bs):
            bi = perm[b * a.bs:(b + 1) * a.bs]
            if st is not None:
                I, zin, z, J, w = make_batch_stored(rgb, depth, st, bi, gen, dev, dav2=a.depth_input == "dav2")
            else:
                I, z, J, w = make_batch(rgb, depth, bi, gen, dev, murky_keep=a.murky_keep, low_red=a.low_red,
                                        camera=a.camera)
                zin = z
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, zin)
            if a.model == "beta":
                Jhat = restore(I, zin, p)
                loss = a.lambda_coef * coef_loss(p, w) + a.lambda_img * img_loss(Jhat, J, z, w)
            else:
                loss = (p - J).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            tl.append(loss.item())

        # validation: fixed scenes x fixed waters, no augmentation
        net.eval()
        vl = {"coef": [], "img_l1": [], "psnr": []}
        with torch.no_grad():
            for b in range(0, nval, a.bs):
                sl = slice(b, b + a.bs)
                if st is not None:
                    I, zin, z, J, w = make_batch_stored(rgb, depth, st, va_idx[sl], vgen, dev, augment=False,
                                                        dav2=a.depth_input == "dav2")
                else:
                    wv = {k: v[sl] for k, v in val_water.items()}
                    I, z, J, w = make_batch(rgb, depth, va_idx[sl], vgen, dev, augment=False, water=wv)
                    zin = z
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = net(I, zin)
                Jhat = restore(I, zin, p) if a.model == "beta" else p
                if a.model == "beta":
                    vl["coef"].append(coef_loss(p, w).item())
                Jc = Jhat.clamp(0, 1)
                vl["img_l1"].append((Jc - J).abs().mean().item())
                mse = ((Jc - J) ** 2).mean((1, 2, 3))
                vl["psnr"] += (-10 * torch.log10(mse + 1e-10)).tolist()
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "time": time.time() - t0, "lr": opt.param_groups[0]["lr"],
               **{f"val_{k}": float(np.mean(v)) for k, v in vl.items() if v}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("val_coef", 0) + rec["val_img_l1"] * 10
        if score < best:
            best, since_best = score, 0
            torch.save({"state": net.state_dict(), "args": vars(a)}, out / "best.pt")
        else:
            since_best += 1
        json.dump(log, open(out / "log.json", "w"), indent=1)
        if plateau:
            plateau.step(score)
            if since_best >= a.patience:
                print(f"early stop: no better validation score for {a.patience} epochs (best at epoch "
                      f"{ep + 1 - since_best})", flush=True)
                break


if __name__ == "__main__":
    main()
