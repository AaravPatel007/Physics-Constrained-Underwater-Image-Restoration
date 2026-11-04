"""Train SunDepthBetaNet with the baseline_v1_60ep_bigdata recipe (src/train.py), changing only the sunlight term.

Same as src/train.py: data (data/nyu_train_big.npz, last 288 frames = validation), on-the-fly synthetic water from
the same seeded generator (identical waters, images, flips, exposure and noise, batch for batch; verify_rng.py),
AdamW lr 5e-4 wd 1e-4, one-cycle schedule, batch 24, bf16 autocast, grad clip 1, checkpoint selection score
val_coef + 10 * val_img_l1.

Different:
  - network: 10 outputs, E computed from the predicted depth (model_sun.py)
  - loss:    coef_loss (unchanged; its E term now compares the *physics* E with the true E = sunlight consistency)
           + lambda_img * img_loss (unchanged)
           + lambda_depth * |log d_s_hat - log d_s|      (new: depth supervision, same log-L1 form as the coefficients)
  - labels:  d_s comes from sun_physics.sample_water_sun (the generator's own d_v, which the original discards)

  python sunlight_phy/train_sun.py --name sun_v1 --data train_big --nval 288 --epochs 60 --murky-keep 0.4
Outputs: sunlight_phy/runs/<name>/best.pt, log.json, config.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import train as T  # noqa: E402
from physics import restore  # noqa: E402

from model_sun import SunDepthBetaNet  # noqa: E402
from sun_physics import KAPPA0, sample_water_sun  # noqa: E402

# train.make_batch calls train.sample_water; give it the copy that also returns the depth labels. The extra keys
# ride along in the water dict ({**w, ...} in make_batch) and are ignored by add_water / coef_loss.
T.sample_water = sample_water_sun


def depth_loss(p, w):
    return (p["d_s"].clamp_min(1e-3).log() - w["d_s"].log()).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-img", type=float, default=1.0)
    ap.add_argument("--lambda-coef", type=float, default=1.0)
    ap.add_argument("--lambda-depth", type=float, default=1.0)
    ap.add_argument("--kappa0", type=float, default=KAPPA0)
    ap.add_argument("--no-detach-kd", action="store_true", help="let the sunlight term send gradients into beta_D")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="train")
    ap.add_argument("--nval", type=int, default=0)
    ap.add_argument("--murky-keep", type=float, default=1.0)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rgb, depth = T.load_split(a.data)
    nval = a.nval or max(64, len(rgb) // 10)
    tr_idx, va_idx = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    val_water = sample_water_sun(nval, dev, vgen)

    net = SunDepthBetaNet(kappa0=a.kappa0, detach_kd=not a.no_detach_kd).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(a), "model": "sun", "no_depth": False, "train_frames": len(tr_idx), "val_frames": nval,
           "params_M": sum(p.numel() for p in net.parameters()) / 1e6, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name()}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    log, best = [], float("inf")
    print(f"{a.name}: {len(tr_idx)} train / {nval} val scenes, {cfg['params_M']:.1f}M params", flush=True)

    for ep in range(a.epochs):
        net.train()
        t0, tl, parts = time.time(), [], {"coef": [], "img": [], "depth": []}
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for b in range(len(perm) // a.bs):
            bi = perm[b * a.bs:(b + 1) * a.bs]
            I, z, J, w = T.make_batch(rgb, depth, bi, gen, dev, murky_keep=a.murky_keep)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(I, z)
            Jhat = restore(I, z, p)
            lc, li, ld = T.coef_loss(p, w), T.img_loss(Jhat, J, z, w), depth_loss(p, w)
            loss = a.lambda_coef * lc + a.lambda_img * li + a.lambda_depth * ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
            for k, v in zip(parts, (lc, li, ld)):
                parts[k].append(v.item())

        # validation: fixed scenes x fixed waters, no augmentation (as in train.py)
        net.eval()
        vl = {"coef": [], "img_l1": [], "psnr": [], "depth_logl1": [], "depth_abs": [], "E_logl1": []}
        with torch.no_grad():
            for b in range(0, nval, a.bs):
                sl = slice(b, b + a.bs)
                wv = {k: v[sl] for k, v in val_water.items()}
                I, z, J, w = T.make_batch(rgb, depth, va_idx[sl], vgen, dev, augment=False, water=wv)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = net(I, z)
                Jhat = restore(I, z, p)
                vl["coef"].append(T.coef_loss(p, w).item())
                vl["depth_logl1"].append(depth_loss(p, w).item())
                vl["depth_abs"] += (p["d_s"] - w["d_s"]).abs().flatten().tolist()
                vl["E_logl1"].append((p["E"].log() - w["E"].log()).abs().mean().item())
                Jc = Jhat.clamp(0, 1)
                vl["img_l1"].append((Jc - J).abs().mean().item())
                mse = ((Jc - J) ** 2).mean((1, 2, 3))
                vl["psnr"] += (-10 * torch.log10(mse + 1e-10)).tolist()
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), **{f"train_{k}": float(np.mean(v)) for k, v in parts.items()},
               "time": time.time() - t0, "lr": opt.param_groups[0]["lr"],
               **{f"val_{k}": float(np.mean(v)) for k, v in vl.items()}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec["val_coef"] + rec["val_img_l1"] * 10  # same selection rule as the baseline
        if score < best:
            best = score
            torch.save({"state": net.state_dict(), "args": {**vars(a), "model": "sun", "no_depth": False},
                        "epoch": ep + 1}, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
