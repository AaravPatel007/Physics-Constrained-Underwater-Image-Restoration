"""Stage 2: a small black-box RefineNet closes the gap between BetaNet's restoration and the ground-truth clean image.

BetaNet stays frozen. RefineNet(Jhat, I, z) -> J_refined is trained with L1 to the true clean image J, on the same
synthetic water as train.py (the residual it learns is exactly "ground truth minus BetaNet output"). Output is
Jhat + small correction, so the 12 interpretable coefficients are still produced; only the image gets the extra step.

  python src/refine.py --base baseline_v1_60ep_bigdata --name refine_v1 --data train_big
  python src/refine.py --base baseline_v1_60ep_bigdata --name refine_v1 --test   # held-out NYU test rooms
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_synth import load, to_np
from metrics import full_reference
from model import RefineNet
from physics import add_water, restore, sample_water, srgb_to_linear
from train import ROOT, load_split, make_batch
from device import DEV, autocast


def psnr_t(a, b):
    return (-10 * torch.log10(((a.clamp(0, 1) - b) ** 2).mean((1, 2, 3)) + 1e-10)).tolist()


def stage1(base, I, z):
    with torch.no_grad(), autocast():
        return restore(I, z, base(I, z))


def test(base, ref, out):
    """Same protocol as eval_synth.py 'random': 654 test scenes, one water each (seed 2026), noise 0.005."""
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    gen, res = torch.Generator().manual_seed(7), {}
    with torch.no_grad():
        for b in range(0, len(rgb), 16):
            sl = slice(b, b + 16)
            J = srgb_to_linear(rgb[sl].to(DEV).permute(0, 3, 1, 2).float() / 255)
            z = depth[sl].to(DEV).float()[:, None]
            w = {k: v[sl] for k, v in water.items()}
            I = (add_water(J, z, w)[0] + 0.005 * torch.randn(J.shape, generator=gen).to(DEV)).clamp(0, 1)
            Jb = stage1(base, I, z)
            with autocast():
                Jr = ref(Jb.clamp(0, 1.5), I, z)
            outs = {"oracle (true coefs)": restore(I, z, w), "BetaNet": Jb, "BetaNet + RefineNet": Jr}
            for name, Jh in outs.items():
                for h, t in zip(to_np(Jh), to_np(J)):
                    for m, v in full_reference(h, t).items():
                        res.setdefault(name, {}).setdefault(m, []).append(v)
    s = {n: {m: float(np.mean(v)) for m, v in d.items()} for n, d in res.items()}
    json.dump(s, open(out / "test_eval.json", "w"), indent=1)
    for n, d in s.items():
        print(f"{n:22s} PSNR {d['psnr']:.2f}  SSIM {d['ssim']:.3f}  dE00 {d['de2000']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="results/<base>/best.pt, the frozen BetaNet")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="train_big")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nval", type=int, default=288, help="validation frames at the end; baseline_v1_60ep_bigdata used 288")
    ap.add_argument("--test", action="store_true", help="evaluate results/<name>/best.pt on the NYU test rooms and exit")
    a = ap.parse_args()

    base = load(a.base)[0].requires_grad_(False)
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    ref = RefineNet().to(DEV)
    if a.test:
        ref.load_state_dict(torch.load(out / "best.pt", map_location=DEV)["state"])
        return test(base, ref.eval(), out)

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rgb, depth = load_split(a.data)
    nval = a.nval or max(64, len(rgb) // 10)  # must equal the base's --nval, else val frames are rooms BetaNet trained on
    tr, va = torch.arange(len(rgb) - nval), torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    val_water = sample_water(nval, DEV, vgen)
    opt = torch.optim.AdamW(ref.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(tr) // a.bs), pct_start=0.05)
    print(f"{a.name}: RefineNet {sum(p.numel() for p in ref.parameters())/1e6:.2f}M params, {len(tr)} train / {nval} val", flush=True)
    log, best = [], -1.0
    for ep in range(a.epochs):
        ref.train(), base.eval()
        t0, tl = time.time(), []
        perm = tr[torch.randperm(len(tr), generator=gen)]
        for b in range(len(perm) // a.bs):
            I, z, J, _ = make_batch(rgb, depth, perm[b * a.bs:(b + 1) * a.bs], gen, DEV)
            Jb = stage1(base, I, z)
            with autocast():
                Jr = ref(Jb.clamp(0, 1.5), I, z)
            loss = (Jr - J).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            opt.step(), sched.step()
            tl.append(loss.item())
        ref.eval()
        pb, pr = [], []
        with torch.no_grad():
            for b in range(0, nval, a.bs):
                sl = slice(b, b + a.bs)
                I, z, J, _ = make_batch(rgb, depth, va[sl], vgen, DEV, augment=False, water={k: v[sl] for k, v in val_water.items()})
                Jb = stage1(base, I, z)
                with autocast():
                    Jr = ref(Jb.clamp(0, 1.5), I, z)
                pb += psnr_t(Jb, J)
                pr += psnr_t(Jr, J)
        rec = {"epoch": ep + 1, "train_l1": float(np.mean(tl)), "val_psnr_betanet": float(np.mean(pb)),
               "val_psnr_refined": float(np.mean(pr)), "time": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if rec["val_psnr_refined"] > best:
            best = rec["val_psnr_refined"]
            torch.save({"state": ref.state_dict(), "args": vars(a)}, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
