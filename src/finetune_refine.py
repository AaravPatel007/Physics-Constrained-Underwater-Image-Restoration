"""Fine-tune RefineNet on real UIEB photos so its correction transfers off synthetic water.

The synthetic-only RefineNet (refine.py) regresses on real dives: it learned to darken/warm the synthetic
restorations, which is the wrong correction on real photos (see results/refine_v1/uieb_eval.json). This
mirrors the De-water BridgeNet failure. Fix: adapt it on real UIEB pairs with the two guards that
finetune.py (BetaNet on Sea-thru) found necessary -

  synth  : every step also runs a synthetic NYU batch with the ORIGINAL objective (|RefineNet(restore) - J|),
           so the synthetic gain is not forgotten. Analog of finetune.py --w-synth.
  anchor : on real photos, an L1 pull of RefineNet's output toward BetaNet's restoration, so it can only
           nudge the physics result, not drift to one dark/brown "average" look. Analog of --w-teacher.
  real   : L1 to the UIEB reference (subjective "preferred" images, not ground truth, so a modest weight).

Starts from results/refine_v1/best.pt. BetaNet stays frozen (12 coefficients unchanged). Split: the 800
uieb_train photos are shuffled once (seed 0); the last 160 are a held-out real val set, never trained on,
and the synthetic-only model is scored on the SAME 160 for a fair before/after.

  python src/finetune_refine.py --base baseline_v1_60ep_bigdata --init refine_v1 --name refine_ft
  python src/finetune_refine.py --base baseline_v1_60ep_bigdata --name refine_ft --test   # eval both domains

--mix adds more real pairs (prep_extra_pairs.py) at the SAME number of steps per epoch: each step draws its real
photo from a source picked with the given weights, e.g. --mix uieb:0.4,lsui:0.4,euvp:0.2. Selection is still on
the 160 held-out UIEB photos, so runs with and without --mix are chosen the same way.

--day1b feeds RefineNet the BetaNet + Day 1b restoration (safe_restore, CURRENT_RECIPE) instead of plain restore(),
and anchors to it. Start from --init refine_ft: stage-1 weights expect plain BetaNet output. The flag is saved in
best.pt, so --test and stress_test.py pick the right input; in the logs "BetaNet" is then the Day 1b input.
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
from safe_restore import CURRENT_RECIPE, safe_restore

N_VAL_REAL = 160  # held-out real photos (never trained on); rest of the 800 are train


def real_split():
    """Fixed shuffle of the 800 uieb_train indices -> (train_idx, val_idx). Seed 0, so train/test agree."""
    d = np.load(ROOT / "data" / "uieb_train.npz")
    idx = np.random.default_rng(0).permutation(int(d["n"]))
    return d, idx[:-N_VAL_REAL], idx[-N_VAL_REAL:]


def extra_split(name):
    """data/<name>_pairs.npz is stored in random order; the last TEST_SPLIT[name] are held out."""
    from prep_extra_pairs import TEST_SPLIT
    d = np.load(ROOT / "data" / f"{name}_pairs.npz")
    n = int(d["n"])
    return d, np.arange(n - TEST_SPLIT[name]), np.arange(n - TEST_SPLIT[name], n)


def load_real(d, i):
    I = srgb_to_linear(torch.from_numpy(d[f"raw_{i}"] / 255.0).permute(2, 0, 1)[None].float().to(DEV))
    z = torch.from_numpy(np.clip(d[f"z_{i}"].astype(np.float32), 0.1, 30)[None, None]).to(DEV)
    ref = torch.from_numpy(d[f"ref_{i}"] / 255.0).permute(2, 0, 1)[None].float().to(DEV)
    return I, z, ref


def betanet_restore(base, I, z):
    with torch.no_grad(), autocast():
        return restore(I, z, base(I, z)).clamp(0, 1.5)


def plain_restore(base, I, z):
    return restore(I, z, base(I, z)).clamp(0, 1.5)


@torch.no_grad()
def day1b_restore(base, I, z):
    """BetaNet + Day 1b, one image at a time (safe_restore's fog cap and white balance are per image)."""
    with autocast():
        p = {k: v.float() for k, v in base(I, z).items()}
    return torch.cat([safe_restore(I[i:i + 1], z[i:i + 1], {k: v[i:i + 1] for k, v in p.items()}, **CURRENT_RECIPE)[0]
                      for i in range(len(I))]).clamp(0, 1.5)


@torch.no_grad()
def eval_real(base, ref_net, d, val_idx, stage1=betanet_restore):
    res = {"BetaNet": {}, "RefineNet": {}}
    for i in val_idx:
        I, z, refimg = load_real(d, int(i))
        Jb = stage1(base, I, z)
        with autocast():
            Jr = ref_net(Jb, I, z)
        tgt = refimg[0].permute(1, 2, 0).cpu().numpy()  # reference is already sRGB [0,1]; to_np(J) is too
        for name, J in [("BetaNet", Jb), ("RefineNet", Jr)]:
            for m, v in full_reference(to_np(J)[0], tgt).items():
                res[name].setdefault(m, []).append(float(v))
    return {n: {m: float(np.mean(v)) for m, v in dd.items()} for n, dd in res.items()}


@torch.no_grad()
def eval_synth(base, ref_net, rgb, depth, nval=288, stage1=plain_restore):
    """Synthetic val PSNR, same 288 held-out frames as refine.py, to confirm the gain survives."""
    va = torch.arange(len(rgb) - nval, len(rgb))
    vgen = torch.Generator().manual_seed(123)
    water = sample_water(nval, DEV, vgen)
    pb, pr = [], []
    for b in range(0, nval, 24):
        sl = slice(b, b + 24)
        I, z, J, _ = make_batch(rgb, depth, va[sl], vgen, DEV, augment=False, water={k: v[sl] for k, v in water.items()})
        Jb = stage1(base, I, z)
        with autocast():
            Jr = ref_net(Jb, I, z)
        for out, acc in [(Jb, pb), (Jr, pr)]:
            acc += (-10 * torch.log10(((out.clamp(0, 1) - J) ** 2).mean((1, 2, 3)) + 1e-10)).tolist()
    return {"betanet": float(np.mean(pb)), "refined": float(np.mean(pr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--init", default="refine_v1", help="synthetic-pretrained RefineNet to start from")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)        # lower than scratch (1e-3): gentle adaptation
    ap.add_argument("--bs-synth", type=int, default=12)
    ap.add_argument("--w-synth", type=float, default=1.0)    # keep the synthetic gain
    ap.add_argument("--w-real", type=float, default=1.0)     # match the UIEB reference
    ap.add_argument("--w-anchor", type=float, default=0.3)   # stay near BetaNet on real (no drift/collapse)
    ap.add_argument("--mix", default="uieb:1", help="real sources and weights, e.g. uieb:0.4,lsui:0.4,euvp:0.2")
    ap.add_argument("--day1b", action="store_true", help="RefineNet input and anchor = BetaNet + Day 1b")
    ap.add_argument("--last", action="store_true", help="keep the last epoch: no selection on UIEB, needed when UIEB is held out")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    base = load(a.base)[0].requires_grad_(False).eval()
    ref_net = RefineNet().to(DEV)
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    d, tr_real, val_real = real_split()
    rgb, depth = load_split("train_big")

    if a.test:
        ck = torch.load(out / "best.pt", map_location=DEV)
        ref_net.load_state_dict(ck["state"])
        ref_net.eval()
        d1b = ck["args"].get("day1b", False)
        r = {"uieb_val_real": eval_real(base, ref_net, d, val_real, day1b_restore if d1b else betanet_restore),
             "synth_val": eval_synth(base, ref_net, rgb, depth, stage1=day1b_restore if d1b else plain_restore),
             "n_real_val": len(val_real), "input": "day1b" if d1b else "betanet",
             "note": "uieb scored vs reference (not ground truth); same 160 held-out photos as the before run"}
        json.dump(r, open(out / "ft_eval.json", "w"), indent=1)
        print(json.dumps(r, indent=1))
        return

    s1_real, s1_synth = (day1b_restore, day1b_restore) if a.day1b else (betanet_restore, plain_restore)
    # start from the synthetic-pretrained RefineNet
    ref_net.load_state_dict(torch.load(ROOT / "results" / a.init / "best.pt", map_location=DEV)["state"])
    tr_idx = torch.arange(len(rgb) - 288)  # synthetic train frames (last 288 are val)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(ref_net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr_real), pct_start=0.1)
    mix = {k: float(v) for k, v in (s.split(":") for s in a.mix.split(","))}
    pools = {k: (d, tr_real) if k == "uieb" else extra_split(k)[:2] for k in mix}
    print(f"{a.name}: fine-tune RefineNet on {len(tr_real)} steps/epoch, real pools "
          f"{ {k: len(p[1]) for k, p in pools.items()} } mix {mix}, {len(val_real)} UIEB val, init {a.init}", flush=True)

    log = [{"epoch": 0, "uieb": eval_real(base, ref_net.eval(), d, val_real, s1_real),
            "synth": eval_synth(base, ref_net, rgb, depth, stage1=s1_synth)}]
    print("start:", json.dumps(log[0]), flush=True)
    best = -1.0
    for ep in range(a.epochs):
        ref_net.train()
        t0, tl = time.time(), []
        if list(mix) == ["uieb"]:  # the original recipe: one pass over the UIEB train photos
            order = [(d, i) for i in tr_real[torch.randperm(len(tr_real), generator=gen)]]
        else:
            names = list(mix)
            src = torch.multinomial(torch.tensor([mix[k] for k in names]), len(tr_real), replacement=True, generator=gen)
            order = [(pools[names[s]][0], pools[names[s]][1][torch.randint(len(pools[names[s]][1]), (1,), generator=gen).item()])
                     for s in src.tolist()]
        for dd, i in order:
            I_r, z_r, ref_r = load_real(dd, int(i))
            Jb_r = s1_real(base, I_r, z_r)
            I_s, z_s, J_s, _ = make_batch(rgb, depth, tr_idx[torch.randint(len(tr_idx), (a.bs_synth,), generator=gen)], gen, DEV)
            Jb_s = s1_synth(base, I_s, z_s)
            with autocast():
                Jr_r = ref_net(Jb_r, I_r, z_r)
                Jr_s = ref_net(Jb_s, I_s, z_s)
            L_real = (Jr_r - srgb_to_linear(ref_r)).abs().mean()
            L_anchor = (Jr_r - Jb_r).abs().mean()
            L_synth = (Jr_s - J_s).abs().mean()
            loss = a.w_real * L_real + a.w_anchor * L_anchor + a.w_synth * L_synth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref_net.parameters(), 1.0)
            opt.step(), sched.step()
            tl.append(loss.item())
        u = eval_real(base, ref_net.eval(), d, val_real, s1_real)
        s = eval_synth(base, ref_net, rgb, depth, stage1=s1_synth)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "uieb": u, "synth": s, "time": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        # select on real val PSNR, but only while synthetic gain is not thrown away (refined >= betanet there)
        if a.last or (u["RefineNet"]["psnr"] > best and s["refined"] >= s["betanet"] - 0.3):
            best = u["RefineNet"]["psnr"]
            torch.save({"state": ref_net.state_dict(), "args": vars(a)}, out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)
    print(f"best real-val PSNR (with synth gain kept): {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
