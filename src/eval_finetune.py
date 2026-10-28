"""Before / after fine-tuning comparison (does not overwrite the main results files).

For every checkpoint name given (first = the "before" model):
  synth   : the 654 held-out NYU test rooms under the same random water as eval_synth (seed 2026):
            median relative coefficient error per channel + PSNR / SSIM / dE00 of the restored image.
            Fine-tuning on real photos must not break what the synthetic supervision taught.
  heldout : the self-supervised physics losses on the held-out real frames (dive D3 + the chart scenes +
            the 24 evaluation frames) and the spread of the coefficients within each dive.
  sites   : mean beta_D / beta_B per dive over every downloaded Sea-thru frame (water-type read-out).
Writes results/finetune_eval.json.

  python src/eval_finetune.py beta_rgbd ft_seathru [ft_uieb ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_synth import load, run
from finetune import DEV, evaluate_real, load_real
from physics import sample_water
from train import load_split

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def site_coefs(net, frames):
    out = {}
    for f in frames:
        I, z = f["I"][None].to(DEV), f["z"][None].to(DEV)
        size = (240, int(round(I.shape[-1] * 240 / I.shape[-2])))
        I, z = (torch.nn.functional.interpolate(t, size, mode="area") for t in (I, z))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(I, z)
        out.setdefault(f["site"], []).append(torch.cat([p["beta_D"], p["beta_B"], p["B_inf"]], 1).float()[0].cpu().numpy())
    res = {s: {"beta_D": np.mean(v, 0)[:3].tolist(), "beta_B": np.mean(v, 0)[3:6].tolist(),
               "B_inf": np.mean(v, 0)[6:].tolist(), "beta_D_cv": (np.std(v, 0) / np.mean(v, 0))[:3].tolist(),
               "n": len(v)} for s, v in sorted(out.items())}
    # site separation of log beta_D: spread of the dive means / mean spread within a dive (per channel, averaged).
    # ~0 = every dive gets the same water (collapse); larger = the dives are told apart
    logs = {s: np.log(np.array(v)[:, :3]) for s, v in out.items()}
    between = np.std([l.mean(0) for l in logs.values()], 0)
    within = np.mean([l.std(0) for l in logs.values()], 0)
    res["separation"] = float(np.mean(between / within))
    return res


def main():
    names = sys.argv[1:] or ["beta_rgbd", "ft_seathru"]
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    test, train = load_real("test"), load_real("train")
    out = {}
    for n in names:
        net, _ = load(n)
        res, coefs, true = run(rgb, depth, water, {n: (net, "beta")}, use_dav2=False)
        rel = {k: np.median(np.abs(coefs[n][k] - true[k]) / true[k], 0).tolist() for k in true}
        out[n] = {"synth": {"image": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()},
                            "coef_median_rel_err": rel},
                  "heldout": evaluate_real(net, test),
                  "heldout_D3": evaluate_real(net, [f for f in test if f["site"] == "D3"]),
                  "sites": site_coefs(net, train + test)}
        print(n, json.dumps({"psnr": out[n]["synth"]["image"][n]["psnr"], "rel": rel,
                             "heldout": {k: v for k, v in out[n]["heldout"].items() if k in ("dark", "range", "gray")}}), flush=True)
    path = ROOT / "results" / "finetune_eval.json"
    old = json.load(open(path)) if path.exists() else {}
    json.dump({**old, **out}, open(path, "w"), indent=1)  # models evaluated earlier are kept


if __name__ == "__main__":
    main()
