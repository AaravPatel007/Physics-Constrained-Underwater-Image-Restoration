"""Old vs longer BetaNet on the same synthetic test as eval_synth.py (654 NYU test rooms, water seed 2026,
plus the 10 exact Jerlov types on 200 scenes). Writes results/synth_eval_big.json; synth_eval.json is untouched."""
import json

import numpy as np
import torch

from eval_synth import DEV, ROOT, load, run
from physics import JERLOV_N, sample_water
from train import load_split

NAMES = ["beta_rgbd", "beta_rgbd_big"]


def rel_err(pred, true):
    return {k: (np.median(np.abs(pred[k] - true[k]) / true[k], 0) * 100).round(1).tolist() for k in true}


def main():
    rgb, depth = load_split("test")
    nets = {n: load(n) for n in NAMES}
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    res, coefs, true = run(rgb, depth, water, nets, use_dav2=False)
    out = {"random": {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()},
           "coef_median_rel_err_pct": {n: rel_err(coefs[n], true) for n in NAMES}, "jerlov": {}}
    sub = slice(0, 200)
    for t in JERLOV_N:
        w = sample_water(200, DEV, torch.Generator().manual_seed(1), water_type=t)
        r, c, tr = run(rgb[sub], depth[sub], w, nets, False)
        out["jerlov"][t] = {m: float(np.mean(d["psnr"])) for m, d in r.items()}
        out["jerlov"][t]["beta_D_true"] = tr["beta_D"][0].round(3).tolist()
        out["jerlov"][t].update({f"beta_D_pred|{n}": c[n]["beta_D"].mean(0).round(3).tolist() for n in NAMES})
    json.dump(out, open(ROOT / "results" / "synth_eval_big.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("random", "coef_median_rel_err_pct")}, indent=1))
    print(json.dumps({t: {m: round(v, 1) for m, v in d.items() if isinstance(v, float)} for t, d in out["jerlov"].items()}))


if __name__ == "__main__":
    main()
