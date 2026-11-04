"""What does the predicted depth d̂_s actually track? (reads results/synth_per_image_sun.npz from eval_sun.py)

1. Three reference depths, compared with d̂_s on the 654-scene synthetic test:
   - the true depth d_s;
   - the E-equivalent depth kappa d_s / 0.8: the depth that reproduces the true E with K_d = 0.8 * (true beta_D);
   - the self-consistent depth: the depth that reproduces the true E with K_d = 0.8 * (the network's OWN beta_D).
     Least squares on the log colour ratios to green, which cancel the max-channel normalisation:
     log E_c - log E_G = -0.8 d (b_c - b_G) with b = predicted beta_D.
2. By the generator's hidden kappa: the depth bias, the PSNR change when the true d_s replaces d̂_s, and the PSNR
   that K_d = 0.8 beta_D costs even with every coefficient true (option A's ceiling).

  python sunlight_phy/diagnose_depth.py   -> results/depth_diagnosis.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent / "results"


def main():
    P = dict(np.load(RES / "synth_per_image_sun.npz"))
    d, dh, k = P["true|d_s"][:, 0], P["pred|sun|d_s"][:, 0], P["true|kappa"][:, 0]
    bh, lE = P["pred|sun|beta_D"], np.log(P["true|E"])
    a = 0.8 * (bh - bh[:, 1:2])
    b = -(lE - lE[:, 1:2])
    d_self = np.clip((a * b).sum(1) / (a * a).sum(1), 0, 15)
    out = {"reference depths": {}}
    for name, ref in [("true d_s", d), ("E-equivalent depth (true beta_D)", k * d / 0.8),
                      ("self-consistent depth (network's own beta_D)", d_self)]:
        e = dh - ref
        out["reference depths"][name] = {"MAE_m": float(np.abs(e).mean()), "median_abs_m": float(np.median(np.abs(e))),
                                         "spearman": float(spearmanr(dh, ref).statistic)}
    gain = P["metric|sun | true d_s|psnr"] - P["metric|sun|psnr"]
    cap = P["metric|oracle (true coefs)|psnr"] - P["metric|true coefs | E(true d_s, K_d=0.8 beta_D)|psnr"]
    out["by kappa"] = {}
    for lo, hi in [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        m = (k >= lo) & (k < hi)
        out["by kappa"][f"{lo}-{hi}"] = {"n": int(m.sum()), "depth_bias_m": float((dh - d)[m].mean()),
                                         "psnr_change_with_true_d_s_dB": float(gain[m].mean()),
                                         "psnr_cost_of_K_d_0.8_beta_D_with_true_coefs_dB": float(cap[m].mean())}
    print(json.dumps(out, indent=1))
    json.dump(out, open(RES / "depth_diagnosis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
