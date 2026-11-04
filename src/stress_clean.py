"""Re-score the paired stress-test sets without test photos that have a copy in any training pool.

Same-photo test as prep_lsui.py on main: correlation of 48x48 grey thumbnails >= 0.955 (colour casts do not count).
Reads results/stress_test/per_image_*.json -> results/stress_test/clean_rescore.json. Run after stress_test.py.

  python src/stress_clean.py
"""
import json

import cv2
import numpy as np

from eval_day1 import boot_ci
from finetune_refine import extra_split, real_split
from stress_test import OUT, load_set

MATCH = 0.955
SETS = ["uieb160", "uieb90", "lsui", "euvp"]


def thumb(a):
    x = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def main():
    d, tr, _ = real_split()
    pool = [thumb(d[f"raw_{i}"]) for i in tr]
    for name in ("lsui", "euvp"):
        e, e_tr, _ = extra_split(name)
        pool += [thumb(e[f"raw_{i}"]) for i in e_tr]
    pool = np.stack(pool)
    out = {}
    for name in SETS:
        T = np.stack([thumb(a) for a, *_ in load_set(name)])
        keep = np.flatnonzero((T @ pool.T).max(1) < MATCH)
        res = json.load(open(OUT / f"per_image_{name}.json"))["per_image"]
        assert len(res["BetaNet"]["psnr"]) == len(T)
        pick = lambda m, k: np.array(res[m][k])[keep]
        out[name] = {"n": len(T), "n_kept": len(keep),
                     "mean": {m: {k: float(pick(m, k).mean()) for k in ("psnr", "ssim", "de2000")} for m in res},
                     "ci": {f"{b} - {a}": {k: boot_ci(pick(a, k), pick(b, k)) for k in ("psnr", "de2000")}
                            for a, b in [("BetaNet", "RefineNet [refine_ft]"), ("BetaNet", "RefineNet [refine_mix]"),
                                         ("RefineNet [refine_ft]", "RefineNet [refine_mix]")]}}
        print(name, len(keep), "of", len(T), json.dumps(out[name]["ci"]), flush=True)
    json.dump(out, open(OUT / "clean_rescore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
