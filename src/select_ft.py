"""Pick the fine-tuned model by a rule fixed before looking at the test charts (results/finetune_eval.json).

  1. synthetic test: mean beta_D median relative error <= 1.10 x the starting model's
  2. between-dive spread (std over dives of the dive-mean log beta_D, averaged over R,G,B) >= 0.5 x the
     starting model's (no collapse to one water). Amended after round 2: the first version divided by the
     within-dive spread, which shrinks too when the model collapses, so it passed the collapsed ft_seathru.
  3. among the survivors: lowest held-out physics loss (range + gray + 10 x dark, all held-out real frames)
Chart errors (2 frames) and UIEB are reported, never used for the choice.

  python src/select_ft.py [--base beta_rgbd]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="beta_rgbd")
    a = ap.parse_args()
    F = json.load(open(ROOT / "results" / "finetune_eval.json"))
    bd = lambda n: float(np.mean(F[n]["synth"]["coef_median_rel_err"]["beta_D"]))
    dives = lambda n: [k for k in F[n]["sites"] if k.startswith("D")]
    sep = lambda n: float(np.mean(np.std([np.log(F[n]["sites"][d]["beta_D"]) for d in dives(n)], 0)))
    phys = lambda n: F[n]["heldout"]["range"] + F[n]["heldout"]["gray"] + 10 * F[n]["heldout"]["dark"]
    rows = []
    for n in F:
        ok = n != a.base and bd(n) <= 1.10 * bd(a.base) and sep(n) >= 0.5 * sep(a.base)
        rows.append((n, bd(n), sep(n), phys(n), ok))
        print(f"{n:22s} beta_D err {100 * bd(n):5.1f}%  separation {sep(n):5.2f}  held-out physics {phys(n):.3f}  "
              f"{'PASS' if ok else ('base' if n == a.base else 'fail')}")
    best = min((r for r in rows if r[4]), key=lambda r: r[3], default=None)
    print("selected:", best[0] if best else "none")
    json.dump({"selected": best[0] if best else None, "rows": rows}, open(ROOT / "results" / "ft_selected.json", "w"), indent=1)


if __name__ == "__main__":
    main()
