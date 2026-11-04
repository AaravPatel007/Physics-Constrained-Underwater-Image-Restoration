"""Check that the depth labels come from the *same* waters the baseline sees (run before training).

1. sample_water_sun == physics.sample_water on every key the original returns, bit for bit, for every way the
   project calls it: validation (seed 123), synthetic test (seed 2026), exact Jerlov types (seed 1), training
   batches with murky_keep 0.4 (rejection sampling), and the low_red option.
2. train.make_batch with sample_water_sun swapped in == the original train.make_batch, batch after batch from the
   same generator (images, depth, water, exposure, noise all identical), so the new model is trained on exactly
   the baseline's data stream.
3. The labels explain the generator's E: normalise(exp(-kappa * beta_D * d_s)) with the 0.03 floor == E.
4. sunlight_from_depth with the true K_d and d_s gives the true E (the physics layer matches the generator).

  python sunlight_phy/verify_rng.py      -> writes sunlight_phy/results/verify_rng.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import physics  # noqa: E402
import train as T  # noqa: E402

from sun_physics import WATER_KEYS, sample_water_sun, sunlight_from_depth  # noqa: E402


def same(a, b):
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in WATER_KEYS)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    cases = {"val (288, seed 123)": dict(n=288, seed=123), "test (654, seed 2026)": dict(n=654, seed=2026),
             "jerlov 3C (200, seed 1)": dict(n=200, seed=1, water_type="3C"),
             "jerlov 9C (200, seed 1)": dict(n=200, seed=1, water_type="9C"),
             "train murky_keep 0.4 (24, seed 0)": dict(n=24, seed=0, murky_keep=0.4),
             "low_red 0.4 (64, seed 5)": dict(n=64, seed=5, low_red=0.4)}
    for name, c in cases.items():
        kw = {k: v for k, v in c.items() if k not in ("n", "seed")}
        g1, g2 = torch.Generator().manual_seed(c["seed"]), torch.Generator().manual_seed(c["seed"])
        ok = True
        for _ in range(3):  # three consecutive calls: the generators must stay in step
            a = physics.sample_water(c["n"], dev, g1, **kw)
            b = sample_water_sun(c["n"], dev, g2, **kw)
            ok &= same(a, b)
        res[f"sample_water identical: {name}"] = bool(ok)

    # make_batch, original vs patched, on fake rooms (no dataset needed)
    gr = torch.Generator().manual_seed(1)
    rgb = torch.randint(0, 256, (96, 240, 320, 3), generator=gr, dtype=torch.uint8)
    depth = (0.5 + 9.5 * torch.rand(96, 240, 320, generator=gr)).half()
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    ok = True
    for b in range(4):
        idx = torch.arange(b * 24, (b + 1) * 24)
        T.sample_water = physics.sample_water
        A = T.make_batch(rgb, depth, idx, g1, dev, murky_keep=0.4)
        T.sample_water = sample_water_sun
        B = T.make_batch(rgb, depth, idx, g2, dev, murky_keep=0.4)
        ok &= all(torch.equal(x, y) for x, y in zip(A[:3], B[:3])) and same(A[3], B[3])
    res["make_batch identical (4 training batches, murky_keep 0.4)"] = bool(ok)
    T.sample_water = physics.sample_water

    # labels explain E (on the CPU, where the generator computes it; GPU exp() differs in the last bit, ~2e-7)
    w = sample_water_sun(654, "cpu", torch.Generator().manual_seed(2026))
    E = torch.exp(-w["beta_D"] * w["d_s"] * w["kappa"])
    E = (E / E.amax(1, keepdim=True)).clamp_min(0.03)
    res["labels reproduce E exactly (test set)"] = bool(torch.equal(E, w["E"]))
    E2 = sunlight_from_depth(w["K_d"], w["d_s"])
    res["sunlight_from_depth(true K_d, true d_s) max |dE|"] = float((E2 - w["E"]).abs().max())
    res["test d_s range [m]"] = [float(w["d_s"].min()), float(w["d_s"].max())]
    res["test kappa range"] = [float(w["kappa"].min()), float(w["kappa"].max())]
    res["test share of images with E_R at the 0.03 floor"] = float((w["E"][:, 0] <= 0.03).float().mean())
    print(json.dumps(res, indent=1))
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "results" / "verify_rng.json", "w"), indent=1)
    assert all(v for k, v in res.items() if "identical" in k or "exactly" in k), "RNG copy is NOT identical"


if __name__ == "__main__":
    main()
