"""Lightweight checks for the v2 generator, the 16-output model and its losses (CPU/GPU, < 1 min).

  python sunlight_depth_v2/tests_v2.py      -> prints each check, writes results_16_output/tests_v2.json, exits 1 on failure
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent / "sunlight_phy"), str(HERE)]
import train as T  # noqa: E402
from model import BetaNet  # noqa: E402
from model_sun import SunDepthBetaNet  # noqa: E402
from model_v2 import SunDepthBetaNetV2  # noqa: E402
from sun_physics import sample_water_sun  # noqa: E402
from water_v2 import daylight_rgb, sample_water_v2, sunlight_v2  # noqa: E402

import train_v2  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = {}


def check(name, ok, info=""):
    R[name] = {"ok": bool(ok), "info": info}
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""), flush=True)


def main():
    # 1. same waters and depths as the old generator on every fixed evaluation set (first call of each stream)
    for n, s, kw in [(654, 2026, {}), (288, 123, {}), (200, 1, {"water_type": "5C"}), (24, 0, {"murky_keep": 0.4})]:
        a = sample_water_sun(n, "cpu", torch.Generator().manual_seed(s), **kw)
        b = sample_water_v2(n, "cpu", torch.Generator().manual_seed(s), **kw)
        check(f"generator: beta_D, beta_B, d_s identical to the old generator (n={n}, seed={s}, {kw})",
              all(torch.equal(a[k], b[k]) for k in ("beta_D", "beta_B", "d_s")))
    w = sample_water_v2(654, "cpu", torch.Generator().manual_seed(2026))
    check("generator: labels reproduce E exactly (E = E0 exp(-K_d d_s), max 1, floor 0.03)",
          torch.equal(sunlight_v2(w["E0"], w["K_d"], w["d_s"]), w["E"]))
    check("generator: E finite, in [0.03, 1], brightest channel = 1",
          bool(torch.isfinite(w["E"]).all() and w["E"].min() >= 0.03 and torch.allclose(w["E"].amax(1), torch.ones(654))))
    ratio = (w["K_d"] / w["beta_D"]).log()
    check("generator: K_d is not a fixed multiple of beta_D (std of log ratio per channel > 0.1)",
          bool((ratio.std(0) > 0.1).all()), f"std {ratio.std(0).numpy().round(3).tolist()}")
    e = daylight_rgb(torch.tensor([4000.0, 5000.0, 6504.0, 8000.0, 10000.0]))
    check("daylight: 6504 K -> white [1, 1, 1]", torch.allclose(e[2], torch.ones(3), atol=1e-6))
    rb = (e[:, 0] / e[:, 2]).tolist()
    check("daylight: red/blue ratio falls with colour temperature", all(x > y for x, y in zip(rb, rb[1:])),
          f"R/B {[round(x, 3) for x in rb]}")

    # 2. the 16-output model
    net = SunDepthBetaNetV2(pretrained=False).to(DEV)
    I, z = torch.rand(4, 3, 120, 160, device=DEV), 0.5 + 5 * torch.rand(4, 1, 120, 160, device=DEV)
    p = net(I, z)
    shapes = {k: tuple(v.shape) for k, v in p.items()}
    want = {"beta_D": (4, 3), "beta_B": (4, 3), "B_inf": (4, 3), "E0": (4, 3), "K_d": (4, 3), "d_s": (4, 1), "E": (4, 3)}
    check("model: head has 16 outputs", net.head[-1].out_features == 16)
    check("model: output shapes", shapes == want, str(shapes))
    check("model: E = E0 exp(-K_d d_s) (normalised, floored), finite, in [0.03, 1]",
          bool(torch.allclose(p["E"], sunlight_v2(p["E0"], p["K_d"], p["d_s"])) and torch.isfinite(p["E"]).all()
               and p["E"].min() >= 0.03 and p["E"].max() <= 1 + 1e-6))
    ext = sunlight_v2(torch.ones(2, 3, device=DEV), torch.tensor([[5.9, 5.9, 0.001]] * 2, device=DEV),
                      torch.tensor([[14.9], [0.01]], device=DEV))
    check("model: E stays finite at extreme K_d and d_s (no underflow)", bool(torch.isfinite(ext).all()), str(ext.tolist()))
    check("model: bounds (K_d in (0.001, 6) /m, d_s in (0, 15) m)",
          bool(p["K_d"].min() > 1e-3 and p["K_d"].max() < 6 and p["d_s"].min() > 0 and p["d_s"].max() < 15))
    po = net(I, z, override={"d_s": torch.full((4, 1), 3.0, device=DEV)})
    check("model: override replaces d_s inside E only", bool(torch.allclose(po["E"], sunlight_v2(p["E0"], p["K_d"],
          torch.full((4, 1), 3.0, device=DEV))) and torch.equal(po["d_s"], p["d_s"])))
    net.zero_grad()
    (p["E"].log().mean()).backward()
    g = net.head[-1].weight.grad.abs().sum(1)
    check("model: the E term trains the E0, K_d and d_s rows, not the water rows",
          bool((g[9:16] > 0).all() and (g[0:9] == 0).all()))

    # 3. data + losses on a real v2 training batch
    gen = torch.Generator().manual_seed(0)
    rgb = torch.randint(0, 256, (24, 120, 160, 3), generator=gen, dtype=torch.uint8)
    depth = (0.5 + 9.5 * torch.rand(24, 120, 160, generator=gen)).half()
    T.sample_water = sample_water_v2
    I, z, J, w = T.make_batch(rgb, depth, torch.arange(24), gen, DEV, murky_keep=0.4)
    check("data: make_batch with the v2 water gives finite images in [0, 1] and all label keys",
          bool(torch.isfinite(I).all() and I.min() >= 0 and I.max() <= 1
               and {"E0", "K_d", "d_s", "E", "beta_D"} <= set(w)))
    p = net(I, z)
    L = train_v2.losses(p, w)
    check("loss: all six terms finite", all(torch.isfinite(v) for v in L.values()) and
          bool(torch.isfinite(T.img_loss(T.restore(I, z, p), J, z, w))), str({k: round(v.item(), 3) for k, v in L.items()}))
    L2 = train_v2.losses({**p, "E0": 2 * p["E0"]}, w)
    check("loss: E0 term is scale-free (chromaticity only)", bool(torch.isclose(L["E0"], L2["E0"], atol=1e-5)))
    for name, m in (("BetaNet (12)", BetaNet(pretrained=False)), ("SunDepthBetaNet v1 (10)", SunDepthBetaNet(pretrained=False))):
        q = m.to(DEV)(I, z)
        check(f"existing model runs on v2 batches: {name}", bool(torch.isfinite(T.coef_loss(q, w))))

    out = HERE / "results_16_output"
    out.mkdir(exist_ok=True)
    json.dump(R, open(out / "tests_v2.json", "w"), indent=1)
    if not all(r["ok"] for r in R.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
