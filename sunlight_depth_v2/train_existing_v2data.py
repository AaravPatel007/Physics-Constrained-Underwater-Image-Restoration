"""Retrain an existing model on the v2 water, with its own unchanged training code.

  --model baseline   src/train.py           BetaNet, 12 outputs (predicts E)
  --model sun_v1     sunlight_phy/train_sun SunDepthBetaNet v1, 10 outputs (d_s; K_d = 0.8 beta_D, E0 white)

The only change is the water generator: train.sample_water (used by make_batch and, in train.py, for the validation
water) and train_sun.sample_water_sun (its validation water) become water_v2.sample_water_v2. Every other argument is
passed through unchanged. The checkpoint goes to sunlight_depth_v2/runs/<name>/.

  python sunlight_depth_v2/train_existing_v2data.py --model baseline --name baseline_v2data \
      --data train_big --nval 288 --epochs 60 --murky-keep 0.4
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent / "sunlight_phy"), str(HERE)]
import train as T  # noqa: E402
from water_v2 import sample_water_v2  # noqa: E402


def main():
    args = sys.argv[1:]
    model = args.pop(args.index("--model") + 1)
    args.remove("--model")
    i = args.index("--name") + 1
    if model == "baseline":  # train.py writes to results/<name>
        args[i] = f"../sunlight_depth_v2/runs/{args[i]}"
        T.sample_water = sample_water_v2
        sys.argv = ["train.py", "--model", "beta", *args]
        T.main()
    elif model == "sun_v1":  # train_sun.py writes to sunlight_phy/runs/<name>
        import train_sun as TS
        args[i] = f"../../sunlight_depth_v2/runs/{args[i]}"
        T.sample_water = TS.sample_water_sun = sample_water_v2
        sys.argv = ["train_sun.py", *args]
        TS.main()
    else:
        raise SystemExit(f"unknown --model {model}")


if __name__ == "__main__":
    main()
