# sunlight_depth_v2: SunDepthBetaNet v2 (16 outputs)

Branch `sunlight_depth_v2`, started from `sunlight_phy_model` @ `42a265d`. Everything this experiment adds is in
this folder; nothing outside it was changed.

The network predicts the surface daylight colour **E0**, the diffuse attenuation **K_d** and the depth below the
surface **d_s**. The sunlight at the scene is then computed, not predicted: **E = E0 ⊙ exp(−K_d d_s)**.

Read first:

1. [SUNLIGHT_DEPTH_V2_SUMMARY.md](SUNLIGHT_DEPTH_V2_SUMMARY.md): what changed, how it was trained and tested,
   results, limitations, how to reproduce.
2. [LITERATURE_REVIEW_SUNLIGHT_PHYSICS.md](LITERATURE_REVIEW_SUNLIGHT_PHYSICS.md): the physics and its sources,
   with presentation talking points.

| | Outputs | Sunlight |
|---|---|---|
| BetaNet (`src/model.py`) | 12: β_D, β_B, B∞, E | E predicted directly |
| SunDepthBetaNet v1 (`sunlight_phy/model_sun.py`) | 10: β_D, β_B, B∞, d_s | E = exp(−0.8 β_D d_s), E0 white |
| **SunDepthBetaNet v2** (`model_v2.py`) | **16**: β_D, β_B, B∞, E0, K_d, d_s | E = E0 ⊙ exp(−K_d d_s) |

## Layout

```
water_v2.py                 v2 synthetic water: E0 (CIE daylight), K_d (Jerlov K_d + sun angle), d_s
model_v2.py                 SunDepthBetaNetV2
train_v2.py                 training of v2 (v1 recipe + E0 / K_d / d_s supervision)
train_existing_v2data.py    retrains BetaNet and v1 on the v2 water with their own unchanged training code
tests_v2.py                 lightweight checks (generator, shapes, numerical validity of E, losses)
eval_v2.py                  synthetic evaluation (654 test rooms, 10 Jerlov types, attribution, latency)
eval_real_v2.py             real photos (UIEB 90 val, Sea-thru colour chart)
figs_v2.py                  all plots, tables and galleries
run_all_v2.sh               everything in order
figures/architecture/       the two architecture diagrams (SVG + PNG) and their source make_diagrams.py
results_16_output/          v2's own results (plots; per-image arrays are git-ignored .npz)
three_model_comparisons/    BetaNet vs v1 vs v2 (JSON, plots, table)
baseline_vs_16_comparisons/ BetaNet vs v2 only (plots, table)
figures/                    additional analyses
qualitative_images/         restoration galleries (synthetic and real)
runs/<name>/                checkpoints (best.pt, git-ignored), log.json, config.json
```

## Reproduce

```bash
bash sunlight_depth_v2/run_all_v2.sh        # tests, 3 trainings (~1 h), evaluation + figures (~20 min)
bash sunlight_depth_v2/run_all_v2.sh eval   # evaluation + figures only
```
