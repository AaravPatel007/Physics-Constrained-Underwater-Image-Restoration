# Review 2 plan (Method 1: keep improving BetaNet)

> **START HERE (state at end of 2 Nov 2026).** Model names: `docs/MODELS.md`.
> - Method 1 model in use = `baseline_v1_60ep_bigdata` (60 epochs, big data) **+ Day 1 fixes** (applied at restore time, no retraining).
> - Day 1 helped (UIEB 18.48 → 19.09 dB, ΔE 13.2 → 12.3, fins specks 10.8 → 4.4). Fins still blue-green.
> - Day 2 = three retrains (`day2a_added_murky_water`, `day2b_added_deep_red_starved_water`, `day2c_added_camera_flaws`); none fixed the tint, all are side results. Conclusion: not a training-range problem.
> - **Prepared, not run: Day 2 run d** (`day2d_added_murky_deep_red_camera_60ep`, all three changes together, from scratch, 60 epochs). Start it with the command in the Day 2 run d section below; ~90 min incl. tests; start `thermal_guard.py` first (the queue script does it).
> - **Day 1b done (2 Nov, late):** the pink halo around objects (val #22) came from the red-from-green smoothing in 15 px patches. Tuned on the 200 UIEB train photos: **3 px patch, fog-cap margin unchanged at 3** (margin 3 is a true optimum: lower = hazier, higher = worse colour). Halo gone; 90 val photos ΔE 12.25 → 12.21 (55/90 better, small but significant). Trade-off: fins specks 4.4 → 5.8. Pink veil in #24 not fixed by any setting (fog cap; that photo's reference is itself purple-grey) = known limitation. See the Day 1b section below. **Day 1b is the recipe in use from now on**: `CURRENT_RECIPE` in `src/safe_restore.py` (Day 3 and later scripts must use it; the Day 1 / Day 2 figures stay as they were made).
> - **Day 3 done (3 Nov):** touch-up network on top of physics + Day 1b (`day3a_touchup_global`, chosen on held-out train photos). 90 val: ΔE 12.22 → 9.66, PSNR 19.11 → 21.68, 73/90 better; fins turn yellow (ΔE 18.6 → 12.6); grey charts 10.3°/10.6° → 5.1°/4.8°, while the no-physics control stays ~10°. See "Day 3 results".
> - **Next: Day 4** (evaluation + slides). Review 2 is ~5–6 Nov.
> - Machine notes: run `src/thermal_guard.py` while training, stop the user's `wec_rl` jobs, python = `/home/aarav-patel/wec_rl/.venv/bin/python`. `src/day2_queue.sh` trains + evaluates a run in one go (`THREADS=4` allowed).
> - **More epochs + DAv2 depth: done (3 Nov), neither replaces v1.** See section "Done: more epochs and DAv2 depth" below. v1 + Day 1b + `day3a_touchup_global` stays the method.
> - Pending idea (not run): "fog ≤ darkest pixels" hinge loss in `finetune.py`; gentler deep-water mix (~15 %).

Written 2 Nov 2026. Review 2 is ~5–6 Nov 2026.
Two methods will be presented:
- **Method 1 (Aarav):** keep the current BetaNet and fix it. This file.
- **Method 2 (friends):** predict sunlight colour from depth, E_c(d_s) = E0_c · exp(−K_d,c · d_s).
  Both methods use the same test images and the same fins slide.

## How the model works (plain words)
Water does two things to a photo:
1. **It eats colour.** Red disappears first, so everything looks blue-green.
2. **It adds a glow (fog).** Light scattered by the water comes into the camera.

The model estimates both effects, **subtracts the fog**, then **boosts the colours the water ate**.
Last, a colour-balance step makes the overall picture look neutral.

## What went wrong on the fins photo (UIEB val image #2)
Measured with both trained models (original and 60-epoch):

| | Red | Green | Blue |
|---|---|---|---|
| Red light in the photo (linear, median) | 0.0003 | 0.26 | 0.23 |
| Fog the model subtracts (60-epoch) | 0.0049 | 0.011 | 0.013 |
| Pixels where subtracted fog > photo | 94 % | 2 % | 1 % |
| Sunlight colour predicted | 0.03 (training floor) | 0.76 | 1.0 |
| Boost (median gain) | 4.1× | 1.2× | 1.1× |

- **Blue/green tint:** the model removes ~16× more red fog than the photo contains. Red goes below
  zero, gets clipped to zero, so red is erased. No red = cyan.
- **Red specks:** not from the boost (only ~4×). The colour-balance step divides red by its average,
  which is almost zero, so the few noise pixels where red survived become bright red dots.
- **Why the model did this:** our fake training water never looks like this photo. Sunlight colour
  was never allowed below 0.03, the fog always kept at least ~10 % red, and the fake images were
  perfectly clean (no JPEG, no 8-bit rounding). Training longer on the same data cannot teach it this.
- **Hard limit:** red is only 2–3 levels out of 255. Physics alone can make the fins neutral, not yellow.
  Only a learned guess (Day 3) can fill in colour that the photo never recorded.

## Day 1: apply the model more safely (no retraining)
- **Never remove more fog than the photo contains:** always keep a small part of what the camera saw.
- **Careful colour balance:**
  - limit how much any colour can be boosted;
  - judge the balance only from pixels that are real signal, not noise;
  - smooth the weak colour before boosting it, guided by the cleaner colours.
- Gives before/after pictures straight away.

### How we tell noise from signal
- Blur the photo slightly and subtract the blurred copy from the original. In smooth areas (sand, open
  water) what's left is mostly noise. Its typical size, per colour, is the **noise floor**
  (usually 1–3 levels out of 255 for JPEG).
- A pixel's colour counts as signal only if it is well above the floor (e.g. 3×). On the fins photo
  almost no red pixel passes, which confirms there is no usable red.
- Colour balance uses only the pixels that pass.

### How we smooth the weak colour (guided filter)
- In each small patch, assume red follows green: red ≈ a × green + b. Fit this over the patch.
- The fit averages many pixels, so random specks cancel; edges stay sharp because they come from green.
- Blend: strong red → keep the original; red near the noise floor → use the smoothed version.
- Patches are not a fixed grid: every pixel has its own patch centred on it (overlapping). A patch that
  crosses an edge fits the rule across both sides, and each pixel uses its own green, so red jumps where
  green jumps. Known weakness: if the two sides have different red-to-green relationships, a thin colour
  fringe (halo) can appear. Fix: smaller patches or a weighted guided filter.

### Order of Day 1 steps
1. Measure the noise floor; decide per pixel how trustworthy each colour is.
2. Smooth red where it is untrustworthy (guided by green).
3. Colour-balance the whole image: only trustworthy pixels are counted when working out the boost, and
   the boost is capped. No pixels are removed.

### Why this is defensible
- Guided filter: He, Sun & Tang, ECCV 2010 / TPAMI 2013, widely used, including in dehazing.
- Rebuilding red from green underwater: Ancuti et al., "Color Balance and Fusion for Underwater Image
  Enhancement", IEEE TIP 2018 (green survives underwater, red does not).
- Our own evidence: speck score before/after, colour-chart error not worse, fins side by side.
- Honest framing: this removes noise; it does not recover true red information.

### Day 1 results (done 2 Nov 2026)
Code: `src/safe_restore.py` (the three fixes), `src/eval_day1.py` (evaluation), `src/fig_day1.py` (figures).
Results: `results/review2_day1/eval.json`. Figures: `figures/5_review2_day1_safe_restore/`.
Settings were tuned on 200 UIEB *train* photos (white-balance gain cap 16, shades-of-gray power 4,
guided-filter eps 1e-4); the test images were never used to choose. The noise margin of the fog cap (3x noise)
was fixed before looking at results.

Longer-trained model (`baseline_v1_60ep_bigdata`), current recipe → all Day 1 fixes:

| Test | Current | All Day 1 fixes | Change (95 % CI) |
|---|---|---|---|
| UIEB val, 90 photos: PSNR | 18.48 dB | **19.09 dB** | +0.61 (+0.22 to +1.04) |
| UIEB val: colour error ΔE00 | 13.23 | **12.25** | −0.99 (−1.67 to −0.36) |
| UIEB val: SSIM | 0.783 | **0.827** | +0.04 (+0.03 to +0.06) |
| UIEB val: speck score | 7.7 | **6.5** | −1.2 (−1.6 to −0.8) |
| Red-starved photos (8): PSNR / ΔE00 | 12.75 / 19.1 | 14.20 / 16.7 | +1.45 (−0.24 to +3.52) |
| Fins photo: PSNR / ΔE00 / specks | 9.7 / 20.0 / 10.8 | 10.1 / 18.7 / **4.4** | — |
| Colour charts (SfM range), 2 frames | 11.7° / 11.2° | 10.3° / 10.3° | — |
| Synthetic test, 654 rooms (divides by E) | 21.22 dB | **21.91 dB** | +0.7 |
| Synthetic with true coefficients | 25.11 dB | **26.80 dB** | +1.7 |

Original model: UIEB 18.01 → 18.91 dB, ΔE 13.86 → 12.72 (CIs exclude zero); synthetic 19.99 → 21.00 dB.

What each fix does (ablation, longer run):
- **Careful white balance** gives most of the colour gain (ΔE 13.2 → 12.3 on its own) and fewer specks.
- **Fog cap** gives the structure gain (SSIM 0.78 → 0.82) and stops red being erased, but on its own it
  *increases* specks (7.7 → 8.4), because more noisy red survives.
- **Red smoothing** removes those extra specks, and on synthetic images it adds +0.6 dB.
- All three together are best on every UIEB measure.

Honest caveats for the talk:
- **The fins are still green.** Specks are gone (10.8 → 4.4), but the fin colour barely changes: in this photo
  red is inside the camera noise, so the fog cap (correctly) cannot tell fog from signal. Day 1 removes the
  artefacts; the tint needs Day 2 (model stops over-estimating red fog) and Day 3 (learned colour).
- **Colour charts:** better than the current recipe (11.7° → 10.3°), but still behind plain gray-world on
  the raw photo (8.9° / 8.1°). Only 2 chart frames, so indicative only.
- A first version of the fog cap without the noise margin gave charts 7.0° but dropped synthetic PSNR with
  the true coefficients from 25.1 to 21.1 dB (noise pulls the darkest pixels below the real fog). The margin
  fixed that; the stronger cap's chart gain is a trade-off to note, not used.
- The speck score also counts real fine texture (the UIEB references score 7.9), so compare it on the same
  image or between methods, not as an absolute.
- UIEB depth is estimated (Depth Anything V2), not measured.

## Day 1b: fix the halo and check the haze (no retraining, done 2 Nov)
Script `src/eval_day1b.py` → `results/review2_day1b_smaller_smoothing_patch/eval.json`, figures in `figures/7_review2_day1b_smaller_smoothing_patch/`.
- **Problem:** the 25-photo gallery showed a pink outline around the statue (val #22) and pink veils/haze in
  some photos (#24).
- **Tried:** smoothing patch 3 / 7 / 15 px × fog-cap safety margin 0 / 1 / 2 / 3 / 4 / 6 noise levels
  (18 settings). White balance kept at the Day 1 values.
- **How it was chosen (rule fixed before running):** all 18 settings on the 200 UIEB *train* photos;
  lowest mean ΔE wins; settings within 0.05 count as tied and the one with the lowest colour error next to
  edges (where halos show) wins. The 90 val photos were used once, at the end.
- **Chosen: 3 px patch, margin 3.** On train, ΔE 12.01 (Day 1) → 11.96. Margin 3 is the best margin for
  every patch size: smaller margins leave fog in (hazier), larger ones remove too much (worse colour).
  So the haze was not caused by too large a margin, as first guessed.
- **90 val photos (test):**

  | | PSNR | ΔE | ΔE next to edges | red specks | haze (ref 0.167) |
  |---|---|---|---|---|---|
  | No Day 1 fixes | 18.48 | 13.23 | 13.80 | 7.72 | 0.128 |
  | Day 1 (15 px) | 19.09 | 12.25 | 12.60 | 6.50 | 0.175 |
  | **Day 1b (3 px)** | **19.11** | **12.21** | **12.56** | 6.56 | 0.170 |

  ΔE change −0.03 (95 % CI −0.05 to −0.01), better in 55/90. Fins photo: ΔE 18.7 → 18.6, specks 4.4 → 5.8
  (the smaller patch smooths the fins' red noise less). Statue #22: ΔE 4.4 → 4.2, halo gone.
- **Not fixed:** the pink veil in #24 (ΔE 18.1 either way, raw model 16.5). It comes from the fog cap, and no
  setting changes it. Its UIEB reference is itself purple-grey. Present it as a limitation.

## Day 2: teach the model about extreme water (retrain, ~45 min per run)
- **More realistic fake training images:**
  - water where red is almost completely gone, in both the sunlight and the fog;
  - real camera flaws: JPEG compression and coarse 8-bit colour steps;
  - keep more very murky water in the training mix (also tests why the longer model got worse at
    telling dives apart).
- **Common-sense rule when training on real dive photos:** the fog you subtract cannot be brighter
  than the darkest things in the picture. Needs no labels and targets exactly this mistake.

### Day 2 run 1 results: more murky water only (done 2 Nov 2026)
Model `day2a_added_murky_water`: the same recipe as `baseline_v1_60ep_bigdata` (big data, 60 epochs), but very murky water kept
at 100 % instead of 40 %. Run: `src/day2_queue.sh day2a_added_murky_water`. Results:
`results/review2_day2/day2a_added_murky_water/` (eval.json = Day 1 tests, dive_separation.json, real_eval_day2a_added_murky_water.json).
Figures: `figures/6_review2_day2/1_more_murky_water/`. Kept as a side result only; the base stays `baseline_v1_60ep_bigdata`.

| Test | `baseline_v1_60ep_bigdata` (40 % murky) | Run a: `day2a_added_murky_water` (100 % murky) |
|---|---|---|
| Dive separation (higher = dives told apart; original model 1.57) | 1.29 | 1.34 |
| D4 (turbid) mean β_D vs clearest-looking other dives | 0.19 (D1 0.20) | 0.18 (D1 0.22, D3 0.18) |
| UIEB 90, all Day 1 fixes: PSNR / ΔE00 | 19.09 / 12.25 | 18.99 / 12.54 (ΔE worse, CI +0.06 to +0.52) |
| Fins, all Day 1 fixes: PSNR / ΔE00 / specks | 10.1 / 18.7 / 4.4 | 10.3 / 17.8 / 4.5 |
| Red-starved 8, all Day 1 fixes: PSNR / ΔE00 | 14.20 / 16.7 | 14.11 / 16.8 |
| Colour charts (SfM), all Day 1 fixes | 10.3° / 10.3° | 10.6° / 9.8° |
| Synthetic test, with fog cap + smoothing | 21.91 dB | 21.82 dB |
| Coefficient error on synthetic (median relative, β_D R/G/B) | 11 / 16 / 20 % | 12 / 15 / 20 % |

Verdict: **murky water was not the cause.** Keeping all murky water barely changes anything: separation moves
1.29 → 1.34 (still below the original's 1.57) and D4 is still not distinct (it ties D3). UIEB colour is slightly
worse, the fins slightly better (one image, within noise). Keep `baseline_v1_60ep_bigdata` as the base;
the drop in separation is more likely from the longer training / bigger data itself. Good slide: "we tested the
professor's 'train more' idea and our own guess; neither fixes the fins, because the fins water was never in the
training range" → leads into run 2 (near-zero-red sunlight and fog).

### Day 2 run 2: deep, red-starved water (done 2 Nov 2026)
Model `day2b_added_deep_red_starved_water`: **continues from the base model** (`--init baseline_v1_60ep_bigdata`), 20 epochs, lower learning
rate (1.5e-4), murky water at 40 % as in the base. 40 % of the fake waters are "deep": sunlight comes from 12–30 m
down (red can drop to 0.2 % of the strongest colour; the old floor was 3 %), and the fog's grey share is
0.2–10 % (the old value was always 10 %, which kept fog red ≥ ~10 %). Validation water is unchanged, so the
scores stay comparable. New extra test: 654 fake rooms in all-deep, red-starved water (seed 2027).
Command: `src/day2_queue.sh day2b_added_deep_red_starved_water --init baseline_v1_60ep_bigdata --epochs 20 --lr 1.5e-4 --murky-keep 0.4 --low-red 0.4`.
Results: `results/review2_day2/day2b_added_deep_red_starved_water/`; figures: `figures/6_review2_day2/2_low_red_water/`.
Kept as a side result only; the base stays `baseline_v1_60ep_bigdata`.

| Test | `baseline_v1_60ep_bigdata` | Run b: `day2b_added_deep_red_starved_water` |
|---|---|---|
| Validation (old water): coefficient loss / PSNR | 0.519 / 24.21 dB | 0.546 / 23.88 dB (flat over epochs 17–20) |
| Deep red-starved synthetic, fog cap + smoothing (oracle 20.13) | 16.90 dB | 17.62 dB |
| Synthetic test, with fog cap + smoothing | 21.91 dB | 21.88 dB |
| UIEB 90, all Day 1 fixes: PSNR / ΔE00 | 19.09 / 12.25 | 19.11 / 12.30 |
| Fins, all Day 1 fixes: ΔE00 / specks | 18.7 / 4.4 | 19.2 / 4.4 |
| Red-starved 8, all Day 1 fixes: PSNR / ΔE00 | 14.20 / 16.7 | 13.55 / 17.4 |
| Colour charts (SfM), all Day 1 fixes | 10.3° / 10.3° | 8.9° / 8.2° |
| Dive separation (original 1.57) | 1.29 | 1.20 |
| Mean β_D per dive D1–D5 | .204 .131 .165 .190 .120 | .169 .115 .143 .186 .110 (D4 now highest) |

Verdict: **it learns the fake deep water but this does not transfer to the real red-starved photos.** +0.7 dB on
the deep synthetic test, but the fins and the 8 real red-starved photos get slightly worse, and normal validation
ends a little worse than the base. Not under-trained (validation flat at the end), so more epochs would not help.
Side gains: charts improve ~1.5–2° and D4 becomes the murkiest dive, but overall separation drops. Conclusion for
the talk: the gap is between fake and real photos, not the water range → motivates run 3 (camera flaws).
If revisited: a gentler mix (~15 % deep) rather than longer training.

### Day 2 run 3: camera flaws (done 2 Nov 2026)
Model `day2c_added_camera_flaws`: same settings as run 2 but `--camera 0.5` instead of `--low-red` (half the training
images get shot noise, sRGB + 8-bit rounding, JPEG quality 50–95 with 4:2:0 chroma; `physics.camera_flaws`).
Command: `THREADS=4 src/day2_queue.sh day2c_added_camera_flaws --init baseline_v1_60ep_bigdata --epochs 20 --lr 1.5e-4 --murky-keep 0.4 --camera 0.5`.
Results: `results/review2_day2/day2c_added_camera_flaws/`; figures: `figures/6_review2_day2/3_camera_flaws/`. Side result only.

| Test | `baseline_v1_60ep_bigdata` | Run c: `day2c_added_camera_flaws` |
|---|---|---|
| Validation (old water): coefficient loss / PSNR | 0.519 / 24.21 dB | 0.518 / 23.87 dB (best epoch 19) |
| Fins, current recipe (no Day 1 fixes): ΔE00 / specks | 20.0 / 10.8 | 19.9 / **5.4** |
| Fins, all Day 1 fixes: ΔE00 / specks | 18.7 / 4.4 | 19.1 / 4.3 |
| UIEB 90, all Day 1 fixes: PSNR / ΔE00 | 19.09 / 12.25 | 18.99 / 12.31 |
| Red-starved 8, all Day 1 fixes: PSNR / ΔE00 | 14.20 / 16.7 | 14.14 / 16.8 |
| Synthetic test / deep red-starved synthetic | 21.91 / 16.90 dB | 22.05 / 16.85 dB |
| Colour charts (SfM), all Day 1 fixes | 10.3° / 10.3° | 11.3° / 11.0° |
| Dive separation (original 1.57) | 1.29 | 1.19 |

Verdict: **camera flaws halve the red specks on their own (10.8 → 5.4 without any Day 1 fix) but do not change the
colour.** With the Day 1 fixes on, the specks are already gone, so the end result equals the base. All three Day 2
runs agree: the cyan tint is not a training-range problem. In the fins photo the red signal is essentially absent;
the physics model cannot invent it, and the UIEB reference is a hand-picked enhancement. → Day 3 (learned touch-up
on the 800 UIEB train pairs) is the remaining lever. Base stays `baseline_v1_60ep_bigdata`.

### Day 2 run d: all three together, from scratch (prepared 2 Nov 2026, NOT run yet)
Why: runs a–c each changed one thing, and b and c were only 20 extra epochs at one learning rate, so "nothing helps"
was not proven. This run tests "does more of everything, trained properly, help?". Model `day2d_added_murky_deep_red_camera_60ep`: from scratch (no
`--init`), 60 epochs, lr 5e-4 (same as baseline_v1), big data, ALL murky water (1.0), 40 % deep red-starved,
half the photos with camera flaws. Same validation water as baseline_v1, so scores are comparable.
Command (from `src/`):
`THREADS=4 ./day2_queue.sh day2d_added_murky_deep_red_camera_60ep --epochs 60 --lr 5e-4 --murky-keep 1.0 --low-red 0.4 --camera 0.5 > ../results/logs/day2d_added_murky_deep_red_camera_60ep_eval.log 2>&1`
(run in the background; ~52 s/epoch, then ~35 min of tests). Then: `python fig_day2.py day2d_added_murky_deep_red_camera_60ep 4_all_three_combined`,
fill the table here, and compare with baseline_v1 on: fins, UIEB, red-starved 8, charts, dive separation.
Decision rule: replaces baseline_v1 only if it is better on UIEB and the red-starved photos without losing on charts / separation.

### Does Day 1 still matter after Day 2? Yes.
- Day 2 fixes the model's **estimates** (fog amount, water colour, sunlight colour).
- Day 1 fixes **what we do with the estimates** (subtracting, boosting).
- Even with perfect estimates, 2–3 levels of noisy red still go negative after subtraction and still
  explode when boosted. We already saw this: in the murkiest water, even the true coefficients failed.
- Also: Day 1 is a result in hand if Day 2 runs late, and "Day 2 with vs without Day 1" shows how much
  each fix contributes.

## Done: more epochs and DAv2 depth (3 Nov)
`src/dav2_queue.sh` → `prep_dav2_train.py` (3 stored fake waters per room, DAv2 run on each underwater image,
`data/nyu_train_big_dav2.npz`, 0.9 GB) → two runs that both continue v1 on those stored waters with **early
stopping** (`train.py --patience 8`: constant LR 1.5e-4, halved after 3 epochs without a better validation score,
stop after 8; max 100). Only difference: the depth given to the network. Tests = the Day 2 pipeline
(`day2_queue.sh`, Day 1 recipe), results in `results/review2_day2/<name>/`, figures in
`figures/6_review2_day2/5_more_epochs_true_depth/` and `6_dav2_depth_input/`.
- Both stopped at epoch 14, best at **epoch 6**. More-epochs run: best fake-water validation 23.79 dB vs v1's
  23.86 at its end → **v1 was already at its minimum**; more training does not improve it.

  | (all Day 1 fixes) | UIEB PSNR | UIEB ΔE | fins ΔE | red-starved ΔE | charts SfM | charts DAv2+1 | fake test | dive separation |
  |---|---|---|---|---|---|---|---|---|
  | v1 | **19.09** | **12.25** | 18.7 | **16.7** | 10.3° / 10.3° | 10.8° / 10.2° | **21.9** | 1.29 |
  | `day2f_added_more_epochs_true_depth` | 18.97 | 12.48 | **18.3** | 16.8 | 9.8° / 9.8° | 10.4° / 9.7° | 21.8 | **1.30** |
  | `day2e_added_dav2_depth` | 18.97 | 12.69 | 18.5 | 17.7 | **9.3° / 8.9°** | **9.6° / 8.8°** | 20.8* | 0.77 |

  *fake test and dive separation give the network true depth, which the DAv2 run was not trained on.
- **Reading:** more epochs = level (UIEB slightly worse, charts ~0.5° better, within noise). DAv2 depth = better on
  the 2 chart frames (~1°) but worse on UIEB photos, which use exactly DAv2 depth, and on red-starved photos;
  dive separation collapses (dives read alike). Likely cause: DAv2 (indoor model) makes different mistakes on
  NYU rooms than on real underwater scenes, so the network learned DAv2's *room* errors. Neither replaces v1, so
  the touch-up is not retrained.

## Later (planned 3 Nov, superseded by the section above): train with the depth we test with (DAv2)
- **Gap:** in training the network gets the *true* NYU depth; on real photos (UIEB, the depth-only chart test) it
  gets Depth Anything V2's *estimate* (wrong scale, blurred edges, fooled by fog). It never learned what DAv2
  depth looks like.
- **No new ground truth needed:** the network predicts the water, not depth. We make the fake water ourselves,
  so its settings are known exactly.
- **Recipe:**
  1. NYU room + true depth → render fake water with the **true** depth (physically right image).
  2. Run DAv2 on that **underwater-looking** image (as on UIEB), so the estimate is fooled by fog the same way.
  3. Network input = image + **DAv2** depth; loss = the known water settings / clean room, as now.
  4. It learns how DAv2 depth relates to real distance and uses that on real photos.
- **Cost fix:** water is redrawn every batch and DAv2 per batch is too slow. Precompute DAv2 on 2–3 water versions
  per training room (~20 000 depth maps, ~15 min on GPU) and sample from those.
  Cheaper fallback (no DAv2): distort true depth at random (scale, offset, blur); only imitates DAv2's errors.
- **Continue, not from scratch:** continue `baseline_v1_60ep_bigdata` for 20 epochs (same as Day 2 b/c, ~20 min).
  The model already knows the water physics; only the depth input changes.
- **Control run is required:** also continue v1 for 20 epochs with true depth and nothing else changed.
  Otherwise a gain could just be "20 more epochs". DAv2 run beats control → DAv2 helped; level → it did not.
- **Runs:**

  | Step | Model name | Time |
  |---|---|---|
  | Precompute DAv2 depth on train rooms (2–3 water versions each) | – | ~15 min |
  | Continue 20 epochs, DAv2 depth input | `day2e_added_dav2_depth_input` | ~20 min |
  | Control: continue 20 epochs, true depth | `day2f_control_20ep_more_true_depth` | ~20 min |
  | Test both with Day 1b: 90 UIEB val, colour charts, fins | – | ~10 min |

- **Only if day2e clearly beats day2f:** a from-scratch DAv2 version overnight (~1.5 h) to confirm.
- **Expect:** the fins tint will not change (its red is gone). Possible gains: fog amount and distance handling on
  ordinary photos (charts, UIEB ΔE). If it helps: "we closed the gap between training and testing".
- **Code still to write:** DAv2 precompute script (`depth.py` `predict(..., "metric-indoor")` already exists),
  a depth-source option in `train.py`, the control run. Stop `wec_rl` jobs and run `thermal_guard.py` first.

## Day 3: small "touch-up" network on top
- Physics restores the colours the photo still has. A second small network, trained on real examples,
  fills in colour that was destroyed completely (e.g. the yellow fins).
- Training data: the 800 UIEB photos with their corrected versions, which we already have.
- The main model still reports the water measurements, so we stay interpretable (not a black box).
- Framing: physics restores what the camera recorded; the touch-up's colours are educated guesses.

### Day 3 results (done 3 Nov)
Script `src/day3_touchup.py` (cache → train → eval); results `results/review2_day3_touchup/`, figures
`figures/8_review2_day3_touchup/gallery/` (first 25 val: raw | physics | 3 touch-ups | reference).
- **Networks (start as "do nothing", trained 120 epochs, best checkpoint kept):**
  - `day3a_touchup_global` (0.1 M params): from a 128 px thumbnail, one colour matrix + one tone curve per
    channel for the whole photo. Cannot paint detail or halos.
  - `day3b_touchup_global_plus_local` (0.2 M): the same + a small per-pixel correction network.
  - `day3c_control_touchup_on_raw`: the global touch-up trained on the **raw** photo (no physics). Shows what
    physics adds.
- **Split:** UIEB train 800 → 680 train / 120 held out (fixed seed), used to pick the epoch and global vs local.
  Rule fixed before running: lowest held-out ΔE wins. Held out: global 9.36, local 9.51, control 9.84
  (start 11.51; raw start 14.00) → **global chosen**. Both overfit after ~60 epochs (held-out ΔE rises again);
  best epochs 50 / 60 / 70.
- **90 UIEB val photos (used once):**

  | | PSNR | ΔE | ΔE edges | specks | better than Day 1b (ΔE) |
  |---|---|---|---|---|---|
  | Physics (v1 + Day 1b) | 19.11 | 12.22 | 12.56 | 6.57 | – |
  | **+ touch-up global (chosen)** | **21.68** | **9.66** | 9.95 | 6.57 | 73/90 |
  | + touch-up global + local | 21.86 | 9.53 | 9.86 | 6.77 | 74/90 |
  | Control: touch-up on raw | 22.63 | 10.24 | 10.79 | 6.30 | 59/90 |

  Chosen vs physics: ΔE −2.56 (95 % CI −3.34 to −1.83), PSNR +2.57 dB (CI +1.9 to +3.2).
- **Fins photo (val #2):** ΔE 18.6 → 12.6, the fins turn **yellow** (control 9.5, also yellow).
- **Grey charts (2 Sea-thru frames, never seen in training):** physics 10.3° / 10.6° → global touch-up
  5.1° / 4.8° (local 4.9° / 2.8°); **control on raw stays at 10.6° / 10.0°.** Same with DAv2 depth.
  Reading: on UIEB the control is close (better PSNR, worse ΔE), because it learns UIEB's editing style. On
  real dive charts only physics + touch-up gets closer to true grey. Only 2 frames from one dive: indicative.
- **Weak spots:** some photos get darker / more saturated than the reference (#20 ΔE 9.5 → 13.5, #22 4.2 →
  6.6, #86 +5.5). The pink veil of #24 is reduced (18.2 → 15.0) but not gone; the control does better there.
- **More training data: LSUI (done 3 Nov).** 4279 underwater photos + chosen references (Peng et al., U-shape
  Transformer). `src/prep_lsui.py` reads `data/lsui/LSUI.zip` (470 MB, kept; the 3.1 GB prepared file was
  deleted after use, rebuild ~20 min). Duplicate check (48 px grey-thumbnail correlation, checked by eye): 2 LSUI
  photos = UIEB train photos, dropped; closest to any UIEB val photo 0.90 (not a duplicate) → test set clean.
  Same held-out rule as before.

  | Touch-up | held-out ΔE | val PSNR | val ΔE | fins ΔE | charts SfM | charts DAv2 |
  |---|---|---|---|---|---|---|
  | `day3a_touchup_global` (UIEB only, **still chosen**) | **9.36** | 21.68 | **9.66** | 12.6 | 5.1° / 4.8° | 5.7° / 4.7° |
  | `day3d_touchup_global_plus_lsui` (UIEB + LSUI mixed) | 10.79 | 20.56 | 11.11 | 13.8 | 6.0° / 8.8° | 7.1° / 8.5° |
  | `day3e_touchup_global_plus_local_plus_lsui` | 10.58 | 20.87 | 10.90 | 13.3 | 7.3° / 7.3° | 7.4° / 7.2° |
  | `day3f_touchup_global_lsui_then_uieb` (LSUI, then 40 epochs UIEB only, lr 3e-4) | 9.54 | 21.59 | 9.96 | 10.3 | 4.5° / 4.3° | 5.2° / 4.0° |

  Reading: mixing LSUI in is clearly worse everywhere (val ΔE +1.5 vs day3a, CI +0.8 to +2.1), including the
  charts. Its references follow a different editing style, so the touch-up learns an in-between style.
  LSUI-then-UIEB is level with day3a on UIEB (ΔE +0.3, CI −0.2 to +0.8, i.e. no real difference), slightly
  better on the 2 chart frames and the fins, but the pre-set rule (held-out ΔE) keeps day3a. More data of a
  different "preferred style" does not help; the limit is the references, not the number of photos.
- **For the talk:** physics removes the water and keeps the model interpretable (water readings per photo);
  the touch-up corrects the colour that physics cannot recover; the control shows the touch-up alone is
  not enough on real dive charts. The fins' yellow is an educated guess learned from UIEB, not recorded light.

## Day 4: evaluation and slides
- New measures: a **speck score** (fine-grained noise in the red channel) and the **share of pixels where
  fog > signal**. Report results on a "hard" subset: UIEB images with almost no red.
- Fins slide: original → Day 1 → Day 2 → Day 3, side by side.

## Story for the talk
"The professor saw the fins problem. We traced it to the model over-removing fog when red is almost
gone, and to colour balancing amplifying noise. We fixed how the model is applied (Day 1), fixed the
training data so it covers this water (Day 2), and added a learned touch-up for colour that is truly
lost (Day 3)."

## Notes for Method 2
- Only the product K_d·d_s can be found from one image; d_s in metres needs a known depth (dive computer)
  or one anchor.
- K_d (diffuse attenuation) is not the same as β_D (beam attenuation).
- E0 (surface sunlight) is roughly a known white, so it can be fixed instead of predicted.
- E is only the sunlight colour; fog (β_B, B∞), β_D and the range map are still needed to remove water.
- The fins photo has the same near-zero red, so Method 2 will hit the same limit.
