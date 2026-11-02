# Review 2 plan (Method 1: keep improving BetaNet)

> **START HERE (state at end of 2 Nov 2026).** Model names: `docs/MODELS.md`.
> - Method 1 model in use = `baseline_v1_60ep_bigdata` (60 epochs, big data) **+ Day 1 fixes** (applied at restore time, no retraining).
> - Day 1 helped (UIEB 18.48 → 19.09 dB, ΔE 13.2 → 12.3, fins specks 10.8 → 4.4). Fins still blue-green.
> - Day 2 = three retrains (`day2a_added_murky_water`, `day2b_added_deep_red_starved_water`, `day2c_added_camera_flaws`); none fixed the tint, all are side results. Conclusion: not a training-range problem.
> - **Next: Day 3** (small touch-up network on the 800 UIEB train pairs, on top of the current model), then Day 4 (evaluation + slides). Review 2 is ~5–6 Nov.
> - Machine notes: run `src/thermal_guard.py` while training, stop the user's `wec_rl` jobs, python = `/home/aarav-patel/wec_rl/.venv/bin/python`. `src/day2_queue.sh` trains + evaluates a run in one go (`THREADS=4` allowed).
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

### Does Day 1 still matter after Day 2? Yes.
- Day 2 fixes the model's **estimates** (fog amount, water colour, sunlight colour).
- Day 1 fixes **what we do with the estimates** (subtracting, boosting).
- Even with perfect estimates, 2–3 levels of noisy red still go negative after subtraction and still
  explode when boosted. We already saw this: in the murkiest water, even the true coefficients failed.
- Also: Day 1 is a result in hand if Day 2 runs late, and "Day 2 with vs without Day 1" shows how much
  each fix contributes.

## Day 3: small "touch-up" network on top
- Physics restores the colours the photo still has. A second small network, trained on real examples,
  fills in colour that was destroyed completely (e.g. the yellow fins).
- Training data: the 800 UIEB photos with their corrected versions, which we already have.
- The main model still reports the water measurements, so we stay interpretable (not a black box).
- Framing: physics restores what the camera recorded; the touch-up's colours are educated guesses.

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
