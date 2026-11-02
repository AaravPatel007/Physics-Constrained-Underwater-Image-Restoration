# BetaNet: learning the water coefficients of an underwater photo

OE5510 (Machine Learning for Ocean Engineering), Group 4. Status as of 28 Oct 2026, the day before the
intermediate review (29 Oct 2026). Section 10 (fine-tuning on real dives) is new since 27 Oct.

This document is the self-contained account of the project: what we set out to do, why the direction
changed, every design decision and the alternatives we rejected, what we built, what the results are,
what failed, and what comes next. It assumes basic familiarity with CNNs but not with underwater optics.
Exact numbers come from `results/*.json`; `SUMMARY.md` is the shorter working log.

---

**How to read this document / talk order.** **Part 1 = Sections 1–9** (BetaNet trained on synthetic water only,
figures fig0–fig14): this is the finished system and gets the main time at the review. **Part 2 = Section 10**
(fine-tuning on real dives, figures fig15–fig20, i.e. fig15 is where fine-tuning starts): attempts made afterwards,
mostly failed, one partial success. Sections 11–13 are plan, repository map and glossary.

---

# PART 1: BetaNet before fine-tuning

## 1. The problem in one paragraph

Water changes the colour of everything the camera sees, and the change depends on distance. Red light is
absorbed within a few metres, so distant objects look blue-green. Particles in the water also scatter
light back into the lens, which adds a haze (backscatter) that grows with distance. Ordinary "image
enhancement" networks learn to make photos look nicer, but they don't tell you *what the water did*. The
goal of this project is a neural network that estimates the physical water parameters for a photo and
uses them to undo the water, so the output is both a set of interpretable coefficients and a clean image.

## 2. How the project got here

| Stage | What | Why it changed |
|---|---|---|
| Original proposal (`underwater_proposal.pptx`) | Train a CNN enhancer on UIEB, then test whether YOLO detection improves on enhanced images | The professor asked for a more physical, ML-for-ocean-engineering direction |
| **Professor's new direction** | A NN that finds the **two coefficients that remove the effect of water: attenuation β_D and backscatter β_B**. Output the coefficients, the clean image, or both. He noted that "Akkaynak already did something like this" | — |
| What "Akkaynak's work" means | **Sea-thru** (Akkaynak & Treibitz, CVPR 2019). It estimates the same coefficients per image, but by classical per-image fitting given a range map, not with a network | We position our work against it |
| Our response | **BetaNet**: a network that predicts the coefficients in one forward pass, plus a fixed physics layer that turns them into the clean image | Current state |

The original YOLO idea is kept as a stretch goal (Section 10).

## 3. The physics we build on

We use the Akkaynak–Treibitz revised image formation model (CVPR 2018), with one extra term for the
colour of the sunlight that reaches the scene:

```
I_c = J_c · E_c · exp(−β_D,c · z)  +  B∞,c · (1 − exp(−β_B,c · z))        for c ∈ {R, G, B}
      └───────── direct signal ─────────┘   └──────────── backscatter ────────────┘
```

| Symbol | Meaning | How many |
|---|---|---|
| I | the underwater photo, in **linear** RGB (not gamma-encoded sRGB) | per pixel |
| J | the scene as it would look in air: what we want back | per pixel |
| z | range: camera-to-object distance in metres (needs a depth map) | per pixel |
| β_D | attenuation of the direct signal (1/m) | 3 per image |
| β_B | backscatter coefficient (1/m) | 3 per image |
| B∞ | veiling light: the colour of the water at infinite distance | 3 per image |
| E | colour of the sunlight at the scene, already filtered by the water above it | 3 per image |

Once the 12 numbers are known, the inversion is closed form:

```
J = (I − B∞ · (1 − e^{−β_B z})) · e^{β_D z} / E
```

So a network only has to predict 12 numbers; the image comes for free, and it can never "invent"
content, because it can only apply a physically valid per-channel correction.

Three facts from Akkaynak's papers shaped the design:

1. **β_D ≠ β_B** (CVPR 2018). The older haze model uses one coefficient for both, which is a large
   error source. That is why the professor said "the two coefficients".
2. **Only the product β·z appears.** If the depth map has the wrong scale, the network must compensate by
   predicting a wrong β. Recovering β in physical units (1/m) needs metric depth.
3. **Real coefficients live on a small manifold** (CVPR 2017), spanned by the 10 Jerlov ocean and coastal
   water types. That gives us a realistic prior for generating training water.

## 4. Prior work and where we fit

| Work | What it does | Neural? | Difference from ours |
|---|---|---|---|
| Sea-thru (CVPR 2019) | Given RGB + range map: backscatter from the darkest pixels per range bin; spatially varying illuminant; range-dependent β_D(z). Grey-patch error 22.3° → 4.9° on its dataset | No: per-image optimisation, ~1 s per image | "What Akkaynak already did". We predict the same quantities in one pass |
| DeepSeeColor (ICRA 2023) | Tiny CNNs whose weights *are* the Sea-thru parameters, fitted with self-supervised losses | Yes, but re-fitted per scene | Closest neural work. It doesn't generalise across a dataset; ours is trained once |
| SeaThru-NeRF (CVPR 2023) | NeRF with the Sea-thru medium model | Per scene, hours | Needs many views of one scene |
| UWCNN (PR 2020) | Synthesises the 10 Jerlov water types on NYU-v2; one CNN per water type | Yes, image→image | We reuse its data recipe and Jerlov table, but it never outputs coefficients |
| Osmosis (ECCV 2024) | RGB-D diffusion prior, water parameters estimated during sampling | Yes, very slow | Strong but heavy |
| Friend's repo "De-water" (public, GitHub `na23b068-svg/De-water`) | DAv2 depth → Sea-thru port → NAFNet refiner. Trained on 3 fixed water types | Yes, image→image | No coefficients; 3 waters make coefficient prediction a 3-way classification. We reuse its Sea-thru code as our classical baseline |

**Our claim:** one feed-forward network, trained once, that outputs β_D, β_B, B∞ and E for any new
underwater image in about 3 ms, with no per-scene optimisation, and restores the image through the
physics.

## 5. Decisions we made, and what we rejected

| Decision | Chosen | Rejected, and why |
|---|---|---|
| What the network outputs | 12 global numbers per image, then a fixed physics layer | Image→image network: no coefficients, can hallucinate (confirmed by our UNet baseline, Section 8). Per-pixel maps: far harder to supervise, not needed for a first version |
| Where training labels come from | **Synthetic water** applied to in-air RGB-D images, so the true coefficients are known exactly | Real underwater photos: nobody has ground-truth β per image. UIEB references: subjective, no depth (Section 9.2) |
| In-air RGB-D source | **NYU Depth v2** (Kinect, metric 0.5–10 m). 2,881 training frames; the official 654 test rooms held out | Outdoor RGB-D (Hypersim, DIODE): better domain match, planned next. NYU is the standard for this (UWCNN uses it) |
| How water is sampled | Continuous sampling around the 10 Jerlov types (β = −ln N_λ from UWCNN), turbidity scaling, per-channel jitter, β_B = β_D × U(0.5, 2), B∞ = albedo × filtered sunlight, E = e^{−K·d}, exposure jitter, sensor noise. **New water every batch** | 3 fixed waters (friend's repo): the network would only learn to classify |
| Network | ResNet-18 encoder on 4 channels (RGB + depth), regression head → 12 positive numbers; 11 M parameters | Larger backbones: not needed at this data size, and we want real-time |
| Loss | Coefficient error (log-L1) **plus** image error (transmission-weighted L1 after the physics inversion) | Image loss only: tested, it gives slightly better images but much worse coefficients (Section 8.4) |
| Depth at test time | True depth where available (Sea-thru SfM); otherwise **Depth Anything V2** (DAv2) | Other monocular depth models: DAv2 is current state of the art and runs on our 4 GB GPU |
| Colour space | All physics in linear RGB. Sea-thru frames are linear RAW; UIEB sRGB is linearised first | Working in sRGB breaks the physics (the model is linear in light) |
| Final colour step on real photos | Gray-world white balance after removing the water, as Sea-thru itself does | — |

## 6. Data

All data is in `data/` (about 8 GB), downloaded from Hugging Face by `src/download_data.py`.

| Dataset | Used for | Depth | What we have |
|---|---|---|---|
| NYU Depth v2 (`vikhyatk/nyu_depth_v2`) | Training + synthetic test | Kinect, metric | 20 of 157 shards (2,881 train frames) + 654 test rooms |
| Sea-thru D1–D5 (`bhowmikabhimanyu/seathru-dataset`) | Real test: colour charts, within-dive consistency | SfM range maps, metric | 24 frames (5 per site + the paper's chart frames) of ~1,100; full set is 38 GB |
| UIEB (`Hikari0608/UIEB`) | Real test: perceptual quality | none (we use DAv2) | 90 validation pairs |
| FLSea, SQUID | Planned robustness tests | SfM / stereo | not downloaded |
| LSUI, DUO (from the proposal's backup slide) | Not needed for the new direction | — | not downloaded |

### Depth: the key finding

Measured on the 24 Sea-thru frames against SfM ground truth (`results/depth_eval.json`):

| DAv2-Small variant | AbsRel ↓ | δ<1.25 ↑ |
|---|---|---|
| relative depth, scale + shift fitted to ground truth | 0.07 | 0.97 |
| metric-indoor, used as is | 0.76 | 0.25 |
| **metric-indoor + one range reading** (median rescale) | **0.12** | 0.86 |
| metric-outdoor, used as is | 2.59 | 0.08 |

DAv2 gets the **shape** of underwater scenes right and the **scale** wrong. Because only β·z enters the
physics, a wrong scale means a wrong β. On a real vehicle one metric reading fixes most of it: an
altimeter or DVL range, a stereo baseline, a laser scaler or an object of known size. We emulate that
as "DAv2 + 1 range reading".

## 7. Evaluation protocol

| Question | Data | Metric |
|---|---|---|
| Are the coefficients right? | Synthetic test (truth known) | Median relative error, R² |
| Is the restored image right? | Synthetic test | PSNR, SSIM, ΔE₀₀ (CIEDE2000 colour difference) |
| Are colours physically right on real photos? | Sea-thru frames with colour charts | Grey-patch angular error (°), the Sea-thru / DeepSeeColor metric |
| Are the coefficients consistent? | Sea-thru D1–D5 | Spread of β within one dive vs between dives (same water ⇒ same β) |
| Perceptual quality | UIEB | PSNR / ΔE₀₀ vs the UIEB references (these are *not* ground truth) |
| Depth quality | Sea-thru SfM | AbsRel, δ<1.25 |
| Real-time? | any | ms per image |

Baselines: raw input, gray-world white balance, classical Sea-thru (friend's port, given true depth),
a black-box UNet trained on the same synthetic data (RGB-D → clean image), and an oracle that inverts
the physics with the *true* coefficients (upper bound for any 12-number method).

## 8. Results

### 8.1 Coefficients (654 unseen NYU rooms, random water)

| | R | G | B |
|---|---|---|---|
| β_D median rel. error (R²) | 14 % (0.71) | 20 % (0.84) | 23 % (0.87) |
| β_B | 22 % | 23 % | 24 % |
| B∞ | 13 % | 16 % | 17 % |
| E | 8 % | 2 % | 1 % |

### 8.2 Images

| | Raw | Gray-world | Classical Sea-thru | **BetaNet** | Black-box UNet | Oracle |
|---|---|---|---|---|---|---|
| Synthetic PSNR (dB) | 14.5 | 15.4 | 16.3* | **20.0** | 25.1 | 25.1 |
| Synthetic ΔE₀₀ ↓ | 24.3 | 17.8 | 17.5* | 14.0 | 8.1 | 9.3 |
| UIEB PSNR / ΔE₀₀ | 16.4 / 17.5 | 17.7 / 13.8 | – | **18.0** / 13.9 | 17.6 / 15.7 | – |
| Grey-patch error, Sea-thru D1 3272 | 29.4° | 8.9° | 15.7° | 11.8° | 2.6° | – |
| Grey-patch error, Sea-thru D1 3425 | 29.4° | 8.1° | 16.6° | 11.1° | 3.9° | – |

\* classical Sea-thru run on the first 112 test images only (it is slow). BetaNet real-photo rows include
the final gray-world step. Published numbers for D1 3272: raw 26°, Sea-thru 8°, DeepSeeColor 14°.

Ablations (synthetic PSNR): with DAv2 depth estimated from the underwater photo 19.5; no depth input 19.3.

Per water type: clear ocean I goes 16.8 → 25.3 dB; coastal 5 goes 12.5 → 18.8 dB; in coastal 9 (the
murkiest) nothing works, not even the oracle (10.3 vs raw 11.0), because the signal is physically gone.

### 8.3 Real dives and speed

- Within a dive, β_D varies 5–20 %. Turbid D4 has the highest β_D(red) at 0.38 /m, clear D5 the lowest at
  0.20 /m. No labels were used: the camera works as a rough water-quality sensor.
- BetaNet 3.2 ms per image, UNet 10 ms, classical Sea-thru 0.38 s at the same 240×320 size: about 120×
  faster (RTX 3050 laptop GPU).

### 8.4 Failed experiments and what they taught us

1. **v1 turned real photos magenta.** v1 had no sunlight term E. At 10 m depth red is lost on the
   sunlight's way down from the surface, but v1 could only blame missing red on the distance to the reef,
   so β_D(red) hit its upper limit and the inversion over-boosted red. Adding E (v2) fixed it; UIEB PSNR
   went 15.9 → 18.0 dB. (`results/v1_no_illuminant/`)
2. **Good images don't guarantee right coefficients.** Image-loss-only training gave a slightly better
   image (20.7 dB) but β_D error 33 % vs 19 % (fig14). Several coefficient combinations explain the same
   photo, so coefficients must be supervised.
3. **The black box wins on synthetic data and fails on real water.** The UNet paints indoor NYU colours
   (one dive's water turns pink), has the worst UIEB colour error, and gives no coefficients.
4. **Gray-world alone beats us on the 2 chart frames.** Only 2 frames, so indicative only, but it shows
   the domain gap between NYU rooms and real reefs.

## 9. Limitations

- Training scenes are indoor rooms; real reefs have different textures, colours and depth ranges.
- One β_D per channel per image; real β_D varies with range (Sea-thru uses a two-exponential β_D(z)).
- One global illuminant E; real light varies across the frame.
- The real chart evaluation uses 2 frames. UIEB references are subjective "preferred" images.
- No forward scatter (blur) or marine snow in the synthetic water yet.

# PART 2: After BetaNet: fine-tuning on real dives (mostly failed attempts)

## 10. Fine-tuning on real dives (session 2, 28 Oct)

The model in Sections 5–8 (`results/beta_rgbd`) was trained only on synthetic water over indoor rooms. This
section adapts it to real underwater photos.

### 10.1 Data

The Sea-thru dataset has no clean reference images, only photos, SfM range maps and colour charts in some
frames. That is enough for physics-based (label-free) losses. The full set is 38 GB, so each frame was
downloaded, shrunk to 300 px high and the original deleted (`src/download_seathru_all.py`): **658 frames,
494 MB** (all of D3, D4, D5; every 2nd frame of D2; every 4th of D1, since consecutive frames are near
duplicates).

| Split | Frames | Why |
|---|---|---|
| Train | 552 (D1, D2, D4, D5) | |
| Test: unseen dive | all 68 of D3 | does it generalise to new water? |
| Test: chart scenes | D1 frames within ±25 of the 2 chart frames | no leakage into the chart test |
| Test: evaluation frames | the 24 frames of Sections 6–8 | same frames as the "before" numbers |

### 10.2 Method (`src/finetune.py`)

Every training step uses two batches:
- **Synthetic NYU batch with the original supervised loss.** This keeps the coefficients in physical units.
- **8 real crops with label-free physics losses**, the same cues classical Sea-thru uses:

| Loss | What it checks |
|---|---|
| dark | The darkest 1 % of pixels at each distance should lie on the predicted backscatter curve (asymmetric: predicting more haze than the darkest pixel is impossible) |
| range | After removing the water, the average colour of near and far parts of the scene should match. This pins β_D |
| gray | Gray-world on the fully restored image. This pins the sunlight colour E |
| dive (optional) | Frames from the same dive should get the same coefficients |
| teacher (optional) | Stay near the starting model's coefficients on real crops (weight = how near) |

BatchNorm statistics are frozen, the learning rate is 5e-5, and 3,000 steps take ~10 min.

The model is chosen by a rule fixed in advance (`src/select_ft.py`):
1. Synthetic β_D error no more than 10 % worse than before.
2. The five dives must still get clearly different coefficients (between-dive spread ≥ half of before).
3. Among the models that pass, the lowest physics losses on held-out frames.

The 2 chart frames and UIEB are only reported, never used for the choice.

### 10.3 Results

| Model | Chart error, + gray-world | Chart error, own E | UIEB PSNR / ΔE₀₀ | D3 colour drift | Dives told apart (spread) |
|---|---|---|---|---|---|
| Before | 11.8 / 11.1° | 15.8 / 17.0° | 18.01 / 13.86 | 0.47 | 0.24 |
| v1: all losses | 10.0 / 8.6° | 9.9 / 9.2° | 17.56 / 14.26 | 0.26 | **0.05 (collapsed)** |
| **No same-dive loss (selected)** | **9.4 / 8.2°** | 9.7 / 10.0° | 17.99 / 14.00 | **0.27** | 0.13 |
| No same-dive loss + teacher 0.1 | 11.1 / 9.6° | 12.3 / 12.3° | 18.11 / 14.09 | 0.38 | 0.25 |
| No same-dive loss + teacher 0.3 | 11.5 / 10.2° | 13.4 / 13.6° | **18.27 / 13.84** | 0.41 | 0.25 |

For reference, gray-world alone scores 8.9 / 8.1° on the charts and 17.72 / 13.79 on UIEB. Every
anchored model keeps the synthetic test intact: PSNR 20.4 dB (20.0 before) and β_D error ~19 %.

What this shows (figures 15–20):
1. **Real fine-tuning fixes colour on held-out dives.** Without any gray-world step, the chart error halves
   (16–17° → ~10°): the network now estimates the sunlight colour itself.
2. **Collapse (failed experiment).** With the same-dive loss, every dive gets one "average water" (β_D red
   0.14–0.16 everywhere, where before the turbid canyon read 0.38 and a clear reef 0.21). The images still look
   fine, so only the per-dive check catches it. This is the identifiability problem of §8.4 again: many
   coefficient sets explain the same photo (fig18).
3. **No synthetic anchor (failed experiment):** the coefficients lose their physical units (synthetic β_D error
   70 %, PSNR 14.3 dB), even though the real-image losses look just as good.
4. **UIEB fine-tuning made UIEB worse (failed experiment):** 16.0–17.2 dB vs 18.0. UIEB has no depth, and DAv2's
   wrong scale feeds the distance-based losses wrong distances.
5. **The teacher weight is a dial** (fig20). Weight 0 gives the best colour on the dives. Weight 0.1–0.3 keeps the
   per-dive water read-out exactly and gives the best UIEB score of any model (18.3 dB, ahead of gray-world).
6. **Visual caveat:** the fine-tuned images are neutral but flatter, and open-water pixels (no SfM depth,
   treated as far away) wash out white.

One note on the rule: it was amended once, after round 2. Its first spread measure divided by the spread within
each dive, which also shrinks in a collapse, so the collapsed v1 passed. It now uses the spread between dive
averages directly.

## 10b. Longer base training (session 3, 28 Oct evening)

The original 40-epoch run was past its elbow but still improving when the learning rate reached zero. We
trained the **same network from scratch** with more data and more epochs (`beta_rgbd_big`):

| | Original (`beta_rgbd`) | Longer run (`beta_rgbd_big`) |
|---|---|---|
| NYU training images | 2,881 (19 shards) | 7,428 (49 shards; same 288 validation frames) |
| Epochs / gradient steps | 40 / ~4.3k | 60 / ~17.8k |
| Very murky water (mean β_D > 0.8 /m) | 4.9 % | 2.0 % (`--murky-keep 0.4`) |

Paired analysis on the same images (`src/analyze_old_vs_new.py`, `results/old_vs_new_analysis.json`,
`figures/3_new_runs/old_vs_new_verdict.png`):

| | Original | Longer run | Longer run + fine-tune (`ft_big_nodive`) |
|---|---|---|---|
| Synthetic PSNR (654 rooms) | 20.0 | **21.2** (+1.21, 95 % CI 1.02–1.39; better on 70 %) | 21.5 |
| Median β_D / β_B error | 20 / 23 % | **17 / 18 %** | 16 / 18 % |
| UIEB PSNR + gray-world | 18.0 | **18.5** (+0.47, CI 0.22–0.72) | 18.2 |
| Chart error, own E / + gray-world | 16.4 / 11.4° | 16.9 / 11.5° | **8.6 / 9.0°** |
| Dive separation (higher = waters told apart) | **1.57** | 1.29 | 1.19 |
| Colour drift with distance, unseen dive D3 | 0.47 | 0.41 | **0.27** |

Findings:
1. **It has converged**: validation error is flat over the last 10 epochs, and training ≈ validation loss.
2. **Better at the coefficients and on synthetic and UIEB images**, with tight confidence intervals. Gains are largest
   in far rooms. One far classroom went cyan; the cause is the haze estimate (B∞), not β_D. Far away the photo is
   mostly haze, so small haze errors wipe out red.
3. **Worse at telling real dives apart**: turbid D4 no longer stands out and all real β drop 15–25 %. Untested guess:
   less murky training water (retrain with `--murky-keep 1.0`).
4. **Fine-tuning stacks**: it gives the best real colour so far, but it lowers separation further, so the losses
   need rebalancing (less gray-world, a small teacher weight).

Decision: the longer run becomes the base model going forward. The review's main slides stay on the original
model (every Part 1 figure is built on it), with one optional "training longer" slide.

# Plan, reference

## 11. Next steps

0. **Adopt `beta_rgbd_big` as the base.** Check why its dive separation dropped (retrain with `--murky-keep 1.0`),
   then fine-tune it with rebalanced losses. Add a distance-consistency metric for real photos.
1. **Choose the teacher weight per use:** 0 for colour correction, 0.1–0.3 when the coefficients are the output.
   A per-image uncertainty head would help decide.
2. **More realistic synthetic data:** outdoor/natural RGB-D (Hypersim, DIODE), forward-scatter blur and
   marine snow (ideas from the friend's `watering_engine.py`).
3. **Range-dependent β_D(z)** (predict the 4 Sea-thru parameters) and a **spatially varying illuminant**.
4. **Hybrid:** BetaNet physics output + a small residual refiner (the friend's 7-channel NAFNet design),
   aiming for black-box image quality while keeping the coefficients.
   - **Done (3 Nov 2026, branch `RefineNet`):** `RefineNet`, 0.12M params on frozen `baseline_v1_60ep_bigdata`
     (input: BetaNet restoration + photo + range; zero-init gamma-space residual). Trained on synthetic only
     (`refine.py`) it lifts the 654 test rooms 21.21 → 23.86 dB (68 % of the gap to the oracle's 25.11) but
     *loses* 1.1 dB on 160 held-out real UIEB photos (17.68 → 16.54), the friend's BridgeNet failure again.
     Fine-tuned on 640 UIEB pairs with a synthetic batch every step and an L1 anchor to BetaNet's output
     (`finetune_refine.py`, the guards from §10): **real 20.20 dB (+2.52 vs BetaNet), synthetic 23.43 dB
     (+2.22)**, better on both. Coefficients unchanged. UIEB scores are against references, not ground truth.
     Figures: `figures/8_refinenet/`; numbers: `results/refine_{v1,ft}/*.json`.
5. **Full real evaluation:** mark every Sea-thru chart frame at full resolution, add SQUID (stereo depth +
   charts).
6. Uncertainty per coefficient; Jerlov water-type read-out. Stretch: YOLO mAP on restored images.

## 12. Repository map and how to reproduce

```
SUMMARY.md                   short working log (read first when resuming)
docs/project_overview.md     this document
docs/research_notes.md       literature, datasets, depth method, metrics, friend's repo review
docs/presentation_plan.md    slide-by-slide plan mapped to the review rubric
docs/qa_prep.md              likely questions with answers
docs/presentation_guide.md   plain-language presenter's guide (equations, data, losses, results, Q&A)
figures/{1_original_runs,2_original_finetune,3_new_runs}  all figures (fig0 before/after and fig10 real Sea-thru are the strongest;
                             fig15–fig20 = fine-tuning on real dives)
src/physics.py               image formation model, Jerlov water sampler, restore()
src/model.py                 BetaNet, UNetDirect baseline
src/train.py                 training with on-the-fly synthetic water
src/eval_synth.py, eval_real.py, run_classical_seathru.py   evaluation
src/fig_*.py                 figure scripts
src/finetune.py              real-dive fine-tuning; eval_finetune.py, select_ft.py, fig_finetune.py around it
src/download_seathru_all.py  streams Sea-thru frames, shrinks them, deletes the originals
src/prep_nyu_more.py, eval_synth_big.py, analyze_old_vs_new.py   longer run: data, eval, paired verdict
results/<run>/best.pt        trained weights (beta_rgbd = synthetic-only model; ft_nodive = selected fine-tune;
                             beta_rgbd_big = longer run; ft_big_nodive = fine-tune of it); .pt files are not in git
external/De-water/           friend's repo, read-only reference
```

```bash
PY=/home/aarav-patel/wec_rl/.venv/bin/python; cd src
$PY download_data.py nyu uieb seathru && $PY prep_nyu.py && $PY prep_seathru.py
bash ../results/run_all.sh                         # 4 BetaNet variants + UNet, ~1.2 h on an RTX 3050
CUDA_VISIBLE_DEVICES="" $PY run_classical_seathru.py
$PY eval_synth.py && $PY eval_real.py && $PY fig_results.py
# real-dive fine-tuning (session 2)
$PY download_seathru_all.py --sites D3 D4 D5 && $PY download_seathru_all.py --sites D2 --stride 2 \
  && $PY download_seathru_all.py --sites D1 --stride 4 && $PY prep_uieb.py
bash ../results/ft_queue.sh; bash ../results/ft_queue2.sh; bash ../results/ft_queue3.sh   # ~1.5 h in total
$PY fig_finetune.py ft_nodive ft_seathru ft_nodive_t01 ft_nodive_t03 ft_teacher ft_seathru_noanchor ft_uieb ft_uieb_ref
# longer base training + verdict (session 3; full command list in SUMMARY.md §5)
$PY prep_nyu_more.py && $PY train.py --model beta --name beta_rgbd_big --data train_big --nval 288 --epochs 60 --murky-keep 0.4
$PY analyze_old_vs_new.py
```

## 13. Glossary

- **β_D (attenuation):** how fast the light coming *from the object* fades with distance, per colour.
- **β_B (backscatter):** how fast the haze added *by the water* builds up with distance.
- **B∞ (veiling light):** the colour you see looking into open water, with no object behind it.
- **E (illuminant):** the colour of daylight at the scene's depth below the surface.
- **Jerlov water types:** the oceanographic classification of water clarity, from clear ocean (I) to murky
  coastal (9).
- **Gray-world:** assume the average colour of a scene is grey and rescale channels to match.
- **Grey-patch angular error:** angle between the RGB vector of a grey chart patch and pure grey; 0° is perfect.
- **AbsRel:** mean of |predicted depth − true depth| / true depth.
- **Oracle:** the physics inversion with the true coefficients; the best any 12-number method can do.
