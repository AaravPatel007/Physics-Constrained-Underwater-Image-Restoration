# Research notes: NN that estimates the water coefficients (β_D, β_B)

Context: the professor asked us to change the project from "does enhancement help YOLO detection" to
"a neural network that finds the two coefficients that remove the effect of water: the
**attenuation (extinction) coefficient β_D** and the **backscatter coefficient β_B**". Output can be
the coefficients, the restored image, or both.

---

## 1. The physics in one equation (Akkaynak & Treibitz revised model)

```
I_c = J_c · E_c · exp(−β_D,c · z)  +  B∞,c · (1 − exp(−β_B,c · z))        c ∈ {R, G, B}
      └──────── direct signal ───────┘   └──────────── backscatter ───────────┘
```

| symbol | meaning | per image? |
|---|---|---|
| I | the underwater photo (linear RGB) | |
| J | the scene as it would look in air (what we want) | |
| z | range: camera-to-object distance (m), **needs a depth map** | per pixel |
| β_D | attenuation of the direct signal (1/m), "extinction coefficient" | 3 numbers |
| β_B | backscatter coefficient (1/m) | 3 numbers |
| B∞ | veiling light = colour of the water at infinite distance | 3 numbers |
| E | colour of sunlight reaching the scene (filtered by the water column *above*) | 3 numbers |

Inverting it is closed-form once the numbers are known:
`J = (I − B∞(1 − e^{−β_B z})) · e^{β_D z} / E`.
So **if a network predicts these ~12 numbers, the clean image comes for free** and the output is both
interpretable and physically constrained.

Key points from Akkaynak's papers that matter for us:
- **β_D ≠ β_B** (CVPR 2018). The older "atmospheric/haze" model used one coefficient for both and
  that is the source of large errors. That is why the professor says "the 2 coefficients".
- **β_D varies with range z** (and with reflectance and sensor). Sea-thru models it as
  β_D(z) = a·e^{bz} + c·e^{dz}. Our v1/v2 prototype uses one β_D per channel per image; a
  range-dependent β_D is a planned extension (Section 6).
- **β and z only appear as the product β·z.** With relative depth (no metres), β cannot be recovered
  in physical units. You need **metric depth**, or at least one metric anchor.
- **Valid coefficients live on a small manifold** (CVPR 2017, "What is the space of attenuation
  coefficients…"), spanned by Jerlov's oceanic (I–III) and coastal (1–9) water types. We sample
  training water from there.

## 2. Has Akkaynak (or anyone) already done this? Prior work

| Work | What it does | Neural? | How it differs from our plan |
|---|---|---|---|
| Akkaynak et al., CVPR 2017, *Space of attenuation coefficients* | Physically valid RGB β's from oceanographic measurements (Jerlov types) | no | We use it as the prior for synthetic training water |
| Akkaynak & Treibitz, CVPR 2018, *A Revised Underwater Image Formation Model* | Shows β_D ≠ β_B, and their dependencies | no | The model we implement |
| **Akkaynak & Treibitz, CVPR 2019, *Sea-thru*** | Given RGB **+ range map**: backscatter from darkest 1% of pixels per range bin, fits B∞, β_B; β_D(z) from an illuminant map (LSAC); 1,100+ images (D1–D5) with SfM range + colour charts. Avg grey-patch angular error raw 22.3° → 4.94° | **no** (per-image optimisation, slow, offline) | **This is "what Akkaynak already did"**: estimating the two coefficients per image, classically. We learn to *predict* them in one forward pass |
| Levy, …, Akkaynak, …, Treibitz, CVPR 2023, *SeaThru-NeRF* | NeRF with the Sea-thru model; medium params per channel learned per scene | yes, per-scene | Needs many views of one scene + hours of training |
| **Jamieson, How, Girdhar, ICRA 2023, *DeepSeeColor*** | Two tiny CNNs whose *weights are* the Sea-thru parameters; self-supervised losses; up to 60 Hz; matches Sea-thru on charts (e.g. D1 3272: raw 26°, Sea-thru 8°, DSC 14°) | yes, but **fit per scene/stream** | Closest work. Still an optimiser per scene; it does not generalise from a dataset. Ours is an *amortised* predictor trained once on many waters |
| Li, Anwar, Porikli, PR 2020, *UWCNN* | Synthesises 10 Jerlov water types from NYU-v2 RGB-D; one CNN **per water type**, image→image | yes | Same data recipe (we reuse its Jerlov N_λ table), but they never output coefficients |
| Varghese et al., ICCV 2023 | Self-supervised depth + restoration with the revised model | yes | Joint depth; complex; good "future" reference |
| Bar Nathan, Levy, Treibitz, Rosenbaum, ECCV 2024, *Osmosis* | RGBD diffusion prior; water parameters estimated during sampling | yes (diffusion) | Very slow; strong but heavy |
| Cai & Metzler 2025 (arXiv 2507.02148) | Benchmarks monocular metric depth underwater (FLSea, SQUID); fine-tunes DAv2 on synthetic underwater Hypersim | yes | Confirms our depth finding (Section 3) |

**Our angle (what is new):** one feed-forward network, trained once, that outputs β_D, β_B, B∞
(and the illuminant E) for *any* new underwater image in about 3 ms, and recovers the image through the
physics equation. It is ~120× faster than classical Sea-thru, with no per-scene optimisation (unlike
DeepSeeColor, SeaThru-NeRF and Osmosis), and its output is interpretable: the coefficients say what kind
of water it is.

## 3. Depth maps: which dataset and what methodology

The model needs range z in metres. Three sources:

| Use | Dataset | Depth source | Size we use | Notes |
|---|---|---|---|---|
| **Train** (synthetic water) | **NYU Depth v2** (HF `vikhyatk/nyu_depth_v2`) | Kinect, metric, 0.5–10 m | 2,881 frames from 19 of 157 shards (every 2nd frame) + **official 654 test rooms** held out | Same recipe as UWCNN; domain gap: indoor scenes |
| **Real test with metric depth** | **Sea-thru D1–D5** (HF `bhowmikabhimanyu/seathru-dataset`, 38 GB total) | SfM range maps (Agisoft) | 24 frames (5 per site + the paper's chart frames), ~0.8 GB | Linear RAW 16-bit; colour charts; the gold standard for this task |
| **Real test, no depth** | **UIEB** (HF `Hikari0608/UIEB`, 1.5 GB) | **Depth Anything V2** (monocular) | 90 val pairs | References are "preferred" enhancements, not ground truth |
| Optional | FLSea (HF, 64 GB), SQUID (stereo + charts, Berman et al. 2020) | SfM / stereo | not downloaded | For later robustness tests |

**Our measured finding on DAv2 underwater** (24 Sea-thru frames vs SfM ground truth;
`results/depth_eval.json`):

| DAv2-Small variant | AbsRel ↓ | δ<1.25 ↑ |
|---|---|---|
| relative (needs scale + shift, fitted to GT) | **0.072** | **0.967** |
| metric-indoor, used as is | 0.759 | 0.246 |
| metric-indoor, ×1 scale from GT median | 0.117 | 0.860 |
| metric-outdoor, used as is | 2.593 | 0.082 |
| metric-outdoor, ×1 scale | 0.123 | 0.863 |

→ **DAv2 gets the *shape* of underwater scenes right but not the *scale*.** Since only β·z
appears in the physics, a wrong scale gives a wrong β. Practical fix used on real vehicles:
**one metric anchor** (altimeter/DVL range, stereo baseline, laser scaler, or known object size).
We emulate that as "DAv2 + 1 range reading" (median rescale).

**Methodology chosen**
1. Train on NYU-v2 RGB-D + *synthetic water* with **known** β_D, β_B, B∞, E (supervised).
2. Water sampled continuously around the 10 Jerlov types (β = −ln N_λ, UWCNN Table I), with
   turbidity scaling, per-channel jitter, β_B = β_D × U(0.5, 2) (they differ), B∞ = albedo ×
   depth-filtered sunlight, illuminant E = e^{−K·d_v}, exposure jitter and sensor noise.
3. Test on held-out NYU rooms (synthetic, coefficients known), then on **real** Sea-thru frames
   (SfM depth vs DAv2 depth) and UIEB (DAv2 depth).

## 4. Metrics

| Question | Data | Metric |
|---|---|---|
| Are the predicted coefficients right? | synthetic test (GT known) | MAE / median relative error per coefficient; predicted-vs-true scatter, R² |
| Is the restored image right? | synthetic test | **PSNR, SSIM, CIEDE2000 (ΔE₀₀)** vs the clean in-air image |
| Are colours *physically* correct on real photos? | Sea-thru frames with charts | **Grey-patch angular error ψ** (°), the Sea-thru / DeepSeeColor / SQUID metric |
| Are the coefficients physically consistent? | Sea-thru D1–D5 sequences | Spread of β within one dive site vs between sites (same water ⇒ same β; no labels needed) |
| How good is the depth? | Sea-thru SfM | AbsRel, RMSE, δ<1.25 |
| Fast enough for an AUV? | any | ms / image |
| Perceptual quality (secondary) | UIEB | PSNR/SSIM/ΔE₀₀ vs UIEB references, UCIQE / UIQM |

Caveats to state in the talk: UIEB references are subjective "best looking" outputs, not ground truth;
UCIQE/UIQM are known to reward over-saturation, so report them but never optimise for them.

## 5. Friend's repo (github.com/na23b068-svg/De-water) assessment

Public; cloned to `external/De-water` (read-only reference; we did not modify it).

**What it is:** DAv2 depth → a Sea-thru re-implementation → a 7-channel NAFNet "BridgeNet" that maps
[raw, Sea-thru output, depth] → clean image. Trained on 300 NYU-v2 scenes "watered" with 3 fixed
Jerlov profiles.

**What we reuse (saves work):**
- `src/pipeline/seathru_pipeline.py`: classical Sea-thru with a given range map. **We use it as our
  classical baseline** (`src/run_classical_seathru.py`; ~1.2 s per 600×400 frame on CPU).
- `src/data/watering_engine.py`: ideas for extra realism (forward-scatter blur, marine snow,
  turbulence). Planned as augmentations for the next phase.
- NAFNet code: candidate for an optional refinement stage after the physics inversion.

**Why we did not just use it as-is:**
- It is image→image. **It never outputs β_D / β_B**, which is what the professor asked for.
- Only **3 fixed water profiles**, so "predicting the coefficients" on that data reduces to a 3-way
  classification. We sample a continuous range of water instead.
- Its own README reports that on 30 real photos BridgeNet does **not** beat Sea-thru (UCIQE 0.41 vs 0.49).
- DAv2 relative depth is mapped linearly to an assumed 1.8–7.5 m. That is exactly the scale problem
  we measured in Section 3.
- `underwater_metrics.py` UIQM is non-standard (UISM = 10 × mean Sobel; UIConM weight 0.575 vs
  the paper's 3.5753), so its numbers are not comparable to the literature.
- The Sea-thru port adds non-physical steps (percentile stretch, a red-channel boost) and
  monkey-patches `np.tanh` globally on import, so we run it in a separate process.

## 6. Ideas to impress (cheap → expensive)

1. **Physics layer inside the network** (done in the prototype): the network outputs the coefficients,
   a differentiable physics layer outputs the image, and training uses both losses. You get both outputs
   asked for, and they are interpretable.
2. **"The camera is a water-quality sensor"** (done, needs only inference): the predicted β's
   identify the dive site / water type, and stay stable across frames of the same dive.
3. **Real-time** (done): 3.2 ms per image on a laptop RTX 3050 vs 0.38 s (same size) for the classical
   pipeline. Matters for AUVs.
4. **Self-supervised fine-tuning on real images** (done 28 Oct, `docs/project_overview.md` §10): Sea-thru-style
   losses (dark-pixel backscatter, colour constant with range, gray-world) on 552 real Sea-thru frames plus a
   synthetic anchor batch. Works on Sea-thru (chart error with the model's own E 16–17° → ~10°). A same-dive
   consistency loss collapses all dives to one water. On UIEB it hurts, because DAv2 depth has the wrong scale.
5. **Range-dependent β_D(z)**: predict the 4 Sea-thru parameters of β_D(z) instead of a constant.
6. **Uncertainty**: predict a variance per coefficient, so the network flags when it can't know
   (e.g. flat scenes with no depth variation).
7. **Hybrid: physics first, then a learned refiner.** Our black-box UNet beats BetaNet on synthetic data
   but hallucinates indoor colours on real water. Feeding BetaNet's physics output to a small residual
   network (the friend's 7-channel BridgeNet/NAFNet is exactly this design) should keep the
   coefficients and the interpretability while closing the image-quality gap.
8. Stretch: the original proposal's YOLO detection as a downstream test of the restored images.

## Sources
- Sea-thru (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf
- Revised model (CVPR 2018): https://openaccess.thecvf.com/content_cvpr_2018/papers/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.pdf
- Space of attenuation coefficients (CVPR 2017): https://openaccess.thecvf.com/content_cvpr_2017/papers/Akkaynak_What_Is_the_CVPR_2017_paper.pdf
- DeepSeeColor (ICRA 2023): https://arxiv.org/abs/2303.04025
- SeaThru-NeRF (CVPR 2023): https://arxiv.org/abs/2304.07743
- Osmosis (ECCV 2024): https://arxiv.org/abs/2403.14837
- UWCNN (PR 2020): https://arxiv.org/abs/1807.03528
- Varghese et al. (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/papers/Varghese_Self-supervised_Monocular_Underwater_Depth_Recovery_Image_Restoration_and_a_Real-sea_ICCV_2023_paper.pdf
- Cai & Metzler 2025, underwater monocular metric depth: https://arxiv.org/abs/2507.02148
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Datasets on Hugging Face: `vikhyatk/nyu_depth_v2`, `bhowmikabhimanyu/seathru-dataset`, `Hikari0608/UIEB`, `bhowmikabhimanyu/flsea-vi`
