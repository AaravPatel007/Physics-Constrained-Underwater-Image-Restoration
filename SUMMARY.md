# PROJECT SUMMARY: read this first when resuming

OE5510 (ML for Ocean Engineering), Group 4. **Intermediate review due 29 Oct 2026** (15-min talk; rubric in
`docs/presentation_plan.md`). Last updated: 27 Oct 2026.

## 1. What changed and why
- Original proposal (`underwater_proposal.pptx`): enhance images (UIEB-trained CNN) → does YOLO detection improve?
- **Professor's new direction:** a NN that finds the **two water coefficients, attenuation β_D and
  backscatter β_B**, that remove the effect of water. Output = the coefficients and/or the clean image.
  The professor said Akkaynak already did something similar → that is **Sea-thru (CVPR 2019)**: classical
  per-image estimation of the same coefficients given a range map (not a NN). The closest *neural* prior work
  is DeepSeeColor (ICRA 2023, per-scene optimisation). See `docs/research_notes.md`.

## 2. Our approach ("BetaNet")
`photo + range map → ResNet-18 → 12 numbers (β_D, β_B, B∞, E per R,G,B) → closed-form physics → clean image`
- Model: `I = J·E·e^{−β_D z} + B∞(1 − e^{−β_B z})` (Akkaynak–Treibitz revised model + illuminant E).
- Training data: NYU-v2 RGB-D (in-air, metric depth) + **synthetic water with known coefficients** sampled
  around the 10 Jerlov types (UWCNN N_λ table), β_B ≠ β_D, B∞ from albedo × sunlight colour, E = e^{−K d}.
  New random water every batch (`src/physics.py`, `src/train.py`).
- Loss = coefficient error (log-L1) + image error (transmission-weighted L1 after the physics inversion).
- Both outputs asked for: the coefficients *and* the image. Interpretable, physically constrained, ~2 ms/image.

## 3. Key results so far (numbers in `results/*.json`)
**Depth (24 real Sea-thru frames vs SfM ground truth, `results/depth_eval.json`)**
DAv2 relative (aligned) AbsRel 0.07 | metric-indoor as-is 0.76 | metric-indoor + 1 range reading 0.12 |
metric-outdoor as-is 2.59. → **Shape right, scale wrong; one metric anchor fixes it.**

**Failed experiment → fix (good for the talk)**
- v1 had no illuminant term. On real Sea-thru frames shot 1–2 m from the reef but 10 m deep, it blamed all
  the missing red on range, β_D(red) saturated at the output cap (1.65/m), and the images went magenta.
  Archived in `results/v1_no_illuminant/`.
- v2 adds E (sunlight colour after the water column) to the data and the network. Magenta gone. UIEB
  PSNR 15.9 → 17.4 dB (BetaNet alone) / 18.0 dB (+ gray-world, Sea-thru's own final step).

**Real data, v2** (`results/real_eval.json`)
| | raw | gray-world | Sea-thru classical (friend's port, SfM) | BetaNet v2 | BetaNet v2 + gray-world |
|---|---|---|---|---|---|
| Grey-patch error D1 3272 (°) | 29.4 | **8.9** | 15.7 | 15.8 | 11.8 |
| Grey-patch error D1 3425 (°) | 29.4 | **8.1** | 16.6 | 17.0 | 11.1 |
| UIEB PSNR vs reference (dB) | 16.4 | 17.7 | – | 17.4 | **18.0** |
| UIEB ΔE₀₀ ↓ | 17.5 | **13.8** | – | 13.8 | 13.9 |
Published on D1 3272: raw 26°, Sea-thru 8°, DeepSeeColor 14° (our raw 29.4° ≈ theirs, so the patch marking looks right).
Honest reading: on the charts gray-world wins (charts at 1–1.5 m, low haze = best case for gray-world).
Visually BetaNet removes the *distance-dependent haze* that gray-world leaves (D2, D3 in fig10).
Known failure: open-water background (no SfM depth) goes too blue (D5); turbid D4 a bit dark.

**Synthetic test (654 unseen NYU rooms):** see `results/synth_eval.json`, `results/figure_numbers.json`
(filled in by `src/eval_synth.py` + `src/fig_results.py`). → section 7 below.

## 4. Files
```
SUMMARY.md                  ← this file
docs/research_notes.md      literature (Akkaynak etc.), datasets, depth methodology, metrics, friend's repo review
docs/presentation_plan.md   slide-by-slide plan mapped to the rubric + speaker notes
docs/qa_prep.md             likely questions with answers
figures/                    all figures (fig1…fig13), regenerate with the fig_*.py scripts
src/physics.py              image-formation model, Jerlov water sampler, inversion (restore)
src/model.py                BetaNet (ResNet-18 → 12 numbers), UNetDirect (black-box baseline)
src/train.py                training (on-the-fly synthetic water); --no-depth, --lambda-img, --lambda-coef
src/eval_synth.py           synthetic test (+ DAv2 depth from the underwater photo, + per-Jerlov-type)
src/eval_real.py            Sea-thru frames (charts, site consistency) + UIEB
src/run_classical_seathru.py  classical Sea-thru baseline (friend's code, separate process)
src/prep_nyu.py, prep_seathru.py, download_data.py, depth.py (DAv2 wrapper), metrics.py
src/fig_physics.py, fig_method.py, fig_results.py, fig_plan.py, vizstyle.py
data/                       downloaded datasets (NYU shards, UIEB, 24 Sea-thru frames) + preprocessed .npz
results/<run>/best.pt|log.json, results/logs/*.log, results/run_all.sh (training queue)
external/De-water/          friend's repo (read-only reference)
```

## 5. How to reproduce (venv: /home/aarav-patel/wec_rl/.venv, do NOT create another)
```bash
PY=/home/aarav-patel/wec_rl/.venv/bin/python; cd src
$PY download_data.py nyu uieb seathru && $PY prep_nyu.py && $PY prep_seathru.py
bash ../results/run_all.sh                 # 4 BetaNet variants + UNet, ~1 h on RTX 3050
CUDA_VISIBLE_DEVICES="" $PY run_classical_seathru.py   # classical baseline (CPU)
$PY eval_synth.py && $PY eval_real.py && $PY fig_results.py
$PY fig_physics.py && $PY fig_method.py && $PY fig_plan.py
```
Packages added to the venv (additive only; torch/numpy untouched): opencv-python-headless, scikit-image,
transformers, huggingface_hub, python-pptx, pyarrow, lpips, torchvision.
GPU: 4 GB. UNet needs --bs 8. DAv2 batch 2 when sharing the GPU (`depth.predict(bs=...)`).
Gotchas: `pkill -f train.py` kills your own shell (use PIDs); the De-water Sea-thru module monkey-patches
np.tanh, so always run it in its own process.

## 6. Next steps (also fig13_plan.png)
1. Realism: forward-scatter blur + marine snow (De-water `watering_engine.py`), outdoor RGB-D.
2. Range-dependent β_D(z) (Sea-thru 2-exponential form).
3. Self-supervised fine-tuning on real photos (dark-pixel backscatter loss + grey-world), as in DeepSeeColor.
4. Full real evaluation: mark all Sea-thru chart frames at full res (D3/D4 charts are rotated and small) + SQUID.
5. Uncertainty head; Jerlov read-out. Stretch: YOLO mAP on restored images (original proposal).
6. Open: slides not built yet (offer to build the .pptx in the old proposal's style).

## 7. Synthetic results (fill/refresh after eval)
(see below; updated automatically by hand after `fig_results.py`)
