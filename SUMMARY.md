# PROJECT SUMMARY: read this first when resuming

OE5510 (ML for Ocean Engineering), Group 4. **Intermediate review on 29 Oct 2026** (15-min group talk;
rubric mapping in `docs/presentation_plan.md`). Last updated: 28 Oct 2026, end of session 3 (longer base training
and verdict, §3c). Presenter's plain-language guide: `docs/presentation_guide.md`.

> **Talk structure (29 Oct):** **PART 1 = BetaNet before fine-tuning** (§1–3, figures fig0–fig14): cover fully.
> **PART 2 = fine-tuning on real dives** (§3b, figures **fig15–fig20**; fig15 is the first fine-tuning figure):
> presented after Part 1 as attempts, mostly failed, one partial success.
> **Optional slide = the longer training run** (§3c, `figures/3_new_runs/`): "training longer helped synthetic accuracy,
> not real-dive separation". The main slides stay on the original model (`beta_rgbd`), since every Part 1 figure is built on it.

# PART 1: BetaNet before fine-tuning

## 1. What changed and why
- Original proposal (`underwater_proposal.pptx`): enhance images (UIEB-trained CNN), then check whether YOLO detection improves.
- **Professor's new direction:** a NN that finds the **two water coefficients, attenuation β_D and
  backscatter β_B**, that remove the effect of water. Output = the coefficients and/or the clean image.
  "Akkaynak already did something like this" = **Sea-thru (CVPR 2019)**: classical per-image estimation
  of the same coefficients given a range map (not a NN). The closest *neural* prior work is DeepSeeColor
  (ICRA 2023, per-scene optimisation). Full review: `docs/research_notes.md`.

## 2. Our approach: BetaNet
`photo + range map → ResNet-18 → 12 numbers (β_D, β_B, B∞, E per R,G,B) → closed-form physics → clean image`
- Model: `I = J·E·e^{−β_D z} + B∞(1 − e^{−β_B z})` (Akkaynak–Treibitz revised model + sunlight colour E).
- Training data: NYU-v2 RGB-D (in-air, metric depth) + **synthetic water with known coefficients**,
  sampled continuously around the 10 Jerlov types (UWCNN N_λ table), β_B = β_D × U(0.5, 2),
  B∞ = albedo × sunlight colour, E = e^{−K·d}, exposure jitter, sensor noise. New water every batch.
- Loss = coefficient error (log-L1) + image error (transmission-weighted L1 after the physics inversion).
- 11 M params, ~10 min training, **3.2 ms / image** inference (RTX 3050, 240×320).

## 3. Results (all numbers come from `results/*.json`)

**Depth** (24 real Sea-thru frames vs SfM, `depth_eval.json`): DAv2 relative (aligned) AbsRel 0.07;
metric-indoor as-is 0.76; metric-indoor + 1 range reading 0.12; metric-outdoor as-is 2.59.
→ **Shape right, scale wrong; one metric anchor fixes most of it.**

**Coefficients, synthetic test** (654 unseen NYU rooms, random water; `figure_numbers.json`)
Median relative error: β_D 14/20/23 % (R/G/B, R² 0.71/0.84/0.87); β_B 22/23/24 %; B∞ 13/16/17 %; E 8/2/1 %.

**Images, synthetic test** (all 654; `synth_eval.json`)
| method | PSNR | SSIM | ΔE₀₀ |
|---|---|---|---|
| raw | 14.5 | 0.62 | 24.3 |
| gray-world | 15.4 | 0.62 | 17.8 |
| Sea-thru classical, true depth (first 112 only) | 16.3 | 0.71 | 17.5 |
| **BetaNet RGB-D** | **20.0** | 0.64 | 14.0 |
| BetaNet + DAv2 depth estimated from the photo | 19.5 | 0.63 | 14.3 |
| BetaNet, no depth input | 19.3 | 0.65 | 14.9 |
| BetaNet, coefficient loss only | 20.1 | 0.65 | 14.1 |
| BetaNet, image loss only | 20.7 | 0.67 | 13.9 |
| UNet black box (RGB-D) | 25.1 | 0.81 | 8.1 |
| oracle: true coefficients | 25.1 | 0.68 | 9.3 |
Per Jerlov type (PSNR, BetaNet vs raw): ocean I 25.3 vs 16.8 … coastal 5 18.8 vs 12.5; coastal 9 10.3 vs 11.0
(**in the murkiest water even the oracle fails**: the signal is gone).
**Identifiability finding:** image-loss-only gets *better* image PSNR (20.7) but β_D error 33 % vs 19 %
(fig14). Good images do not imply right coefficients → coefficient supervision is needed.

**Real data** (`real_eval.json`)
| | raw | gray-world | Sea-thru classical (SfM) | BetaNet v2 + GW (SfM) | BetaNet v2 + GW (DAv2+1) | UNet black box |
|---|---|---|---|---|---|---|
| grey-patch error D1 3272 (°) | 29.4 | 8.9 | 15.7 | 11.8 | 12.1 | **2.6** |
| grey-patch error D1 3425 (°) | 29.4 | 8.1 | 16.6 | 11.1 | 10.9 | **3.9** |
| UIEB PSNR (dB) | 16.4 | 17.7 | – | **18.0** | – | 17.6 |
| UIEB ΔE₀₀ ↓ | 17.5 | **13.8** | – | 13.9 | – | 15.7 |
Published for D1 3272: raw 26°, Sea-thru 8°, DeepSeeColor 14° (our raw 29.4° is close, so the patch marking looks right).
Only 2 chart frames, so treat these as indicative. The UNet imposes warm indoor (NYU) colours: grey patches
come out neutral, but D5's water turns pink and it has the worst UIEB colour error. **Trade-off: black box
= best synthetic numbers, no coefficients, hallucinated colours; BetaNet = interpretable, physically
consistent, behind on synthetic PSNR.** → Next: hybrid (BetaNet physics + residual refiner, e.g. the friend's BridgeNet).

**Water-type sensor** (fig11): β_D within one dive varies 5–20 %; turbid D4 has the highest
β_D (R 0.38/m), clear D5 the lowest (R 0.20/m). No labels used.

**Failed experiment → fix:** v1 (no sunlight colour E) blamed missing red on range → β_D(red) hit the
output cap → magenta real images (`results/v1_no_illuminant/`). v2 adds E: magenta gone, UIEB PSNR
15.9 → 18.0 dB (with GW).

**Speed:** BetaNet 3.2 ms, UNet 10 ms, classical Sea-thru 0.38 s (240×320) / 1.2 s (600×400) → ~120× faster.

# PART 2: After BetaNet: fine-tuning on real dives (failed attempts + one partial success)

## 3b. Session 2 (28 Oct): fine-tuning on real dives (`docs/project_overview.md` §10 has the full story)
**Data:** streamed every 1st frame of D3/D4/D5, every 2nd of D2 and every 4th of D1 from HF, shrunk to 300 px
high and deleted the originals: 658 frames with SfM range in `data/seathru_train/` (494 MB, of a 38 GB dataset).
Held out: all of D3 (unseen dive), the 24 eval frames, and D1 frames within ±25 of the 2 chart frames. 552 train frames.
UIEB train split + DAv2 depth in `data/uieb_train.npz` (800 images, 459 MB).

**Method (`src/finetune.py`):** every step = one synthetic NYU batch (supervised, keeps the units) + 8 real crops
with label-free physics losses: *dark* (darkest 1 % per range bin on the backscatter curve), *range* (restored colour
must not drift with distance), *gray* (gray-world after dividing by E), optional *dive* (same dive → same β) and
*teacher* (L1 on log β_D, β_B, B∞ to the frozen starting model). BatchNorm frozen, lr 5e-5, 3000 steps ≈ 10 min.
Selection rule fixed in `src/select_ft.py`: synthetic β_D error ≤ 1.1× before, between-dive spread ≥ 0.5× before,
then lowest held-out physics loss. Chart frames and UIEB are reported, never used to choose.

| model | chart, gray-world (2 frames) | own E, no GW | UIEB PSNR / ΔE₀₀ | D3 colour drift | dive spread | synth β_D err |
|---|---|---|---|---|---|---|
| before (`beta_rgbd`) | 11.8 / 11.1° | 15.8 / 17.0° | 18.01 / 13.86 | 0.473 | 0.24 | 19.1 % |
| v1 all losses (`ft_seathru`) | 10.0 / 8.6° | 9.9 / 9.2° | 17.56 / 14.26 | 0.264 | **0.05 collapsed** | 18.7 % |
| v1 without synthetic anchor | 9.8 / 8.4° | 10.8 / 9.8° | 17.14 / 15.16 | 0.256 | 0.06 | **69.7 %** |
| **no same-dive loss (`ft_nodive`, selected)** | **9.4 / 8.2°** | 9.7 / 10.0° | 17.99 / 14.00 | **0.269** | 0.13 | 18.7 % |
| no dive loss + teacher 0.1 | 11.1 / 9.6° | 12.3 / 12.3° | 18.11 / 14.09 | 0.375 | 0.25 | 19.0 % |
| no dive loss + teacher 0.3 | 11.5 / 10.2° | 13.4 / 13.6° | **18.27 / 13.84** | 0.414 | 0.25 | 18.9 % |
| dive loss + teacher 1.0 | 12.1 / 11.4° | 13.4 / 13.8° | 17.97 / 13.91 | 0.443 | 0.25 | 18.8 % |
| v1 + UIEB physics losses | 9.8 / 8.5° | 9.6 / 9.2° | 16.02 / 16.42 | – | 0.06 | 18.7 % |
| v1 + UIEB + reference loss | 9.8 / 8.5° | 9.5 / 9.3° | 17.19 / 14.86 | – | 0.06 | 18.9 % |
Gray-world alone: chart 8.9 / 8.1°, UIEB 17.72 / 13.79. Synthetic test PSNR is 20.4 for every anchored model (20.0 before).

**Findings to present (failed attempts first):** (1) real fine-tuning fixes colour on held-out dives (chart error with the network's own E
halves: 16–17° → ~10°, i.e. gray-world is no longer needed) without hurting synthetic accuracy. (2) **Collapse:**
with the same-dive loss every dive gets one "average water" (β_D red 0.14–0.16 everywhere); the images still look
fine, so only the dive-spread check catches it (identifiability again). (3) No synthetic anchor → coefficients
lose their units (β_D error 70 %). (4) UIEB fine-tuning made UIEB *worse* (DAv2 depth has the wrong scale, so the
range-based losses are fed wrong distances). (5) The teacher weight is a dial: 0 = best colour (chart 8.8° mean),
0.1–0.3 = keeps the per-dive water read-out fully and gives the best UIEB (18.3 dB, beats gray-world) — fig20.
Visual caveat: fine-tuned outputs are neutral but flatter; open-water pixels (no SfM depth → treated as far) wash out white.
Note: the selection rule was amended once after round 2 (first spread metric divided by within-dive spread and let
the collapsed model pass); documented in `select_ft.py`.

## 3c. Session 3 (28 Oct evening): longer base training; review scope
**Scope:** the talk follows the Part 1 / Part 2 structure above. Fine-tuning is shown as attempts. Its next version
needs rebalanced losses: a lower gray-world weight (it flattens colourful scenes) and no same-dive loss. Presenter
guide: `docs/presentation_guide.md`.

**Longer run `beta_rgbd_big`:** same model trained from scratch on 49 NYU shards (7,428 frames, `data/nyu_train_big.npz` via
`src/prep_nyu_more.py`; the same 288 validation frames), 60 epochs, murky water (mean β_D > 0.8 /m) kept at 40 % → 2 % of
training (`physics.sample_water(murky_keep=…)`, `train.py --murky-keep`). The process ended at epoch 48 (laptop at 95 °C;
the thermal guard was not running), then continued from epoch 47 with `--resume-epoch` (the LR schedule is fast-forwarded,
AdamW moments restart). Best checkpoint = epoch 59.
Results (`results/synth_eval_big.json`, `real_eval_big.json`): synthetic 20.0 → **21.2 dB**, ΔE 14.0 → 13.5; β_D error
14/20/23 → 12/16/20 %, β_B 22/23/24 → 17/17/20 %; +0.8 to 1.4 dB in every Jerlov type. Real: chart unchanged (11.7/11.2°);
UIEB + GW 18.0 → 18.5 dB, ΔE 13.9 → 13.2 (now beats gray-world). Real β all ~15–25 % lower; D4 no longer clearly the
most turbid. Per-image: better by >3 dB in ~1/3 of rooms, worse by >3 dB in ~1/10. Cyan classroom example (NYU test 120, 3C): the
cause is the haze estimate, not β_D. Swapping coefficients one at a time: β_D alone gives 26.2 dB, B∞ alone 15.5 dB. At 8 m
>90 % of red is haze, so a 7 % red-haze over-estimate wipes red out, and leftover green haze is amplified ×6.
UIEB per image: better on 58/90 (`figures/3_new_runs/uieb_original_vs_longer.png`, `src/fig_uieb_compare.py`).
Sea-thru (`seathru_original_vs_longer.png`, `src/fig_seathru_compare.py`): images nearly the same, but the per-dive read-out
is worse. D4 (turbid) is no longer distinct (0.38 → 0.28, ties D1). Untested guess: less murky training water; test with
`--murky-keep 1.0`.
**Verdict** (`src/analyze_old_vs_new.py` → `results/old_vs_new_analysis.json`, `figures/3_new_runs/old_vs_new_verdict.png`;
also `ft_big_nodive` = the ft_nodive recipe on the new base):
- Synthetic: +1.21 dB, CI 1.02–1.39. Better on 70 % of rooms; the largest gain is in far rooms (+1.4).
- β_D and β_B errors are significantly lower. UIEB + GW: +0.47 dB, CI 0.22–0.72.
- The chart frames are unchanged. Dive separation is worse: 1.57 → 1.29.
- Converged: val coef error is flat over the last 10 epochs.
- Fine-tuning the new base gives the chart (own E) 8.6°, D3 drift 0.27 and synth 21.5, but separation 1.19.
- Decision: the new base model going forward. For the 29 Oct talk, keep the original for the main slides and add one
  "training longer" slide. Rebalance the fine-tuning losses (less gray, small teacher weight) before fine-tuning the new base. Figures: `figures/3_new_runs/` (loss curves, examples, results).

## 4. Files
```
SUMMARY.md                  ← this file
docs/research_notes.md      literature (Akkaynak etc.), datasets, depth methodology, metrics, friend's repo review
docs/presentation_plan.md   slide-by-slide plan mapped to the rubric + speaker split
docs/qa_prep.md             likely questions with answers
docs/presentation_guide.md  plain-language guide for presenting: equations, data, losses, results, Q&A, longer run
figures/1_original_runs (fig0…fig14, Part 1), 2_original_finetune (fig15…fig20, Part 2), 3_new_runs (longer run; see figures/README.md)
src/physics.py              image-formation model, Jerlov water sampler, restore()
src/model.py                BetaNet (ResNet-18 → 12 numbers), UNetDirect (black-box baseline)
src/train.py                training (on-the-fly synthetic water); --no-depth, --lambda-img, --lambda-coef
src/eval_synth.py           synthetic test (+ DAv2 depth from the underwater photo, + per Jerlov type, timing)
src/eval_real.py            Sea-thru frames (charts, site consistency, UNet) + UIEB
src/run_classical_seathru.py  classical Sea-thru baseline (friend's code; separate process)
src/prep_nyu.py, prep_seathru.py, download_data.py, depth.py (DAv2), metrics.py
src/download_seathru_all.py streams Sea-thru from HF, shrinks, deletes originals (--sites, --stride; resumable)
src/prep_uieb.py            UIEB train split + DAv2 depth → data/uieb_train.npz
src/finetune.py             real-dive fine-tuning (physics losses + synthetic anchor; --w-dive/--w-teacher/--w-uieb/--w-ref)
src/eval_finetune.py        before/after: synthetic test, held-out physics losses, per-dive β (merges into finetune_eval.json)
src/select_ft.py            fixed selection rule → results/ft_selected.json
src/fig_finetune.py         fig15–fig20 (before / collapsed / selected)
src/thermal_guard.py        CPU-temperature guard for project processes (not written in session 2; leave it running)
results/ft_*/               fine-tuned checkpoints + log.json; real_eval_<name>.json, *_outputs_<name>.npz per model
results/ft_queue*.sh        the three fine-tuning queues (2 CPU threads, nice 10, 3-min cool-downs)
src/fig_physics.py (fig1-3), fig_method.py (fig4-5), fig_results.py (fig0, 6-12, 14), fig_plan.py (fig13), vizstyle.py
src/prep_nyu_more.py        streams 30 more NYU shards → data/nyu_train_big.npz (7,428 frames; same 288 val frames last)
src/eval_synth_big.py       original vs longer run on the synthetic test → results/synth_eval_big.json (synth_eval.json untouched)
src/analyze_old_vs_new.py   paired verdict analysis (4 models) → results/old_vs_new_analysis.json + old_vs_new_verdict.png
src/fig_loss_curves.py, fig_big.py, fig_uieb_compare.py, fig_seathru_compare.py   figures in figures/3_new_runs/
results/beta_rgbd_big/      longer run (best = epoch 59); results/ft_big_nodive/ = ft_nodive recipe on it
data/                       NYU shards (19/157 train + val), nyu_train_big.npz (49 shards), UIEB, Sea-thru frames, preprocessed .npz
results/<run>/best.pt|log.json; results/logs/*.log; run_all.sh (training queue); finalize.sh (evals + figures)
external/De-water/          friend's repo (read-only reference)
```

## 5. Reproduce (venv: /home/aarav-patel/wec_rl/.venv, do NOT create another)
```bash
PY=/home/aarav-patel/wec_rl/.venv/bin/python; cd src
$PY download_data.py nyu uieb seathru && $PY prep_nyu.py && $PY prep_seathru.py
bash ../results/run_all.sh                       # 4 BetaNet variants + UNet, ~1.2 h on RTX 3050
CUDA_VISIBLE_DEVICES="" $PY run_classical_seathru.py
$PY eval_synth.py && $PY eval_real.py && $PY fig_results.py
$PY fig_physics.py && $PY fig_method.py && $PY fig_plan.py
```
Packages added to the venv (additive; torch/numpy untouched): opencv-python-headless, scikit-image,
transformers, huggingface_hub, python-pptx, pyarrow, lpips, torchvision.
Gotchas: 4 GB GPU, so the UNet needs --bs 8 and DAv2 runs at bs 2 while training. `pkill -f train.py` kills your own
shell (use PIDs). The De-water Sea-thru module monkey-patches np.tanh, so run it in its own process. The UNet needs
image sides divisible by 8 (reflect-pad).
Session-2 gotchas: `eval_synth.py` evaluates *every* `results/*/best.pt`, which now includes the ft_* runs and a
leftover `results/_smoke/` test run (delete `_smoke/` and `ft_pilot/` by hand; they are throwaway). Keep torch at
2–4 threads (`OMP_NUM_THREADS`) on this laptop; unthrottled training took the CPU to 86 °C. Launching with
`nohup … &` from Claude once stalled on a hidden permission prompt for 3 h; use the tool's background mode.

Longer run (session 3):
```bash
$PY prep_nyu_more.py                                   # ~10 min download, +2.85 GB
$PY train.py --model beta --name beta_rgbd_big --data train_big --nval 288 --epochs 60 --murky-keep 0.4   # ~45 min
$PY eval_synth_big.py && $PY eval_real.py --ckpt beta_rgbd_big --tag _big
$PY finetune.py --init beta_rgbd_big --name ft_big_nodive --steps 3000 --w-dive 0
$PY eval_real.py --ckpt ft_big_nodive --tag _ft_big_nodive && $PY eval_finetune.py beta_rgbd beta_rgbd_big ft_nodive ft_big_nodive
$PY analyze_old_vs_new.py && $PY fig_loss_curves.py && $PY fig_big.py && $PY fig_uieb_compare.py && $PY fig_seathru_compare.py
```
Session-3 gotchas: GPU training with 4 threads still reached 95 °C after ~30 min, and `thermal_guard.py` was not
running. Use 2 threads plus a watchdog (pause above 85 °C, resume below 70 °C). Resuming from `best.pt` restarts the
AdamW moments (it holds weights only). New figures get descriptive filenames, not numbers.

## 6. Next steps (fig13_plan.png; step 3 done in session 2)
0. New base model = `beta_rgbd_big` (converged; better coefficients). First check why its dive separation dropped:
   retrain with `--murky-keep 1.0`. Then fine-tune it with rebalanced losses (lower `--w-gray`, small `--w-teacher`).
   Add better real metrics: colour consistency with distance (no reference needed) and more chart frames. Colour
   (non-grey) patches can only be used with the camera's colour response, which we don't have.
1. Realism: forward-scatter blur + marine snow (De-water `watering_engine.py`), outdoor/natural RGB-D.
2. Range-dependent β_D(z) (Sea-thru 2-exponential form).
3. ~~Self-supervised fine-tuning on real photos~~ done (§3b). Follow-ups: pick the teacher weight per use
   (colour vs water read-out); mark more chart frames at full resolution (D2/D5 charts are ~20 px at 1/4 res);
   UIEB only with a metric-depth fix (DAv2 scale) or reference-free losses that don't use range.
   Cheap display fixes: cap the amplification for no-depth pixels (open water washes out), contrast stretch.
4. Hybrid: BetaNet physics output + small residual refiner (friend's BridgeNet/NAFNet) → black-box quality, kept interpretable.
5. Full real evaluation: mark all Sea-thru chart frames at full res (D3/D4 charts are rotated and small) + SQUID.
6. Uncertainty head; Jerlov read-out. Stretch: YOLO mAP on restored images (original proposal).
7. Open: the slide deck is not built yet; offered to build it (.pptx) in the old proposal's style.
