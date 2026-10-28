# Intermediate review: slide plan (15 min, ~14 slides)

Rubric points and where each is covered:
| Rubric item | Slide(s) |
|---|---|
| Intro slide: topic + all names & roll numbers | 1 |
| Importance; impact if not solved | 2 |
| Method explained so a non-expert can follow | 4, 6, 7 |
| Results with graphs | 8, 9, 11, 12 (Part 1); 13, 14 (Part 2) |
| Discussion of unique contributions | 9, 12, 13, 15 |
| Plan of action + milestones | 14 |
| Uniqueness / failed experiments that guided design | 10 (v1→v2), 6 (depth scale), 13 (collapse, no anchor, UIEB) |
| Seamless hand-over | speaker split below |
| Slides not text-heavy | ≤ 3 short lines per slide, figure-first |
| Fielding questions | `docs/qa_prep.md` |

**Structure of the talk.** Part 1 (slides 1–12, ≈ 10 min) is the finished BetaNet, trained on synthetic water only
(figures fig0–fig14): cover it properly. Part 2 (slides 13–14, ≈ 3 min) is what we tried *after* that to adapt it
to real dives (fine-tuning, fig15–fig20): presented honestly as attempts, mostly failed, with one partial success.
Then plan and takeaways. Figure numbers: **fig15 is the first fine-tuning figure**; everything below it is Part 1.

Suggested split for 4 speakers: **A** slides 1–3, **B** 4–7, **C** 8–12, **D** 13–15 (≈ 3.5 min each).
Hand-over line at the end of each block is given in *italics*.

# PART 1: BetaNet before fine-tuning (synthetic training only)

---

**1. Title** (A, 20 s)
"Learning the Water: a neural network that estimates attenuation and backscatter coefficients"
Group 4 · names + roll numbers · OE5510 · 29 Oct 2026. Background: `fig0_hero.png` (raw vs restored dives).

**2. Why this matters** (A, 1 min). Figure: `fig3_attenuation_curves.png`
- Red light is gone after a few metres; haze grows with distance.
- AUV surveys, reef health (colour = bleaching), inspection, species ID all depend on true colour.
- If unsolved: wrong colour measurements, detectors fail in new water, and the best fix (Sea-thru) is offline and slow.

**3. What changed since the proposal** (A, 40 s)
- Proposal: enhance (copy reference images) → detect.
- Feedback: learn the *physics*: the two coefficients β_D (attenuation) and β_B (backscatter).
- New question: can one network read the water's coefficients off a single photo?
*"B will show what these coefficients actually do to a photo."*

**4. The physics in one picture** (B, 1 min). Figure: `fig1_image_formation.png`
- Photo = (scene × sunlight colour × fade with distance) + haze that grows with distance.
- β_D: how fast each colour fades. β_B: how fast the haze builds. B∞: the haze colour. E: the sunlight colour at depth.
- Know these numbers + distance → undo the water exactly.

**5. What Akkaynak did and where we fit** (B, 1 min). Small table:
| | needs | speed | learns from data |
|---|---|---|---|
| Sea-thru (Akkaynak & Treibitz 2019) | photo + depth | seconds–minutes / image, offline | no (optimises per image) |
| DeepSeeColor (2023) | photo + depth | real-time, but re-fits per scene | per scene |
| **Ours** | photo + depth | **~3 ms / image** | **trained once on many waters** |

**6. Data and depth** (B, 1 min). Figures: `fig2_jerlov_types.png` (left), `fig5_depth_eval.png` (right)
- Train: NYU-v2 rooms + synthetic water from the 10 Jerlov ocean types → true coefficients known.
- Test on real: Sea-thru dive photos (true depth, colour charts) and UIEB.
- Finding: Depth-Anything-V2 gets the *shape* right but the *scale* wrong (AbsRel 0.76 → 0.12 with one range reading).

**7. Method** (B, 40 s). Figure: `fig4_method_overview.png`
- ResNet-18 → 12 numbers → physics layer → image. Both outputs the professor asked for.
- Loss = coefficient error + image error. ~10 min training on a laptop GPU.
*"C will show how well it works."*

**8. Does it find the right coefficients?** (C, 1 min). Figure: `fig6_coef_scatter.png`
- 654 rooms never seen in training, random water.
- Median error: β_D 14–23 %, β_B 22–24 %, veiling light 13–17 %, sunlight colour 1–8 %.
- Backup slide: `fig14_identifiability.png`. Training on image quality alone gives *better* PSNR (20.7 vs 20.0)
  but β_D error 33 % vs 19 %. Good images ≠ right coefficients, so the coefficients must be supervised.

**9. Is the image right? vs baselines and ablations** (C, 1 min). Figures: `fig7_synth_results.png`, `fig8_jerlov_breakdown.png`
- PSNR: raw 14.5 → gray-world 15.4 → classical Sea-thru 16.3 → **BetaNet 20.0** (DAv2 depth: 19.5) → oracle 25.1.
- Black-box UNet: 25.1 on synthetic. It wins here because it learns the training distribution, but it gives no coefficients.
- Depth helps (no-depth 19.3). In the murkiest water (coastal 9) even the oracle fails: the signal is physically gone.

**10. A failed experiment that taught us the physics** (C, 1 min). Figure: columns 4 vs 5 of `fig10_real_seathru.png`
- v1 on real dives → magenta images. Why: 10 m deep, red is lost *on the way down*, not between camera and reef.
- Fix: add the sunlight colour E to the data and the network (v2) → magenta gone, UIEB PSNR 15.9 → 18.0 dB.

**11. Real underwater photos** (C, 1 min). Figure: `fig10_real_seathru.png`
- Grey patches (2 chart frames): raw 29° → ours 11–12° (classical Sea-thru port 16°, DeepSeeColor paper 14°,
  gray-world 8°, UNet 3–4°).
- But the UNet paints indoor colours: pink open water in D5, and the worst colour error on UIEB (ΔE 15.7 vs ours 13.9).
  Gray-world balances colour but leaves the haze. Ours removes haze that grows with distance, and gives the coefficients.
- Honest trade-off → next step is a hybrid: physics first, then a small learned refiner.
*"D: what else the coefficients give us, and what's next."*

**12. The camera as a water sensor + speed** (C, 45 s). Figure: `fig11_site_consistency.png`
- Same dive → similar β (5–20 % spread); the turbid dive D4 has the highest β_D (red 0.38 /m), clear D5 the lowest
  (0.20 /m). No labels used.
- 3.2 ms per image (laptop GPU) vs 0.38 s for classical Sea-thru at the same size (~120× faster).
*"D: we then tried to adapt the network to real dives. Most of it failed, and the failures are instructive."*

---

# PART 2: After BetaNet: fine-tuning on real dives (all attempts, mostly failed)

**13. Fine-tuning on real dives: what we tried and what failed** (D, 2 min). Figures: `fig15_ft_real.png` (one row), `fig18_ft_sites.png`
- Setup: 552 real dive frames with depth, no clean references → label-free physics checks (darkest pixels = haze;
  colour must not drift with distance; gray-world), with a synthetic batch in every step to keep the units.
- **Attempt 1, all losses (v1):** chart error 12 / 11° → 10 / 9°, but a "same dive → same water" loss **collapsed**
  every dive to one average water (dive spread 0.24 → 0.05). Images still looked fine; only the per-dive check caught it.
- **Attempt 2, no synthetic anchor:** real-image losses just as good, but synthetic β_D error 19 % → 70 %: coefficients lost their units.
- **Attempt 3, add UIEB:** UIEB got *worse* (18.0 → 16.0 dB): monocular depth has the wrong scale, so range losses are fed wrong distances.
- **What worked (partly):** drop the dive loss (`ft_nodive`): chart error with the network's own E halves (16–17° → ~10°)
  without hurting synthetic accuracy (PSNR 20.0 → 20.4). Caveat: only 2 chart frames; outputs are flatter in contrast.
- Backup slide: `fig20_ft_tradeoff.png`. The teacher weight trades colour fixes against keeping the water
  read-out; 0.3 gives the best UIEB score of any model (18.3 dB, ahead of gray-world).

**13b. Optional: training longer** (D, 45 s, or keep as backup). Figure: `3_new_runs/old_vs_new_verdict.png`
(backups: `fig21_loss_curves.png`, `fig23_big_results.png`, `seathru_original_vs_longer.png`)
- Same network, 2.6× more NYU rooms, 60 epochs: synthetic 20.0 → 21.2 dB (95 % CI +1.0 to +1.4), β_D / β_B error
  20 / 23 % → 17 / 18 %. It has converged (validation flat over the last 10 epochs).
- But real dives are told apart *less* well (separation 1.57 → 1.29; turbid D4 no longer stands out), and chart
  colour is unchanged. Better synthetic scores ≠ better real-water read-out: the synthetic-to-real gap again.
- Fine-tuning it fixes chart colour (16.9° → 8.6° with its own sunlight colour) but lowers separation further, so
  the losses need rebalancing first.

**14. Plan of action** (D, 45 s). Figure: `fig13_plan.png` (step "self-supervised fine-tuning" is now done)
- More realistic synthetic water → range-dependent β_D(z) → hybrid physics + refiner →
  full chart evaluation (all Sea-thru + SQUID) → uncertainty; stretch: detection mAP.
- Immediate: adopt the longer-trained model as the base; rebalance the fine-tuning losses (less gray-world,
  small teacher weight); add a distance-consistency metric for real photos.

**15. Takeaways** (D, 30 s)
- A small network can read the water's coefficients from one photo + depth, in milliseconds.
- Physics in the loop gives interpretable, checkable outputs; failures are diagnosable (v1→v2, collapse).
- Label-free fine-tuning on real dives helps colour but is fragile; the open gaps are metric depth scale and a richer chart test.
