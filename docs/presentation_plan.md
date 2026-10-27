# Intermediate review: slide plan (15 min, ~14 slides)

Rubric points and where each is covered:
| Rubric item | Slide(s) |
|---|---|
| Intro slide: topic + all names & roll numbers | 1 |
| Importance; impact if not solved | 2 |
| Method explained so a non-expert can follow | 4, 6, 7 |
| Results with graphs | 8, 9, 11, 12 |
| Discussion of unique contributions | 9, 12, 14 |
| Plan of action + milestones | 13 |
| Uniqueness / failed experiments that guided design | 10 (v1→v2), 6 (depth scale) |
| Seamless hand-over | speaker split below |
| Slides not text-heavy | ≤ 3 short lines per slide, figure-first |
| Fielding questions | `docs/qa_prep.md` |

Suggested split for 4 speakers: **A** slides 1–3, **B** 4–7, **C** 8–11, **D** 12–14 (≈ 3.5 min each).
Hand-over line at the end of each block is given in *italics*.

---

**1. Title** (A, 20 s)
"Learning the Water: a neural network that estimates attenuation and backscatter coefficients"
Group 4 · names + roll numbers · OE5510 · 29 Oct 2026. Background: one raw/restored Sea-thru pair.

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
| **Ours** | photo + depth | **~2 ms / image** | **trained once on many waters** |

**6. Data and depth** (B, 1 min). Figures: `fig2_jerlov_types.png` (left), `fig5_depth_eval.png` (right)
- Train: NYU-v2 rooms + synthetic water from the 10 Jerlov ocean types → true coefficients known.
- Test on real: Sea-thru dive photos (true depth, colour charts) and UIEB.
- Finding: Depth-Anything-V2 gets the *shape* right but the *scale* wrong (AbsRel 0.76 → 0.12 with one range reading).

**7. Method** (B, 40 s). Figure: `fig4_method_overview.png`
- ResNet-18 → 12 numbers → physics layer → image. Both outputs the professor asked for.
- Loss = coefficient error + image error. ~10 min training on a laptop GPU.
*"C will show how well it works."*

**8. Does it find the right coefficients?** (C, 1 min). Figure: `fig6_coef_scatter.png`
- 654 rooms never seen in training, random water. [quote median % errors from results/figure_numbers.json]

**9. Is the image right? vs baselines and ablations** (C, 1 min). Figures: `fig7_synth_results.png`, `fig8_jerlov_breakdown.png`
- Beats raw, gray-world and classical Sea-thru; compare with the black-box UNet and the oracle.
- Ablations: no depth input, coefficient-loss only, image-loss only.

**10. A failed experiment that taught us the physics** (C, 1 min). Figure: columns 4 vs 5 of `fig10_real_seathru.png`
- v1 on real dives → magenta images. Why: 10 m deep, red is lost *on the way down*, not between camera and reef.
- Fix: add the sunlight colour E to the data and the network (v2) → magenta gone, UIEB PSNR 15.9 → 18.0 dB.

**11. Real underwater photos** (C, 1 min). Figure: `fig10_real_seathru.png`
- Grey patches on the chart: raw 29° → ours 11–12° (classical Sea-thru port 16°, DeepSeeColor paper 14°, gray-world 8°).
- Gray-world balances colour but leaves the haze; ours removes haze that grows with distance.
*"D: what else the coefficients give us, and what's next."*

**12. The camera as a water sensor + speed** (D, 1 min). Figure: `fig11_site_consistency.png`
- Same dive → similar β; the turbid site stands apart. No labels used.
- ~2 ms per image (laptop GPU) vs ~0.4–1.2 s for classical Sea-thru on small frames.

**13. Plan of action** (D, 1 min). Figure: `fig13_plan.png`
- More realistic synthetic water → range-dependent β_D(z) → self-supervised fine-tuning on real photos →
  full chart evaluation (all Sea-thru + SQUID) → uncertainty; stretch: detection mAP.

**14. Takeaways** (D, 30 s)
- A small network can read the water's coefficients from one photo + depth, in milliseconds.
- Physics in the loop gives interpretable, checkable outputs; failures are diagnosable (v1→v2).
- Biggest open gap: synthetic→real, and metric depth scale.
