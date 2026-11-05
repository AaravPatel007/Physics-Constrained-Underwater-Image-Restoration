# Agent logs (read this first when picking up the work)

Newest session on top. Each entry: what was done, what is running/unfinished, decisions and why, how to resume.
Project background: `SUMMARY.md`, model names: `docs/MODELS.md`, datasets: `docs/external_datasets.md`.

---

## Session 6 (4 Nov 2026, RTX 3060 laptop, performance mode, no thermal guard by the user's choice)

**Goal (from the user):** follow the session 5 hand-off steps, use teammate's Day 1b for the final metrics, take care of the
datasets that need no human, and do not commit anything back. **Everything below is local and uncommitted.**

**Verdict: use `refine_mix_d1b`** (BetaNet + Day 1b + RefineNet, `--mix uieb:0.4,lsui:0.4,euvp:0.2 --day1b --init refine_mix_full`).
PSNR / ΔE00, duplicates removed (`results/stress_test_s6b/clean_rescore.json`; synthetic from `summary.json`):

| Test set | BetaNet | + Day 1b | + `refine_d1b` | + `refine_mix_d1b` | `refine_mix_d1b` − Day 1b, PSNR (95% CI) |
|---|---|---|---|---|---|
| UIEB val (89) | 17.67 / 13.71 | 19.21 / 12.19 | 20.89 / 10.78 | 20.77 / 10.85 | +1.56 (1.05, 2.11) |
| LSUI test (355) | 18.12 / 13.93 | 20.23 / 12.43 | 21.07 / 11.56 | 22.39 / 10.35 | +2.16 (1.90, 2.43) |
| EUVP test (492) | 15.96 / 20.36 | 18.07 / 17.27 | 17.90 / 16.48 | 20.05 / 14.63 | +1.97 (1.84, 2.12) |
| Synthetic (654) | 21.21 / 13.48 | 20.32 / 13.76 | 23.51 / 10.08 | 23.63 / 9.90 | not computed |

- `refine_d1b` vs `refine_mix_d1b` on UIEB val is noise (−0.12 dB, CI −0.37 to +0.13). `refine_mix_d1b` wins on LSUI and EUVP, but it
  trained on their train splits, so part of that lead is familiarity. `refine_d1b` never saw them and still beats Day 1b on LSUI.
- Plain-BetaNet models, UIEB val: `refine_ft` 19.67 / 12.02, `refine_mix_full` 20.06 / 11.76, `refine_lsui` 19.38 / 12.75,
  `refine_lsui_uieb` 20.07 / 11.72. Day 1b input is worth more than any data recipe alone.
- Grey charts: 6.38° (`refine_mix_d1b`), 7.44° (`refine_d1b`), 12.97° (Day 1b). Two frames only, so indicative.
- Red specks: `refine_mix_d1b` is below Day 1b on every no-reference set, C60 included.
- **Still behind main's `day3a`:** 21.68 / 9.66 on UIEB val, ahead of `refine_mix_d1b` by 0.91 dB and 1.19 ΔE00.
- **Colour cast:** Day 1b alone is lower on UCCS green (4.39 vs 7.10), tank (8.68 vs 10.78), TUB (3.95 vs 7.76), C60 (9.59 vs 9.78).
- **Turbidity is not fixed.** TUB drift: Day 1b 12.28, `refine_mix_d1b` 13.30, `refine_d1b` 15.01 (ignore TUB PSNR/ΔE, see session 5).
- **C60 shows a failure the averages hide** (`figures/stress_test_c60/gallery_c60.png`): on a deep-blue photo, Day 1b paints a red-brown
  wash and `refine_mix_d1b` keeps a saturated red blotch on the dark rock, worse than `refine_d1b`. On a green photo the Day 1b family
  turns the scene purple. C60 cast and UCIQE barely move, so look at the images, not only the numbers.

**Code (uncommitted):**
- `finetune_refine.py`: `--day1b`, saved in `best.pt` args; `--test` reads it. In logs, the "BetaNet" column means Day 1b input for those runs.
- `stress_test.py`: Day 1b refiners fed the Day 1b output, CIs against Day 1b, `--out`, new `c60` set.
- `stress_clean.py`: `--out`, reads refiners from that folder's `summary.json`.
- `prep_extra_pairs.py`: `N_KEEP` raised to the full sets (lsui 4279, euvp 5550). **This changes the LSUI/EUVP splits, so `refine_mix` (old
  subset) must not be scored on them: its training photos may be in the new test set.**
- `results/logs/dashboard.py`: live status page, `python3 results/logs/dashboard.py`, then http://127.0.0.1:8765. The teammate table and
  dataset usage labels in it are hand-kept.

**Results:** `results/stress_test_s6b` is the one to read (six models). `stress_test_s6` is the earlier five-model run, `stress_test_c60`
holds C60. `--only` rewrites `summary.json`, so always give it its own `--out`. Duplicates removed from test sets: uieb160 3, uieb90 1,
LSUI 45, EUVP 8. Run scripts are `results/logs/s6_*.sh`.

**Data:** in `data/external/`: TUB, Sea-thru full (36 GB), C60 (password-protected, extracted by the user; do not commit or redistribute).
- SQUID and R-CAUSTIC are paused by the user (see Decisions below). Zenodo drops connections. `s6_resume.sh <zip>` resumes one SQUID
  zip until its md5 matches. Satil and Nachsolim have resume loops. Michmoret and Katzaa only have the main loop, so check them with `files.tsv`.
- HICRD: the user is fetching `metadata.zip` and `HICRD_paired.zip` into `data/external/hicrd/`. Unpaired is skipped for now.
- Training reads only the `.npz` files. Plan agreed in principle: shrink Sea-thru full and FLSea to training size, verify, then delete
  originals; stream FLSea shard by shard. The shrink script does not exist yet and no originals have been deleted.

**Thermals:** CPU ran ~100 °C until the laptop moved to an AC room, then ~88 °C and Day 1b epochs fell from 287 s to ~196 s. `OMP_NUM_THREADS=2` kept.

**Checked GitHub after Day 1b (user ran `git fetch`; nothing merged):** `safe_restore.py` is unchanged on `main`, `sunlight_phy_model` and
`sunlight_depth_v2`, so nobody has changed Day 1b. `main` adds the `day3` touch-ups (`src/day3_touchup.py`, about 6 minutes of training, weights
not in git). Kairav's sunlight branches (BetaNet predicting depth below the surface, then E0 + K_d + d_s) do not beat BetaNet: UIEB ΔE with
Day 1b 12.67 baseline, 13.06 v1, 12.95 v2. Neither targets the deep-blue, green or turbidity failures. `day3a` has not been run on C60.

**Audit of Aarav's touch-up (user's request; review is on 5 Nov 2026). His code is untracked here, nothing merged.**
- Weights: the user supplied his `day3a` `best.pt` (safe load, global, epoch 50, md5 237f9463…), now in
  `results/review2_day3_touchup/day3a_touchup_global/`. My stopped partial run is `best_partial_session6.pt` there. `day3b` was retrained here
  with his unmodified script. His code copies: `src/day3_touchup.py`; `src/day3_seed.py` differs only in seed and output paths (not run).
- **`day3a` reproduces, on his scorer and on this laptop:** UIEB val 21.67 / 9.66 (his 21.68 / 9.66), held-out ΔE 9.365 (same), charts 5.10° / 4.80°
  (same). Day 1b 19.11 / 12.22 (same). So the number is real on that set. This says nothing yet about other datasets.
- **`day3b` is not reliably better:** retrained with the same script and seed it gives 21.68 / 9.70 (his 21.86 / 9.53), held-out 9.547 (his 9.512),
  68 of 90 photos better than Day 1b (his 74). Same-recipe reruns move it by about 0.2 dB, as much as the gap to `day3a`.
  Its charts here: 4.09° / 2.92°. On his own run, `day3b` − `day3a` on UIEB val was +0.19 dB (CI −0.14, +0.53).
- Code review: selection rule fixed in advance, 90 val photos not used for training or checkpoints, LSUI duplicates checked. Pure L1 to UIEB references,
  no synthetic or anchor term, unlike RefineNet. His no-physics control reaches 22.63 dB, higher than the touch-ups, so PSNR on UIEB partly rewards
  UIEB's editing style; the control does not improve the charts (10.6° / 10.0°).
- Fairness in our stress test: **`uieb160` is skipped for touch-ups because his 680 training photos contain those photos** (both splits use
  `permutation` seed 0). With touch-ups in a run, `stress_clean.py` counts all 800 UIEB train photos as seen.
- **Stress verdict (`results/stress_test_day3`, one scorer, duplicates removed; `day3a` = his weights, `day3b` = retrained here). `day3a` holds on UIEB, not beyond it.** PSNR / ΔE00:

  | Test set | Day 1b | `day3a` | `day3b` | `refine_d1b` | `refine_mix_d1b` |
  |---|---|---|---|---|---|
  | UIEB val (89) | 19.21 / 12.19 | **21.57 / 9.88** | 21.53 / 9.92 | 20.89 / 10.78 | 20.77 / 10.85 |
  | LSUI test (355) | 20.23 / 12.43 | 20.40 / 11.64 | 20.34 / 11.34 | 21.07 / 11.56 | **22.39 / 10.35** (trained on LSUI) |
  | EUVP test (492) | 18.07 / 17.27 | 17.42 / 17.58 | 17.56 / 17.23 | 17.90 / 16.48 | **20.05 / 14.63** (trained on EUVP) |
  | Synthetic (654) | 20.32 / 13.76 | 20.24 / 13.56 | 20.05 / 13.56 | 23.51 / 10.08 | **23.63 / 9.90** |

  - `day3a` beats `refine_d1b` on UIEB val (+0.68 dB, CI 0.20 to 1.20; −0.89 ΔE, CI −1.42 to −0.39) and `refine_mix_d1b` (+0.80 dB, −0.96 ΔE). It gets 21.57 / 9.88 here vs 21.67 / 9.66 on his scorer.
  - On unseen data `day3a` loses its gain: LSUI vs Day 1b +0.17 dB (CI −0.26 to 0.59), −0.79 ΔE; **EUVP vs Day 1b −0.65 dB (CI −0.80 to −0.52), +0.31 ΔE, so worse than no touch-up**. Fair comparison is `refine_d1b` (same UIEB-only real data): it beats `day3a` on LSUI PSNR (+0.67) with ΔE level, and on EUVP (+0.48 dB, −1.10 ΔE).
  - LSUI and EUVP references are also someone's enhancement, not ground truth, so part of this is style mismatch.
  - Grey charts, same scorer, 2 frames: `day3b` 5.68°, `refine_mix_d1b` 6.38°, `day3a` 7.32°, `refine_d1b` 7.44°, Day 1b 12.97°. His 5.1° / 4.8° come from his scorer; do not mix them. All touch-ups and RefineNets are close, and all beat Day 1b.
  - TUB drift: Day 1b 12.28, `refine_mix_d1b` 13.30, `day3a` 13.47, `refine_d1b` 15.01, `day3b` 15.05. None beats Day 1b. Red specks: `day3a` is close to Day 1b (worse on U45, 7.09 vs 6.32). Both RefineNets are lower than `day3a` on six of the seven no-reference sets; tank is level.
  - **C60: `day3a` does not fix the failures** (`figures/stress_test_day3/gallery_c60.png`). The green photo goes blue-purple instead of purple, and the red-brown wash on the deep-blue photo stays. Neither touch-up nor RefineNet removes them.
- Runs: `results/stress_test_day3` (all sets, `refine_d1b`, `refine_mix_d1b`, `day3a`, `day3b`), `results/stress_test_hicrd`. Their logs share `results/logs/s6_stress_c60.log`
  and `s6_stress_hicrd.log`. **Results pending when this was written: read those folders, do not assume.**

**HICRD (in `data/external/hicrd/`, 1,700 train + 300 test pairs, 1842×980):** references were restored using site water data; the photos are the
"good-quality" subset. Train pairs by site: 1: 25, 2: 16, 4: 159, 5: 945, 7: 457, 8: 98. Test files have no site names. No depth maps.
- `src/hicrd_water.py`: K_d from the published exponential fits, averaged per camera channel with the Ximea response, 400–700 nm (ignores the light's own
  spectrum). K_d R/G/B per metre: site 1 0.385/0.136/0.109, 2 0.237/0.039/0.031, 4 0.398/0.145/0.118, 5 0.366/0.133/0.109, 7 0.352/0.091/0.071, 8 0.283/0.082/0.060.
  Site 3 flagged unreliable, site 6 no data, site 7 has a profile but no transect data. β_D and K_d are different quantities, so compare ranking, not equality.
- **Water check result: BetaNet's β_D does not follow the measured water** (`results/hicrd/water.json`, `figures/hicrd/beta_d_vs_kd.png`). Across the 6 sites, rank correlation
  with K_d is −0.31 R, −0.26 G, −0.26 B (only 6 sites, so weak evidence either way, but no sign of tracking). β_D is nearly constant: site means R 0.36–0.51, G 0.14–0.18, B 0.17–0.21, while
  K_d varies far more in G (0.04–0.15) and B (0.03–0.12). It also puts blue above green at all 6 sites; measured K_d has green above blue at all 6. The all-points rank correlation of 0.54 only reflects red being larger.
  Depth is unmeasured: halving DAv2 depth raises β_D ~26–33%, doubling lowers it ~31–36%, so absolute size is not testable here (ratio β_D/K_d at x1: R 1.36, G 1.86, B 2.88).
  Limits: β_D is not K_d, sites 1 and 2 have only 25 and 16 photos, ranking under other depth scales was not computed. Fair claim: unconfirmed on real measured water, not "proven wrong".
- **Pair scores, 300 test pairs, none removed as duplicates (`results/stress_test_hicrd`, `figures/stress_test_hicrd/gallery_hicrd.png`). No model trained on HICRD. Every learned correction is worse than plain Day 1b.** PSNR / ΔE00:
  gray-world 23.81 / 9.02, Day 1b 21.51 / 11.55, `day3b` 20.48 / 12.11, `day3a` 19.83 / 12.39, BetaNet 18.88 / 14.54, raw 18.47 / 17.21, `refine_d1b` 17.81 / 18.79, `refine_mix_d1b` 17.50 / 17.67.
  - vs Day 1b: `day3a` −1.68 dB (CI −2.19, −1.16), +0.84 ΔE; `refine_d1b` −3.69 dB (−4.05, −3.34), +7.24 ΔE; `refine_mix_d1b` −4.01 dB (−4.40, −3.61), +6.12 ΔE. `day3a` beats `refine_d1b` by 2.02 dB and 6.40 ΔE.
  - **Not a bug: RefineNet overshoots.** On every 6th test photo (50), mean red is 0.47 for both RefineNets vs 0.31 in the reference, and brightness 0.41–0.43 vs 0.33; the images turn visibly pink. Matching brightness alone lifts them to 23.15 / 24.49 dB, but the red shift stays. `day3a` is 0.37 bright. Day 1b 0.28.
  - Gray-world winning says HICRD references are close to neutral and dark. They are restorations, not ground truth, so this is one more style, though built from measured water.
  - **This overturns "`refine_mix_d1b` is the most general".** Do not use that line. It generalises to styles it trained on (LSUI, EUVP), and fails on a new one.
- `hicrd` is a new paired set in `stress_test.py` and `stress_clean.py`.

**Random web photo check (`results/online_test/compare.html`, untracked):** 20 Wikimedia Commons photos, seeded random order, dropped only non-underwater items and one near-copy; raw, Day 1b, `refine_mix_d1b`, `day3a`. No references, so visual only. First look at 15 of them, thumbnail size: neither model makes photos look taken above water; blue water stays blue. Green shipwreck photo: both remove the green, Day 1b does not. Shallow sun-caustic photo: RefineNet adds dark blotches. Not a blind test. Photos are CC licensed: keep the credits if used on slides.

**BridgeNet added to the web photo page (user's hunch):** `~/teamwork_projects/underwater_bridge_nn/checkpoints/best_model.pth` run read-only (nothing written there) through its own DAv2 + Sea-thru + NAFNet, ~2 s per photo. Same 20 photos, fifth column. First look at 14 of them: BridgeNet is the only system that really removes the water colour (#1, #4, #6, #10, #12), but it adds artefacts: blotchy, grid-like or lilac water (#1, #4, #12), a broken black-and-white result on the green wreck (#7), and an orange/brown cast (#3, #9). Unscored: it has never been run on UIEB, HICRD or our other sets. Not a blind test.

**BridgeNet diagnosis (7 web photos, by eye):** its Sea-thru stage is unstable (strong pink/red on #1, #4, purple on #3), and the NAFNet stage rewrites that heavily; the blotchy water and the broken #7 appear only after NAFNet (#7's Sea-thru output is still a normal green scene). Checkpoint: 6.8 M params, epoch 45. Hypothesis, untested: it trained on indoor NYU rooms, which have no open water, so it invents pale wall-like backgrounds. A no-training fix (BridgeNet low-frequency colour on Day 1b detail) did not work: better on #10 and #12, near Day 1b on #1, worse on #7.

**Hybrid prototype (`results/online_test/hybrid_prototype.py`, sixth column of `compare.html`), no training:** per-pixel weight for BridgeNet from local luminance correlation with the raw photo, texture and a highlight check; `day3a` elsewhere. Thresholds fixed before viewing and not tuned. By eye on 20 photos: removes most of BridgeNet's blotchy open-water artefacts (#1, #2, #10, #12), keeps its sand and rock. Does not stop its colour casts, because they keep the structure (#3 orange, #8 pink, #9 brown). Adds seams and halos on some (#4 around the fish, #17, #7, #18). Unscored, not blind, one photo set. Mean trust does not detect casts (#3 is 1.00), so it cannot act as a whole-photo selector.

**Overnight queue (user asleep, authorised all feasible steps; `results/logs/s6_night.sh`, status in `s6_night.log`):** (1) `stress_test_bridge`: BridgeNet + hybrid next to `day3a` and RefineNet. (2) `refine_all_d1b` on UIEB+LSUI+EUVP+HICRD; leave-one-dataset-out `refine_no_{hicrd,lsui,euvp,uieb}` and control `refine_only_uieb`, all from `refine_v1`, `--last` (no checkpoint selection, so held-out data is never touched). (3) `refine_d1b_free` rematch: no anchor, no synthetic term. (4) `stress_test_final`. (5) `day3b` seeds 1, 2. `stress_clean.py` now counts HICRD train as seen; 0 of 300 HICRD test photos have a near-copy in HICRD train (median best match 0.50).
Blind test ready: `python3 results/blind_test/server.py`, then http://127.0.0.1:8766. 40 photos fixed by seed (20 web, 10 C60, 5 HICRD, 5 U45), 5 systems, names hidden, each pair equally often; votes in `votes.jsonl`, tally at `/results`. Its RefineNet is now `refine_all_d1b` (swapped on 5 Nov before any vote, `swap_refine.py`; old key in `key_refine_mix.json`; all 40 Day 1b images reproduced exactly, so the inputs match). No real votes yet.

### Overnight results (5 Nov, queue finished 07:18, no failed steps)

All scores PSNR / ΔE00, duplicates removed, from `results/stress_test_final/` and `results/stress_test_bridge/`. References are edited or restored images, not ground truth. Synthetic has true ground truth and no duplicate step. Figures in `figures/stress_test_final/` and `figures/stress_test_bridge/`.

| System | Trained on | UIEB val (89) | LSUI (355) | EUVP (492) | HICRD (300) | Synthetic (654) |
|---|---|---|---|---|---|---|
| Gray-world | Nothing | 17.68 / 13.83 | 19.74 / 13.15 | 19.26 / 16.39 | 23.81 / 9.02 | 15.40 / 17.77 |
| Day 1b | Nothing (recipe) | 19.21 / 12.19 | 20.23 / 12.43 | 18.07 / 17.27 | 21.51 / 11.55 | 20.32 / 13.76 |
| `day3a` | UIEB | 21.57 / 9.88 | 20.40 / 11.64 | 17.42 / 17.58 | 19.83 / 12.39 | 20.24 / 13.56 |
| `refine_all_d1b` | All 4 + synthetic | 20.77 / 10.80 | 22.46 / 10.14 | 20.22 / 14.42 | 27.54 / 6.60 | 23.82 / 9.84 |
| `refine_mix_d1b` | No HICRD | 20.77 / 10.85 | 22.39 / 10.35 | 20.05 / 14.63 | 17.50 / 17.67 | 23.63 / 9.90 |
| `refine_only_uieb` | UIEB + synthetic | 20.52 / 11.24 | 20.94 / 11.68 | 18.04 / 16.63 | 17.96 / 18.25 | 23.34 / 10.07 |
| `refine_d1b_free` | UIEB, no anchor | 21.87 / 10.18 | 20.55 / 11.89 | 17.79 / 17.18 | 16.52 / 17.05 | 18.73 / 16.02 |
| BridgeNet | NYU synthetic | 15.77 / 18.42 | 17.49 / 15.40 | 18.03 / 15.60 | 22.09 / 11.42 | not run |
| Hybrid | No training | 17.91 / 14.38 | 19.13 / 12.95 | 18.45 / 15.22 | 22.14 / 11.30 | not run |

Other tests. Charts (grey-patch error, two photos, lower is better): BridgeNet 3.10°, hybrid 3.34°, `refine_mix_d1b` 6.38°, `day3a` 7.32°, `refine_d1b_free` 8.63°, `refine_all_d1b` 8.83°, Day 1b 12.97°. TUB drift (lower is better): BridgeNet 11.46, Day 1b 12.28, `refine_all_d1b` 12.88, `refine_mix_d1b` 13.30, `day3a` 13.47, `refine_d1b_free` 15.12, hybrid 17.51.

**1. Does training on more datasets generalise? Only a little.** Each row scores a set that the model never saw, against the UIEB-only control, which also never saw it. Paired 95% CI in brackets.

| Unseen set | Left-out model | Control | PSNR gain | ΔE00 change | Gain once the set is in training (`refine_all_d1b` vs control, PSNR) |
|---|---|---|---|---|---|
| LSUI | 21.16 / 11.29 | 20.94 / 11.68 | +0.22 [+0.05, +0.39] | −0.39 [−0.55, −0.22] | +1.52 |
| EUVP | 18.37 / 16.81 | 18.04 / 16.63 | +0.33 [+0.27, +0.41] | +0.18 [+0.08, +0.27] (worse) | +2.18 |
| HICRD | 17.83 / 16.72 | 17.96 / 18.25 | −0.12 [−0.19, −0.04] | −1.53 [−1.67, −1.41] | +9.58 |

Most of the gain comes from having that dataset in training, not from general skill. Without HICRD, RefineNet stays 3.68 dB and 5.17 ΔE00 behind fixed Day 1b on HICRD. With UIEB left out, `refine_no_uieb` scores 19.68 / 12.26 on UIEB val, +0.46 dB on Day 1b [+0.05, +0.88] with level colour. **Do not claim that adding niche datasets gets us to "any underwater photo".** This is one training run per setting, so run-to-run noise is not in these CIs.

**2. `refine_all_d1b` is the best RefineNet, not a general winner.** It fixes the HICRD pink and costs nothing on UIEB (+0.00 dB against `refine_mix_d1b`). It beats `day3a` on LSUI, EUVP and HICRD, but it trained on those sets and `day3a` did not, so that is not a fair win. `day3a` still leads on UIEB val by 0.80 dB [0.30, 1.29] and 0.91 ΔE00 [0.39, 1.48]. Charts got worse (6.38° to 8.83°, two photos). HICRD test names are anonymised, so test photos may come from training sites; 27.54 is a within-dataset score. On the 20 web photos it looks almost the same as `refine_mix_d1b` by eye, and blue water stays.

**3. Fair comparison with `day3a` on data neither saw.** LSUI: `refine_no_lsui` leads by 0.75 dB [0.38, 1.15], colour level. EUVP: `refine_no_euvp` leads by 0.95 dB [0.80, 1.11] and 0.77 ΔE00 [0.60, 0.96]. HICRD: `day3a` leads `refine_no_hicrd` by 2.00 dB [1.76, 2.24] and 4.33 ΔE00 [3.91, 4.76]. Split result, no overall winner. RefineNet had more training sets in these rows.

**4. Rematch: on UIEB, RefineNet ties `day3a` once it has the same freedom, and pays for it elsewhere.** `refine_d1b_free` against `day3a` on UIEB val: PSNR +0.30 [−0.16, +0.72], ΔE00 +0.29 [−0.19, +0.79], a statistical tie. So `day3a`'s UIEB lead over RefineNet came from the training setup (anchor and synthetic terms), not from a better network. The free model then loses to `day3a` on HICRD (−3.31 dB), synthetic (18.73 vs 20.24), TUB drift and charts. `day3a`'s roughly 60 outputs cannot overfit in the same way. That is the best argument for it as the main method.

**5. `day3b` seeds.** Four `day3b` runs on Aarav's scorer: 21.86 / 9.53 (Aarav), 21.68 / 9.70 (retrain), 22.09 / 9.22 (seed 1), 21.89 / 9.54 (seed 2). Mean 21.88 / 9.50, spread 0.41 dB. `day3a` is 21.67 / 9.66 from one run, so its own spread is unknown. Three of four `day3b` runs edge it on UIEB, inside what seeds alone can move. On charts both new seeds are worse than `day3a` (5.30° and 6.28° against 4.95°, SfM range, two photos). Keeping `day3a` as chosen stays reasonable.

**6. BridgeNet and hybrid.** Far behind on UIEB and LSUI, where references reward editing style. Best of the learned systems without HICRD training on HICRD, and best on charts (two photos). BridgeNet has the lowest TUB drift of all processed systems; the hybrid has the worst. The hybrid is not a clear upgrade. Scores do not capture why it looks good on some photos: the blind test has to answer that.

**Slide framing:** `day3a` is the main method. Lead with its UIEB result and its robustness against the free RefineNet. Say plainly that it does not beat Day 1b on EUVP or HICRD, and that gray-world beats every system not trained on HICRD. RefineNet is "what we tried": strong in its training domains, and proof that the gain mostly follows the data. Avoid "looks like it was taken above water" and "works on any photo". Still unsolved: blue water, deep-blue and green failures on C60, turbidity, β_D not tracking measured water.

**Decisions on 4 Nov evening (user):** downloads stay paused (SQUID, R-CAUSTIC); FLSea, Sea-thru shrink cut for now; no deletions; do HICRD first. Not fixing Day 1b's tints
or turbidity before the review: present them as limitations. `results/logs/s6_disk_guard.sh` pauses downloads below 8 GB free.

**Next:** blind test with 4–5 people, tally at `/results` only after all vote, then slides from "Overnight results" above. Review about 13:00 on 5 Nov. After the review: a fix for the deep-blue and green failures, turbidity, and a decision on committing session 6.

---

## Session 5 (4–5 Nov 2026, MacBook M2, overnight, unattended)

**Goal (from the user):** retrain RefineNet with the *small* new datasets only (bigger runs later on the
other laptop with ~120 GB free), stress-test the pipeline (PSNR, SSIM, colour error ΔE00, …) and compare with
the earlier models, keep this log, update graphify at the end, one commit at the end that includes the new
RefineNet weights, pushed to branch `RefineNet`. No unnecessary commits.

### Findings (read this first)
All on the original BetaNet `baseline_v1_60ep_bigdata`. Real references are preferred edits, not ground truth. Numbers:
`results/stress_test/summary.json`, duplicate-free: `clean_rescore.json` (`src/stress_clean.py`), figures: `figures/9_stress_test/`.

**Verdict:** keep `refine_ft` as the RefineNet to use. `refine_mix` (the new weights) is not a replacement: it wins only
on LSUI and EUVP, the styles it trained on, and it is slightly worse on UIEB. Main's touch-up is still clearly ahead.

Paired sets, PSNR dB / ΔE00, test photos with a copy in training removed (except synthetic):

| Set (photos kept) | BetaNet | + Day 1b | `refine_ft` | `refine_mix` | mix − ft, PSNR [95 % CI] | mix − ft, ΔE [95 % CI] |
|---|---|---|---|---|---|---|
| Synthetic NYU test (654) | 21.22 / 13.48 | 20.32 / 13.76 | 23.42 / 10.55 | 23.40 / 10.52 | −0.02 [−0.09, +0.05] | −0.03 [−0.10, +0.04] |
| UIEB held out (158/160) | 17.70 / 13.22 | 19.02 / 11.81 | **20.21 / 11.50** | 19.97 / 11.73 | −0.24 [−0.41, −0.06] | +0.23 [+0.06, +0.40] |
| UIEB val (89/90) | 17.66 / 13.72 | 19.21 / 12.19 | 19.67 / 12.02 | 19.64 / 12.16 | −0.03 [−0.24, +0.18] | +0.14 [−0.09, +0.38] |
| LSUI held out (379/400) | 18.35 / 13.97 | 20.10 / 12.61 | 20.37 / 12.27 | **21.30 / 11.43** | +0.93 [+0.77, +1.09] | −0.84 [−1.01, −0.68] |
| EUVP held out (497/500) | 15.98 / 20.32 | 18.07 / 17.22 | 16.92 / 19.00 | **19.17 / 16.02** | +2.25 [+2.09, +2.41] | −2.98 [−3.26, −2.72] |

Sets no model trained on:

| Test | BetaNet | + Day 1b | `refine_ft` | `refine_mix` |
|---|---|---|---|---|
| TUB drift ΔE, 320 photos, lower = steadier (raw 10.2) | 15.53 | **12.28** | 14.10 | 14.26 |
| Grey charts, 2 Sea-thru frames, degrees (gray-world 8.46) | 25.33 | 12.97 | 14.44 | **11.10** |
| Colour cast, U45 (45) | 20.81 | **13.92** | 17.19 | 17.72 |
| Colour cast, UIQS (200) | 18.18 | **5.96** | 10.43 | 10.80 |
| Colour cast, TankImage-I (78) | 18.14 | **8.65** | 16.73 | 15.91 |
| Colour cast, UCCS green (100) | 23.21 | **4.40** | 14.04 | 13.55 |

What this means:
- **Fine-tuning on real photos is what helps.** `refine_v1` (synthetic only) is worst on real pairs (UIEB 16.54 / 17.10)
  and best on synthetic (23.84 / 10.08); `refine_ft` gives up 0.4 dB on synthetic for +3.7 dB on UIEB.
- **More datasets did not make RefineNet more robust.** On unseen sets `refine_mix` and `refine_ft` are about level.
  The one gain is the charts (11.10° vs 14.44°), but that is 2 frames from one dive: indicative only. On the UIEB fins
  photo `refine_ft` turns the fins yellow, `refine_mix` leaves them yellow-green.
- **Caveat on that test:** `refine_mix` saw only ~2 passes of the LSUI subset and <1 of EUVP, and UIEB fell from 12
  passes to ~5, at equal compute. This does not rule out a longer run on all the data.
- **RefineNet's lead over Day 1b is small.** UIEB val: +0.46 dB, −0.17 ΔE. Day 1b has less colour cast than RefineNet on
  every unseen set except UCCS blue (11.81 vs 11.18 / 10.27), and is the steadiest of the restorations as turbidity
  rises. Low cast alone is not quality (gray-world is ~1).
- **Main's touch-up is ahead.** `day3a_touchup_global` on UIEB val: 21.68 / 9.66 (their report) vs `refine_ft` 19.67 /
  11.99 (all 90 photos). Fair enough to read: same BetaNet and photos, and this harness gives Day 1b 19.22 / 12.16 vs their
  19.11 / 12.22. Not run here (no weights). Their chart degrees (Day 1b 10.3° / 10.6°) do not match this harness
  (12.59° / 13.34°), so do not compare chart numbers across the two.
- **Turbidity is a BetaNet weak spot.** Its fog reading β_B rises to ~0.6 by 11–14 NTU, then stays flat up to 35 NTU
  (Spearman with NTU 0.46). BetaNet, Day 1b and RefineNet all drift more than raw does. RefineNet does not fix this.
- TUB PSNR/ΔE against the stage-1 raw photo favour raw by construction; ignore them, use drift.

Best next experiments: RefineNet on Day 1b output instead of raw BetaNet (not tried; Day 1b and RefineNet fix different
things), LSUI first then UIEB on all the data, and running `day3a` here if its weights turn up.

### Machine and setup facts (Mac)
- Apple M2, 16 GB, no CUDA. PyTorch uses the M2 GPU via MPS. Venv: `.venv/` (git-ignored), built by
  `src/mac_setup.sh` (uv, Python 3.12, torch 2.15). Run scripts from `src/` with `../.venv/bin/python`.
- `src/device.py` (new): `DEV` = cuda → mps → cpu, and `autocast()` = bf16 autocast on CUDA only (MPS runs fp32).
  The training/eval scripts used in this session import it instead of hard-coding `"cuda"`, so the same code
  still runs unchanged on the Linux/CUDA laptop.
- Measured speed (bs 24, 240×320, fp32 MPS): BetaNet 364 ms/step, RefineNet 569 ms/step.
- Disk: ~19 GB free at start. Big datasets (SQUID other sites 40 GB, DIODE, full FLSea) deliberately NOT downloaded.
- Internet was slow (0.2–2 MB/s), which dominated wall time.

### Problem found: BetaNet weights are not in git
`*.pt` is git-ignored except `results/refine_*/best.pt`. `results/baseline_v1_60ep_bigdata/` only has `log.json`.
RefineNet needs the frozen BetaNet it was trained on, so the committed `refine_v1`/`refine_ft` weights cannot be
reproduced exactly here. **Decision:** retrain BetaNet on this Mac with the documented v1 recipe under a NEW name
(`baseline_v1_mac`, so the team's `baseline_v1_60ep_bigdata/log.json` is untouched), then train every RefineNet
variant of this session on top of it. Comparisons inside this session are therefore apples-to-apples; numbers from
earlier sessions (CUDA BetaNet) are quoted side by side but are not the same base.
**To the next agent:** if you get the original `baseline_v1_60ep_bigdata/best.pt` from the Linux laptop, put it in
that folder and re-run the session-5 commands with `--base baseline_v1_60ep_bigdata`.

**Update (13:2x):** the user supplied the original weights, now at `results/baseline_v1_60ep_bigdata/best.pt`. See the
log below. Session 5 now uses them; `baseline_v1_mac`, `refine_v1_mac` and `refine_ft_mac` are only a reproduction check.

### What was added (code)
- `src/device.py`: device helper (above). Scripts switched to it: train, refine, finetune_refine, eval_refine,
  eval_synth, eval_real, eval_day1, depth.
- `src/mac_setup.sh`: venv + `data/` rebuild on a Mac (NYU, NYU big, Sea-thru small; raw NYU parquets deleted after prep).
- `src/prep_uieb.py`: `resize_pair` / `save_pairs` factored out (same output as before) so other paired sets reuse them.
- `src/prep_extra_pairs.py` (new): LSUI (2,000 of 4,279 pairs, last 400 = test) and EUVP-Dark (2,500 of 5,550,
  last 500 = test) → `data/{lsui,euvp}_pairs.npz`, same format as `uieb_train.npz` (DAv2 metric-indoor depth).
  Subsets only because of disk; use all pairs on the 120 GB laptop (change `N_KEEP`).
- `src/finetune_refine.py`: `--mix uieb:0.4,lsui:0.4,euvp:0.2` draws each step's real photo from several sources at
  the SAME steps per epoch as the UIEB-only recipe (fair compute). Default `uieb:1` = the original code path.
- `src/stress_test.py` (new): all systems × all test sets → `results/stress_test/summary.json`, per-image json,
  figures in `figures/9_stress_test/`. Sets and metrics are described in its docstring. TUB note: against the
  stage-1 *raw* photo the raw input always looks best (any restoration changes colours), so the main TUB metric is
  **drift** = colour error between a method's output at turbidity stage k and its own output at stage 1.
- `src/run_session5.sh` (new): the whole session-5 pipeline, resumable (skips finished steps).

### Weights and git
`results/refine_*/best.pt` are un-ignored (~0.5 MB each) and go in the final commit. BetaNet is NOT in git: `*.pt` stays
ignored and the user copies `results/baseline_v1_60ep_bigdata/best.pt` (45 MB) between machines by hand.
The user runs the git commands for the code push themselves. No `Co-Authored-By` or other Claude attribution in commit messages or PRs (user's rule). GitHub account on the Mac: `na23b068-svg` (push rights: yes).

### Handoff: bigger run on the RTX 3060 laptop (120 GB free, 6 GB VRAM)
`refine_mix` and the stress test are done on the Mac (see Findings), do not repeat them. Not tested on this laptop yet.
Rules: `git pull` before you push, leave `main` unmerged, keep `*.pt` BetaNet weights out of git.
1. Pull branch `RefineNet`; copy the BetaNet weights to `results/baseline_v1_60ep_bigdata/best.pt`.
2. If `data/` is missing: `src/mac_setup.sh` is Mac-only, so install CUDA torch plus the packages it lists, then from
   `src/` run `download_data.py nyu uieb seathru`, `prep_nyu.py`, `prep_nyu_more.py`, `prep_seathru.py`, `prep_uieb.py`.
   `prep_seathru.py` overwrites `results/depth_eval.json`; restore it with `git checkout`.
3. Check the base, from `src/`: `python finetune_refine.py --base baseline_v1_60ep_bigdata --name refine_ft --test`. It must
   give about 20.20 dB / ΔE 11.51 on UIEB, then `git checkout results/refine_ft/ft_eval.json` (keep session 4's file).
4. Get LSUI and EUVP Dark (sources in `docs/external_datasets.md`) into `data/external/lsui/LSUI/{input,GT}/` and
   `data/external/euvp_dark/train.parquet`. Set `N_KEEP` in `src/prep_extra_pairs.py` to 4279 / 5550, then run
   `prep_extra_pairs.py lsui euvp`. This changes the test split, so scores are not comparable with the subset runs.
5. Remove duplicates before training: 21/400 LSUI and 3/500 EUVP test photos had a copy in training, because LSUI repeats
   scenes. `src/stress_clean.py` has the check; move any test photo with a training copy out of the test split, or
   run it afterwards to re-score without them.
6. Train all-data mix, from `src/`: `python finetune_refine.py --base baseline_v1_60ep_bigdata --init refine_v1 --name
   refine_mix_full --mix uieb:0.4,lsui:0.4,euvp:0.2 --epochs 40`, then the same with `--test`. Steps per epoch are
   fixed at 640, so `--epochs` sets how much data it sees (12 epochs is about 2 passes over the LSUI subset).
7. Train LSUI first, then UIEB: `--init refine_v1 --name refine_lsui --mix lsui:1`, then `--init refine_lsui --name
   refine_lsui_uieb` with no `--mix`. It was level with UIEB-only for main's touch-up, while mixing made it worse.
8. Stress-test the winners: `python stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_ft,<new names>`, then
   `stress_clean.py` (it names `refine_ft` and `refine_mix` in its CI pairs, so edit those). The test sets in
   `data/external/` (TUB, RUIE, U45, TankImage-I) are not in git: fetch them or use `--only` for the rest.
9. RefineNet on Day 1b output is the most promising idea but needs new code: `betanet_restore` in `finetune_refine.py`
   and `systems` in `stress_test.py` must feed `safe_restore(...)` output, and the stage-1 weights expect raw BetaNet
   output, so start with `refine_ft` as `--init`.
10. If VRAM runs out, lower the batch size: `--bs-synth` in `finetune_refine.py`, `--bs` in `refine.py` and `train.py`.
Log results here, newest at the top of a new session entry. `--epochs 40` is a guess at the right length, check the curve.

### Status checklist (kept current)
- [x] Clone branch `RefineNet` → `~/Desktop/MLOE-RefineNet`; graphify graph built (`graphify-out/`, code + docs, figures skipped)
- [x] Dataset survey → `docs/external_datasets.md`
- [x] Small external datasets downloaded into `data/external/` (git-ignored)
- [x] Venv + base data rebuilt (`src/mac_setup.sh`, log `results/logs/mac_setup.log`)
- [x] BetaNet: original weights supplied by the user (Mac retrain `baseline_v1_mac` kept only as a check)
- [x] RefineNet stage 1 and UIEB-only stage 2 on the original: `refine_v1`, `refine_ft` (session 4)
- [x] RefineNet stage 2 with UIEB + LSUI + EUVP on the original: `refine_mix`
- [x] Stress test + figures
- [x] Code push (`1a8c47d`), docs, graphify update, final commit with `refine_mix` weights and results

### Log
- (01:3x) Cloned, graphify run (513 nodes, 1,477 edges, 29 communities).
- (02:0x) Dataset survey; downloads paused once for a disk check at the user's request, resumed with small sets only.
- (02:1x) Device patch (`src/device.py`), `.venv/` added to `.gitignore`, `src/mac_setup.sh` started.
- (02:1x) EUVP pairs (2,500 → 0.97 GB) and LSUI pairs (2,000 → 1.2 GB) prepared. Stress test smoke-tested end to
  end with random weights on U45 + TUB (320 TUB images = 8 test experiments × cameras × 10 stages); outputs deleted.
- (02:45) `mac_setup.sh` done (NYU big = 7,140 + 288 frames, NYU test 654, Sea-thru small; DAv2 depth numbers
  match the CUDA ones: relative AbsRel 0.072). **Disk hit 3.2 GB free.** Freed: uv cache, the partial FLSea sample,
  `data/nyu_train.npz` (contained in `nyu_train_big.npz`), raw `data/seathru/` (converted to `seathru_small.npz`).
  All rebuildable with `src/mac_setup.sh`. FLSea sample and SeaThru-NeRF were NOT downloaded in the end (disk).
- (02:5x) `src/run_session5.sh` running (prep_uieb → BetaNet → RefineNet stage 1 → stage 2 ×2 → stress test).
- (04:31) BetaNet `baseline_v1_mac` done: 60 epochs, ~103 s/epoch on MPS (CUDA: 39 s), final synthetic val PSNR 25.19.
- (05:11) RefineNet stage 1 `refine_v1_mac` done (12 epochs). NYU test 654: BetaNet 21.00 dB / ΔE00 13.29 →
  + RefineNet 23.47 / 10.43 (oracle 25.11 / 9.28). Same pattern as session 4's `refine_v1`.
- (05:12) **Disk fell to ~0.35 GB free.** Cause: swap (5.3 GB), because the two parallel stage-2 runs each hold the
  1.9 GB NYU set in unified memory. Fix: killed the mix run (it restarts after the UIEB-only one, i.e. stage 2
  now runs one at a time on a 16 GB Mac) and deleted `data/external/lsui` + `data/external/euvp_dark` (raw downloads,
  already converted into `data/{lsui,euvp}_pairs.npz`; re-download per `docs/external_datasets.md`).
- (~06:30–11:07) The Mac was shut down overnight, which killed the run at epoch 3 of 12 (OneCycle LR, not resumable).
  After the reboot swap was gone and disk read 26 GB free (was ~0.6 GB). Swap explains only ~5 GB of that jump; the
  other ~20 GB was something else the reboot cleared (cause not identified, probably system temp files or caches).
- (11:2x) `run_session5.sh` stage 2 changed to one run at a time, partial `refine_ft_mac` deleted, pipeline restarted
  under `caffeinate -i` (stops idle sleep, not lid-close sleep).
- (12:58) `refine_ft_mac` done: UIEB160 20.23 dB / ΔE 11.55 (its BetaNet: 17.43 / 13.66), synthetic val 26.82 (25.34).
  Matches session 4's `refine_ft` (20.20 / 11.51). `refine_mix_mac` started.
- (13:1x) **Checked `origin/main` (3 commits by Aarav, newest 3 Nov 17:12, not on this branch).** This clone is
  single-branch, so `main` was fetched by hand; nothing merged.
  - No physics change: `physics.py`, `model.py`, `metrics.py`, `safe_restore.py`, `depth.py`, `eval_day1.py` identical.
    `train.py` only gains opt-in flags (`--patience`, `--stored-water`, `--depth-input`), defaults keep the v1 recipe.
  - Main's chosen method is v1 + Day 1b + `day3a_touchup_global` (same role as RefineNet): 90 UIEB val 21.68 dB /
    ΔE 9.66, grey charts ~5°. Compare with RefineNet on `uieb90` and charts (different BetaNet, so indicative). No
    `.pt` files on main: neither `day3a` nor the original BetaNet can be run here.
  - Main found that mixing LSUI in made the touch-up worse (ΔE 9.66 → 11.11), while LSUI first and UIEB after was
    level. Read `refine_mix_mac` with that in mind.
  - User decision: keep `main` separate, session 5 pushes to `RefineNet` without merging it.
  - Merging main later: `src/train.py` conflicts with the device patch, `results/logs/prep_lsui.log` is on both
    sides, and main's new scripts hard-code `DEV = "cuda"`.
- (13:1x) Duplicate check of this session's splits (main's method, thumbnail correlation ≥ 0.955). Test photos with a
  duplicate in training: UIEB160 2/160, UIEB90 1/90 (UIEB vs itself, so session 4 had them too), LSUI400 21/400,
  EUVP500 3/500. No cross-dataset duplicates into UIEB160/90. Re-score without these photos before quoting
  LSUI/EUVP results.
- (13:21) User supplied the original BetaNet (`~/Downloads/best.pt`, internal run name `beta_rgbd_big`), copied to
  `results/baseline_v1_60ep_bigdata/best.pt`. Check, with `refine_ft` on UIEB160: BetaNet 17.67 / ΔE 13.24, RefineNet
  20.20 / ΔE 11.51, synthetic 27.82 → 29.92. Session 4 logged 17.68 / 13.24, 20.20 / 11.51, 27.82 → 29.91: same model.
- (13:2x) Same check on the Mac pair: BetaNet 17.43 / 13.66, RefineNet 20.23 / 11.55, synthetic 25.34 → 26.82. So the
  Mac BetaNet is 2.5 dB worse on synthetic, with higher train loss (0.578 vs 0.555) and coefficient error (0.616 vs
  0.534). Cause not found; the `train.py` patch is device lines only. `val_psnr` in the two training logs is not
  comparable across machines (Mac reads higher while the same-machine test reads lower).
- (13:2x) Stopped `refine_mix_mac` at about epoch 5 (its log: `s5_refine_mix_mac_stopped.log`). `run_session5.sh` now
  trains `refine_mix` on the original BetaNet from `refine_v1`, then stress-tests `refine_v1`, `refine_ft`, `refine_mix`.
- (14:38) `refine_mix` done: UIEB160 19.95 / ΔE 11.75, synthetic val 29.82. First stress run crashed on the Sea-thru
  charts (RefineNet needs sides divisible by 8). `stress_test.py` now pads and crops; rerun finished 15:15 (18 min).
- (15:2x) `src/stress_clean.py` added and run (scores without duplicates). Prep logs renamed `s5_prep_*.log` because
  main also has a `results/logs/prep_lsui.log`. Findings at the top of this entry.
- (15:3x) Graphify updated with `graphify update .` (576 nodes, 1,704 edges, 39 communities; code only). The 5 edited docs
  were not re-extracted and figures stay excluded; run `/graphify --update` for docs. `graphify-out/` is not in git.
- (16:xx) Bug fixed in `finetune_refine.py`: a single-source `--mix lsui:1` silently trained on UIEB (it took the original
  path whenever the mix had one source). `refine_ft` and `refine_mix` not affected. Handoff steps rewritten with commands.
