# Agent logs (read this first when picking up the work)

Newest session on top. Each entry: what was done, what is running/unfinished, decisions and why, how to resume.
Project background: `SUMMARY.md`, model names: `docs/MODELS.md`, datasets: `docs/external_datasets.md`.

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
