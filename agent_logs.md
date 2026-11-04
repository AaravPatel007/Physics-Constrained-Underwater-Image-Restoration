# Agent logs (read this first when picking up the work)

Newest session on top. Each entry: what was done, what is running/unfinished, decisions and why, how to resume.
Project background: `SUMMARY.md`, model names: `docs/MODELS.md`, datasets: `docs/external_datasets.md`.

---

## Session 5 (4–5 Nov 2026, MacBook M2, overnight, unattended)

**Goal (from the user):** retrain RefineNet with the *small* new datasets only (bigger runs later on the
other laptop with ~120 GB free), stress-test the pipeline (PSNR, SSIM, colour error ΔE00, …) and compare with
the earlier models, keep this log, update graphify at the end, one commit at the end that includes the new
RefineNet weights, pushed to branch `RefineNet`. No unnecessary commits.

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
The Mac is still finishing `refine_mix` and the stress test, so do not repeat those here. Not tested on this laptop yet.
1. Pull branch `RefineNet`; copy the BetaNet weights to `results/baseline_v1_60ep_bigdata/best.pt`. `src/mac_setup.sh`
   is Mac-only: install a CUDA torch and follow its data steps by hand if `data/` is missing.
2. Check the base: `refine_ft` on the 160 UIEB photos must give about 20.20 dB / ΔE 11.51.
3. Get LSUI and EUVP (`docs/external_datasets.md`) into the paths `src/prep_extra_pairs.py` reads and set `N_KEEP` to
   all pairs (4279 / 5550). This changes the test split, so scores are not comparable with the subset runs.
4. Remove duplicates first: 21/400 LSUI and 3/500 EUVP test photos had a copy in training (see the log below; main's
   `src/prep_lsui.py` has the thumbnail check). Also check against UIEB.
5. Train with `finetune_refine.py --mix`. Steps per epoch are fixed at 640, so raise `--epochs` to really cover the
   data (12 epochs is about 2 passes over the LSUI subset, under 1 over EUVP).
6. Also try LSUI first, then UIEB only (`--mix lsui:1`, then `--init` that run with the default mix): it was level
   with UIEB-only for main's touch-up, while mixing made it worse.
7. If VRAM runs out, lower the batch size: `--bs-synth` in `finetune_refine.py`, `--bs` in `refine.py` and `train.py`.

### Status checklist (kept current)
- [x] Clone branch `RefineNet` → `~/Desktop/MLOE-RefineNet`; graphify graph built (`graphify-out/`, code + docs, figures skipped)
- [x] Dataset survey → `docs/external_datasets.md`
- [x] Small external datasets downloaded into `data/external/` (git-ignored)
- [x] Venv + base data rebuilt (`src/mac_setup.sh`, log `results/logs/mac_setup.log`)
- [x] BetaNet: original weights supplied by the user (Mac retrain `baseline_v1_mac` kept only as a check)
- [x] RefineNet stage 1 and UIEB-only stage 2 on the original: `refine_v1`, `refine_ft` (session 4)
- [ ] RefineNet stage 2 with UIEB + LSUI + EUVP on the original: `refine_mix`
- [ ] Stress test + figures
- [ ] Code push (user runs it), then docs, graphify update, final commit with `refine_mix` weights and results

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
