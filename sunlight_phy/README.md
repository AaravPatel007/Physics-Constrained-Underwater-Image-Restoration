# sunlight_phy: BetaNet with sunlight computed from depth below the surface

Experiment branch `sunlight_phy_model` (from `main` @ `721a609`). This folder holds everything the experiment adds.
Nothing outside it was changed: the baseline code, checkpoints, results and docs on `main` are untouched.

**Question:** if the network predicts **one physical number**, the depth below the surface d_s, and the sunlight
colour is computed as **E_c = E0_c · exp(−K_d,c · d_s)**, is the model better than the baseline BetaNet, which
predicts the three sunlight numbers E_R, E_G, E_B directly?

| | Baseline BetaNet (`baseline_v1_60ep_bigdata` recipe) | Physics-informed (`SunDepthBetaNet`) |
|---|---|---|
| Outputs | 12: β_D(3), β_B(3), B∞(3), **E(3)** | 10: β_D(3), β_B(3), B∞(3), **d_s(1)** |
| Sunlight E | predicted | `E = exp(−0.8·β_D·d_s)`, normalised, floored at 0.03 |
| Everything else | — | identical: backbone, inputs, β/B∞ heads, restoration, data, training recipe |

**Result (4 Nov 2026):** no improvement. On the synthetic test the new model reaches 21.09 vs 21.40 dB PSNR and
13.79 vs 13.38 CIEDE2000. On real photos it is worse when its own E is used, and about equal when E is replaced
(gray-world / Day 1b). Its depth estimate has r = 0.73 against the true depth, with a mean error of 1.75 m; the
best possible from E is 0.65 m. Details and caveats (one seed per model) are in docs/02_results.md.

Start with the documents:
1. [docs/01_design_and_physics.md](docs/01_design_and_physics.md): the physics, the decisions and why, and what the network can know.
2. [docs/02_results.md](docs/02_results.md): all results, with figures and interpretation.
3. [docs/03_reproduction_check.md](docs/03_reproduction_check.md): is our retrained baseline the same as the original v1?

## Folder layout

```
sunlight_phy/
  README.md                 this file
  docs/                     design / results / reproduction check
  sun_physics.py            E from depth (physics layer) + generator copy that returns the depth labels
  model_sun.py              SunDepthBetaNet
  train_sun.py              training (v1 recipe + depth loss)
  verify_rng.py             proves the depth labels belong to exactly the baseline's waters
  eval_sun.py               synthetic evaluation (654 test rooms, 10 Jerlov types, latency, error attribution)
  eval_real_sun.py          real photos: UIEB 90 val, Sea-thru colour charts
  diagnose_depth.py         what the predicted depth tracks; cost of K_d = 0.8 beta_D -> results/depth_diagnosis.json
  figs_sun.py               all figures -> figures/
  repro_check.py            retrained baseline vs committed v1 numbers -> docs/03_reproduction_check.md
  run_all.sh                everything, in order
  data_setup/               downloads (parallel curl) and data preparation
  runs/baseline_v1_repro/   baseline retrained here: best.pt, log.json
  runs/sun_v1/              physics-informed model: best.pt, log.json, config.json
  runs/logs/                console logs
  results/                  JSON summaries + per-image arrays
  figures/                  PNG figures (1_ ... 14_)
```

Checkpoints (`*.pt`) and per-image arrays (`*.npz`) are git-ignored (repository rule), so they only exist on the
machine that produced them.

## Environment and data (this machine)

- Everything downloaded lives in `~/Downloads/oeml_project_files/`:
  - `venv/` (Python 3.12, torch 2.11 + CUDA 12.8 for the RTX 5060, torchvision, scikit-image, opencv, transformers, …)
  - `data/` (the repository's `data` is a symlink to it; the link is excluded locally in `.git/info/exclude`)
  - `hf_home/` (Depth Anything V2 weights), `torch_home/` (ImageNet ResNet-18 weights), `logs/`
- Package wheels came from the existing uv cache (`~/.cache/uv`) where possible, to avoid re-downloading ~3 GB.
- NYU data were rebuilt from exactly the shards v1 used. The committed download logs list 19 train shards; the
  original `src/download_data.py` lists only 6 of them. `src/prep_nyu.py` and the 30-shard extension (same pick
  as `src/prep_nyu_more.py`) rebuild them. The raw parquet shards were deleted after building the `.npz` files, to
  save disk.

## Reproduce

```bash
cd <repo root>      # branch sunlight_phy_model
bash sunlight_phy/run_all.sh
```

Steps and their order are in `run_all.sh`. Single steps:

```bash
PY=~/Downloads/oeml_project_files/venv/bin/python
export HF_HOME=~/Downloads/oeml_project_files/hf_home TORCH_HOME=~/Downloads/oeml_project_files/torch_home
$PY sunlight_phy/verify_rng.py
(cd src && $PY train.py --model beta --name ../sunlight_phy/runs/baseline_v1_repro --data train_big --nval 288 --epochs 60 --murky-keep 0.4)
$PY sunlight_phy/train_sun.py --name sun_v1 --data train_big --nval 288 --epochs 60 --murky-keep 0.4
$PY sunlight_phy/eval_sun.py && (cd src && $PY ../sunlight_phy/eval_real_sun.py)
$PY sunlight_phy/diagnose_depth.py
$PY sunlight_phy/figs_sun.py && $PY sunlight_phy/repro_check.py
```
