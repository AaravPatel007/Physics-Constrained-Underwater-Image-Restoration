#!/usr/bin/env bash
# Reproduce the sunlight-from-depth experiment end to end (see README.md). Run from the repository root.
# Data, downloaded weights and the Python environment live in ~/Downloads/oeml_project_files (data/ is a link to it).
set -euo pipefail
P=${P:-$HOME/Downloads/oeml_project_files}
PY=$P/venv/bin/python
export HF_HOME=$P/hf_home TORCH_HOME=$P/torch_home OMP_NUM_THREADS=4
R=$(cd "$(dirname "$0")/.." && pwd)
L=$R/sunlight_phy/runs/logs
mkdir -p "$L"

step() { echo "== $1"; }

if [ ! -f "$R/data/nyu_train_big.npz" ]; then
  step "data: NYU (19 + 3 shards, then 30 more), UIEB val, Sea-thru eval frames"
  (cd "$R/sunlight_phy/data_setup" && python3 fetch_hf.py nyu-keep)
  (cd "$R/src" && $PY prep_nyu.py && $PY ../sunlight_phy/data_setup/prep_nyu_more_local.py)
  (cd "$R/src" && $PY ../sunlight_phy/data_setup/prep_real_sun.py)
fi

step "check: depth labels belong to exactly the baseline's waters"
$PY "$R/sunlight_phy/verify_rng.py"

step "train: baseline, v1 recipe, original src/train.py unchanged (~30-50 min)"
(cd "$R/src" && $PY train.py --model beta --name ../sunlight_phy/runs/baseline_v1_repro --data train_big --nval 288 \
  --epochs 60 --murky-keep 0.4 2>&1 | tee "$L/baseline_v1_repro.log")

step "train: physics-informed, same recipe (~30-50 min)"
$PY "$R/sunlight_phy/train_sun.py" --name sun_v1 --data train_big --nval 288 --epochs 60 --murky-keep 0.4 \
  2>&1 | tee "$L/sun_v1.log"

step "evaluate + figures"
$PY "$R/sunlight_phy/eval_sun.py" --baseline baseline_v1_repro --sun sun_v1 2>&1 | tee "$L/eval_sun.log"
(cd "$R/src" && $PY ../sunlight_phy/eval_real_sun.py --baseline baseline_v1_repro --sun sun_v1 2>&1 | tee "$L/eval_real_sun.log")
$PY "$R/sunlight_phy/diagnose_depth.py"
$PY "$R/sunlight_phy/figs_sun.py"
$PY "$R/sunlight_phy/repro_check.py"
