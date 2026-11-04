#!/usr/bin/env bash
# Reproduce the v2 experiment end to end (see SUNLIGHT_DEPTH_V2_SUMMARY.md). Run from the repository root.
# Needs the data of sunlight_phy (data/nyu_train_big.npz, nyu_test.npz, UIEB val, seathru_small_sun.npz) and the
# venv in ~/Downloads/oeml_project_files. About 1 h of training (3 x ~20 min on an RTX 5060 Laptop) + ~20 min eval.
#   bash sunlight_depth_v2/run_all_v2.sh            everything
#   bash sunlight_depth_v2/run_all_v2.sh eval       only evaluation, figures and diagrams (checkpoints exist)
set -euo pipefail
P=${P:-$HOME/Downloads/oeml_project_files}
PY=$P/venv/bin/python
export HF_HOME=$P/hf_home TORCH_HOME=$P/torch_home OMP_NUM_THREADS=4
R=$(cd "$(dirname "$0")/.." && pwd)
L=$R/sunlight_depth_v2/runs/logs
mkdir -p "$L"
cd "$R"
COMMON="--data train_big --nval 288 --epochs 60 --murky-keep 0.4"

if [ "${1:-all}" != "eval" ]; then
  echo "== tests"; $PY sunlight_depth_v2/tests_v2.py
  echo "== train: SunDepthBetaNet v2 (16 outputs)"
  $PY sunlight_depth_v2/train_v2.py --name sun_v2 $COMMON 2>&1 | tee "$L/sun_v2.log"
  echo "== train: BetaNet (12) on the v2 water, src/train.py unchanged"
  $PY sunlight_depth_v2/train_existing_v2data.py --model baseline --name baseline_v2data $COMMON 2>&1 | tee "$L/baseline_v2data.log"
  echo "== train: SunDepthBetaNet v1 (10) on the v2 water, sunlight_phy/train_sun.py unchanged"
  $PY sunlight_depth_v2/train_existing_v2data.py --model sun_v1 --name sun_v1_v2data $COMMON 2>&1 | tee "$L/sun_v1_v2data.log"
fi
echo "== evaluate: synthetic"; $PY sunlight_depth_v2/eval_v2.py 2>&1 | tee "$L/eval_v2.log"
echo "== evaluate: real photos"; (cd src && $PY ../sunlight_depth_v2/eval_real_v2.py 2>&1 | tee "$L/eval_real_v2.log")
echo "== figures, tables, galleries"; $PY sunlight_depth_v2/figs_v2.py 2>&1 | tee "$L/figs_v2.log"
echo "== architecture diagrams"; $PY sunlight_depth_v2/figures/architecture/make_diagrams.py
