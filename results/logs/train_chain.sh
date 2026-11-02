#!/bin/bash
# BetaNet 60ep (exact SUMMARY.md recipe), then RefineNet on top. thermal_guard throttles both.
set -e
cd "$(dirname "$0")/../.."
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
nohup $PY src/thermal_guard.py > results/logs/thermal_guard.log 2>&1 &
GUARD=$!
trap "kill $GUARD 2>/dev/null" EXIT
cd src
$PY train.py --model beta --name baseline_v1_60ep_bigdata --data train_big --nval 288 --epochs 60 --murky-keep 0.4
$PY refine.py --base baseline_v1_60ep_bigdata --name refine_v1 --data train_big --nval 288
echo TRAIN_CHAIN_DONE
