#!/bin/bash
# Session 5 pipeline (Mac, MPS), on the original BetaNet: RefineNet stage 2 with UIEB+LSUI+EUVP -> stress test.
# The UIEB-only recipe and stage 1 on this base already exist (refine_ft, refine_v1, session 4).
# Needs src/mac_setup.sh done, data/lsui_pairs.npz + data/euvp_pairs.npz (prep_extra_pairs.py). Logs: results/logs/s5_*.log
#   bash src/run_session5.sh            (each step is skipped if its output already exists, so it can be re-run)
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
L=../results/logs
BASE=baseline_v1_60ep_bigdata
step() { echo "[$(date +%T)] $*" | tee -a $L/s5_pipeline.log; }

until grep -q MAC_SETUP_DONE $L/mac_setup.log 2>/dev/null && [ -f ../data/lsui_pairs.npz ] && [ -f ../data/euvp_pairs.npz ]; do sleep 60; done

if [ ! -f ../data/uieb_train.npz ]; then
  step "prep_uieb"; $PY prep_uieb.py > $L/s5_prep_uieb.log 2>&1
  rm -f ../data/uieb/data/train-*.parquet  # disk: only uieb_train.npz and the val parquet are read from here on
fi

# stage 2: the refine_ft recipe + LSUI + EUVP at the same number of steps
[ -f ../results/refine_mix/ft_eval.json ] || {
  step "stage 2: UIEB + LSUI + EUVP"; $PY finetune_refine.py --base $BASE --init refine_v1 --name refine_mix \
    --mix uieb:0.4,lsui:0.4,euvp:0.2 > $L/s5_refine_mix.log 2>&1
  $PY finetune_refine.py --base $BASE --name refine_mix --test >> $L/s5_refine_mix.log 2>&1; }

step "stress test"; $PY stress_test.py --base $BASE --refiners refine_v1,refine_ft,refine_mix > $L/s5_stress_test.log 2>&1
step "S5_PIPELINE_DONE"
