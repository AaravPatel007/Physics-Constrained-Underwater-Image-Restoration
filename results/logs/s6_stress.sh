#!/bin/bash
# Session 6 step 8: stress test + duplicate-free rescoring, written to stress_test_s6 so the committed run stays
cd ~/Downloads/MLOE---Group-4/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_ft,refine_d1b,refine_mix_full,refine_lsui,refine_lsui_uieb --out stress_test_s6 || { echo S6_STRESS_FAILED; exit 1; }
$PY stress_clean.py --out stress_test_s6 || { echo S6_STRESS_FAILED; exit 1; }
echo S6_STRESS_DONE
