#!/bin/bash
# Session 6: HICRD test pairs on every system, after the day3a stress test frees the GPU. uieb90 rides along for the bar figure.
cd ~/Downloads/MLOE---Group-4
until grep -qE 'S6_AUDIT2_DONE|S6_AUDIT2_FAILED' results/logs/s6_audit2.log; do sleep 15; done
cd src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_d1b,refine_mix_d1b --touchups day3a_touchup_global,day3b_touchup_global_plus_local --only uieb90,hicrd --out stress_test_hicrd && $PY stress_clean.py --out stress_test_hicrd && echo S6_HICRD_DONE || echo S6_HICRD_FAILED
