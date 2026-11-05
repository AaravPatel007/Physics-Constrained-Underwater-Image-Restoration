#!/bin/bash
# Session 6: C60 on all six refiners once the final stress test is done. uieb90 rides along because the bar figure needs a paired set.
cd ~/Downloads/MLOE---Group-4
until grep -qE 'S6_FINAL_DONE|S6_FINAL_FAILED' results/logs/s6_final.log; do sleep 20; done
cd src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
~/Downloads/Garment/piper_x/.venv_vla/bin/python stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_ft,refine_d1b,refine_mix_full,refine_mix_d1b,refine_lsui,refine_lsui_uieb --only uieb90,c60 --out stress_test_c60 && echo S6_C60_DONE || echo S6_C60_FAILED
