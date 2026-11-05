#!/bin/bash
# Session 6: Aarav's own day3a weights (his best.pt) through his scorer, then our stress test next to day3b (retrained here) and RefineNet.
cd ~/Downloads/MLOE---Group-4/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY day3_touchup.py eval > ../results/logs/s6_day3_eval.log 2>&1 || { echo S6_AUDIT2_FAILED; exit 1; }
echo S6_EVAL_DONE
$PY stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_d1b,refine_mix_d1b --touchups day3a_touchup_global,day3b_touchup_global_plus_local --only synth,uieb90,lsui,euvp,tub,uccs_blue,uccs_green,uccs_blue-green,uiqs,u45,tank,c60,charts --out stress_test_day3 > ../results/logs/s6_stress_c60.log 2>&1 || { echo S6_AUDIT2_FAILED; exit 1; }
$PY stress_clean.py --out stress_test_day3 >> ../results/logs/s6_stress_c60.log 2>&1 || { echo S6_AUDIT2_FAILED; exit 1; }
echo S6_AUDIT2_DONE
