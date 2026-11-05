#!/bin/bash
# Session 6: once the reproduction ends, put the touch-ups through our stress test, then retrain day3b with two more seeds.
# uieb160 is skipped: the touch-ups trained on those photos. day3_seed.py = day3_touchup.py with the training seed changed only.
cd ~/Downloads/MLOE---Group-4
until grep -qE 'S6_DAY3_DONE|S6_DAY3_FAILED' results/logs/s6_day3.log; do sleep 20; done
grep -q S6_DAY3_FAILED results/logs/s6_day3.log && { echo S6_AUDIT_FAILED; exit 1; }
cd src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY stress_test.py --base baseline_v1_60ep_bigdata --refiners refine_d1b,refine_mix_d1b --touchups day3b_touchup_global_plus_local,day3a_touchup_global --only synth,uieb90,lsui,euvp,tub,uccs_blue,uccs_green,uccs_blue-green,uiqs,u45,tank,c60,charts --out stress_test_day3 || { echo S6_AUDIT_FAILED; exit 1; }
$PY stress_clean.py --out stress_test_day3 || { echo S6_AUDIT_FAILED; exit 1; }
echo S6_STRESS_DAY3_DONE
for s in 1 2; do SEED=$s $PY day3_seed.py train day3b_touchup_global_plus_local && SEED=$s $PY day3_seed.py eval || { echo S6_AUDIT_FAILED; exit 1; }; done
echo S6_AUDIT_DONE
