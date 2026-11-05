#!/bin/bash
# Session 6 overnight queue. Each step is independent: a failure is logged and the next step still runs.
cd ~/Downloads/MLOE---Group-4/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
L=../results/logs; BASE=baseline_v1_60ep_bigdata
MIX4=uieb:0.3,lsui:0.3,euvp:0.2,hicrd:0.2
stress() { # out, then args
  o=$1; shift
  echo "[$(date +%T)] stress $o"
  $PY -B stress_test.py --base $BASE --out $o "$@" > $L/s6_$o.log 2>&1 && $PY -B stress_clean.py --out $o >> $L/s6_$o.log 2>&1 && echo "[$(date +%T)] done $o" || echo "FAILED $o"
}
run() { # name, finetune args
  n=$1; shift
  [ -f ../results/$n/ft_eval.json ] && { echo "skip $n"; return; }
  echo "[$(date +%T)] train $n: $*"
  nice -n 10 $PY -B finetune_refine.py --base $BASE --name $n "$@" > $L/s6_$n.log 2>&1 || { echo "FAILED $n"; return; }
  nice -n 10 $PY -B finetune_refine.py --base $BASE --name $n --test >> $L/s6_$n.log 2>&1 || echo "FAILED test $n"
  echo "[$(date +%T)] done $n"
}
# 1. BridgeNet and the hybrid on every scored set, next to the two RefineNets and day3a
stress stress_test_bridge --refiners refine_d1b,refine_mix_d1b --touchups day3a_touchup_global --bridge --only uieb90,lsui,euvp,hicrd,tub,c60,charts
# 2. Does adding datasets help on water never seen? All runs start from refine_v1 (synthetic only) and keep the last epoch,
#    so the held-out set is never used for training or for picking a checkpoint. refine_only_uieb is the equal-budget control.
run refine_all_d1b --init refine_mix_d1b --day1b --mix $MIX4
run refine_only_uieb --init refine_v1 --day1b --last
run refine_no_hicrd --init refine_v1 --day1b --last --mix uieb:0.4,lsui:0.4,euvp:0.2
run refine_no_lsui --init refine_v1 --day1b --last --mix uieb:0.4,euvp:0.3,hicrd:0.3
run refine_no_euvp --init refine_v1 --day1b --last --mix uieb:0.4,lsui:0.4,hicrd:0.2
run refine_no_uieb --init refine_v1 --day1b --last --mix lsui:0.4,euvp:0.3,hicrd:0.3
# 3. Rematch: RefineNet on UIEB with no anchor and no synthetic term, the same freedom day3a had
run refine_d1b_free --init refine_ft --day1b --last --w-anchor 0 --w-synth 0
# 4. Everything through the full stress test
stress stress_test_final --refiners refine_mix_d1b,refine_all_d1b,refine_only_uieb,refine_no_hicrd,refine_no_lsui,refine_no_euvp,refine_no_uieb,refine_d1b_free --touchups day3a_touchup_global --only synth,uieb90,lsui,euvp,hicrd,tub,uccs_blue,uccs_green,uccs_blue-green,uiqs,u45,tank,c60,charts
# 5. day3b seed check
for s in 1 2; do echo "[$(date +%T)] day3b seed $s"; SEED=$s $PY -B day3_seed.py train day3b_touchup_global_plus_local > $L/s6_day3b_seed$s.log 2>&1 && SEED=$s $PY -B day3_seed.py eval >> $L/s6_day3b_seed$s.log 2>&1 || echo "FAILED seed $s"; done
echo S6_NIGHT_DONE
