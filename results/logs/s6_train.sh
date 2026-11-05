#!/bin/bash
# session 6 (RTX 3060 laptop, performance mode, no guard by the user's choice): hand-off steps 9, 6, 7. Re-runnable.
cd "$(dirname "$0")/../../src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
L=../results/logs; BASE=baseline_v1_60ep_bigdata
run() { # name, extra finetune args
  n=$1; shift
  [ -f ../results/$n/ft_eval.json ] && { echo "skip $n"; return; }
  echo "[$(date +%T)] train $n: $*"
  nice -n 10 $PY finetune_refine.py --base $BASE --name $n "$@" > $L/s6_$n.log 2>&1 || { echo "FAILED $n"; return 1; }
  nice -n 10 $PY finetune_refine.py --base $BASE --name $n --test >> $L/s6_$n.log 2>&1
  echo "[$(date +%T)] done $n"
}
run refine_d1b --init refine_ft --day1b
run refine_mix_full --init refine_v1 --mix uieb:0.4,lsui:0.4,euvp:0.2 --epochs 40
run refine_lsui --init refine_v1 --mix lsui:1
run refine_lsui_uieb --init refine_lsui
run refine_mix_d1b --init refine_mix_full --day1b --mix uieb:0.4,lsui:0.4,euvp:0.2
echo S6_TRAIN_DONE
