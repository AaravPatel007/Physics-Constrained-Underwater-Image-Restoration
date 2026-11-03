#!/bin/bash
# Review 2: "more epochs" + "train with DAv2 depth", as one matched pair. Both continue baseline_v1_60ep_bigdata on
# the same stored waters with early stopping (constant LR 1.5e-4, halved after 3 epochs without a better validation
# score, stop after 8; at most 100 epochs). The only difference is the depth the network is given:
#   day2f_added_more_epochs_true_depth : true depth  -> "does more training help?" + the control
#   day2e_added_dav2_depth             : DAv2's depth guess, run on the fake underwater image (as on real photos)
# ~50 min DAv2 precompute + 2 x (training until early stop + ~35 min tests). Logs: results/logs/<name>.log
set -u
cd "$(dirname "$0")"
PY=/home/aarav-patel/wec_rl/.venv/bin/python
export THREADS=8
while pgrep -f "prep_dav2_train.py" > /dev/null; do sleep 30; done  # a precompute started earlier may still run
[ -f ../data/nyu_train_big_dav2.npz ] || OMP_NUM_THREADS=8 nice -n 10 $PY prep_dav2_train.py > ../results/logs/prep_dav2_train.log 2>&1 \
  || { echo "precompute failed"; tail -5 ../results/logs/prep_dav2_train.log; exit 1; }
echo "$(date +%T) precompute done"
COMMON="--init baseline_v1_60ep_bigdata --epochs 100 --patience 8 --lr 1.5e-4 --murky-keep 0.4 --stored-water"
./day2_queue.sh day2f_added_more_epochs_true_depth $COMMON --depth-input true
./day2_queue.sh day2e_added_dav2_depth $COMMON --depth-input dav2
echo "$(date +%T) queue done"
