#!/bin/bash
# Day 2 (review 2): train one run (optional), then evaluate it next to baseline_v1_60ep_bigdata (the current baseline).
#   ./day2_queue.sh <run name> [train.py args ...]     no train args = the run is already trained
# Model names: docs/MODELS.md.
# Run 1: ./day2_queue.sh day2a_added_murky_water        (trained with --murky-keep 1.0 from scratch, --epochs 60)
# Run 2: ./day2_queue.sh day2b_added_deep_red_starved_water --init baseline_v1_60ep_bigdata --epochs 20 --lr 1.5e-4 --murky-keep 0.4 --low-red 0.4
# Run 3: THREADS=4 ./day2_queue.sh day2c_added_camera_flaws --init baseline_v1_60ep_bigdata --epochs 20 --lr 1.5e-4 --murky-keep 0.4 --camera 0.5
# Outputs: results/review2_day2/<run name>/ (eval.json + uieb_outputs_<model>.npz from eval_day1.py,
# real_eval_*.json, dive_separation.json). Logs: results/logs/<run name>.log (training), <run name>_eval.log.
set -u
cd "$(dirname "$0")"
PY=/home/aarav-patel/wec_rl/.venv/bin/python
NEW=$1; shift
OUT=../results/review2_day2/$NEW
mkdir -p $OUT
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}

pgrep -f "src/thermal_guard.py" > /dev/null || (cd .. && exec $PY src/thermal_guard.py >> results/logs/thermal_guard.log 2>&1) &

if [ $# -gt 0 ]; then
  echo "$(date +%T) training $NEW: $*"
  nice -n 10 $PY train.py --model beta --name $NEW --data train_big --nval 288 "$@" > ../results/logs/$NEW.log 2>&1 \
    || { echo "training failed"; tail -5 ../results/logs/$NEW.log; pkill -f "src/thermal_guard.py"; exit 1; }
fi
echo "$(date +%T) training finished: $(tail -1 ../results/logs/$NEW.log | cut -c1-80)"

# 1. Day 1 tests (UIEB 90 + fins, charts, synthetic + deep red-starved synthetic), with and without the Day 1 fixes
nice -n 10 $PY eval_day1.py --out review2_day2/$NEW --models baseline_v1_60ep_bigdata $NEW --synth-low-red && echo "$(date +%T) eval_day1 done"

# 2. real dives + UIEB with the original recipe (same as earlier real_eval_*.json)
nice -n 10 $PY eval_real.py --ckpt $NEW --tag _$NEW && mv ../results/real_eval_$NEW.json \
  ../results/real_seathru_outputs_$NEW.npz $OUT/ && mv ../results/uieb_outputs_$NEW.npz $OUT/real_eval_uieb_outputs_$NEW.npz
echo "$(date +%T) eval_real done"

# 3. dive separation (does it tell the 5 dives apart, esp. murky D4?) + held-out dive losses
nice -n 10 $PY eval_finetune.py baseline_v1_60ep_bigdata $NEW && $PY -c "
import json; d = json.load(open('../results/finetune_eval.json'))
json.dump({k: d[k] for k in ['baseline_v1_60ep_bigdata', '$NEW']}, open('$OUT/dive_separation.json', 'w'), indent=1)"
echo "$(date +%T) eval_finetune done"

pkill -f "src/thermal_guard.py"
echo "$(date +%T) all done"
