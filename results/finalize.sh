#!/bin/bash
# Runs after the training queue: synthetic eval -> real eval -> result figures
cd /home/aarav-patel/oe5510/final_project/src
PY=/home/aarav-patel/wec_rl/.venv/bin/python
until [ -f ../results/logs/queue_done ]; do sleep 10; done
$PY eval_synth.py > ../results/logs/eval_synth.log 2>&1
$PY eval_real.py  > ../results/logs/eval_real.log 2>&1
$PY fig_results.py > ../results/logs/fig_results.log 2>&1
echo FINALIZED > ../results/logs/finalized
