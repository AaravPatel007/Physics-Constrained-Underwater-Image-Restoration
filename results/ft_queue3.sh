#!/bin/bash
# Fine-tuning queue 3: teacher-weight sweep on top of the no-dive setting (ft_nodive was selected in round 2
# but only just kept the dives apart; teacher 1.0 barely moved from the starting model).
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY="nice -n 10 /home/aarav-patel/wec_rl/.venv/bin/python"
COOL="sleep 180"
cd "$(dirname "$0")/../src"
L=../results/logs

for w in 0.1 0.3; do
    n=ft_nodive_t${w/./}
    $PY finetune.py --name $n --steps 3000 --w-dive 0 --w-teacher $w > $L/$n.log 2>&1; $COOL
    $PY eval_real.py --ckpt $n --tag _$n > $L/eval_real_$n.log 2>&1
done
$PY eval_finetune.py ft_nodive_t01 ft_nodive_t03 > $L/eval_finetune3.log 2>&1
$PY select_ft.py > $L/select_ft.log 2>&1
echo FTDONE > $L/ft_done3
