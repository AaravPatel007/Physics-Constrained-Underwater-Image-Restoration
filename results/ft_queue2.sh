#!/bin/bash
# Fine-tuning queue 2: fixes for the coefficient collapse seen in ft_seathru (every dive -> same beta_D).
#   ft_nodive   : no same-dive loss
#   ft_teacher  : same-dive loss + teacher term (stay near the starting model's coefficients on real crops)
# then evals for every model and the before/after figures. Same laptop-friendly settings as ft_queue.sh.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY="nice -n 10 /home/aarav-patel/wec_rl/.venv/bin/python"
COOL="sleep 180"
cd "$(dirname "$0")/../src"
L=../results/logs

$PY finetune.py --name ft_nodive --steps 3000 --w-dive 0 > $L/ft_nodive.log 2>&1; $COOL
$PY finetune.py --name ft_teacher --steps 3000 --w-teacher 1.0 > $L/ft_teacher.log 2>&1; $COOL

for n in ft_nodive ft_teacher; do
    $PY eval_real.py --ckpt $n --tag _$n > $L/eval_real_$n.log 2>&1
done
$PY eval_finetune.py beta_rgbd ft_seathru ft_seathru_noanchor ft_nodive ft_teacher ft_uieb ft_uieb_ref > $L/eval_finetune.log 2>&1
echo FTDONE > $L/ft_done2
