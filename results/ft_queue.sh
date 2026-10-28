#!/bin/bash
# Fine-tuning queue (session 2). Waits for the Sea-thru download, then:
#   stage 1  Sea-thru self-supervised fine-tune (+ ablation without the synthetic anchor)
#   stage 2  UIEB fine-tune on top of stage 1 (physics losses only / + reference loss)
#   evals    eval_real per model (tagged files), eval_finetune (before/after), fig_finetune
# Laptop-friendly: 2 CPU threads for torch, low priority, a cool-down pause between jobs.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY="nice -n 10 /home/aarav-patel/wec_rl/.venv/bin/python"
COOL="sleep 180"
cd "$(dirname "$0")/../src"
L=../results/logs
until grep -q ALLDONE $L/dl_seathru_all.log; do sleep 60; done

$PY finetune.py --name ft_seathru --steps 3000 > $L/ft_seathru.log 2>&1; $COOL
$PY finetune.py --name ft_seathru_noanchor --steps 3000 --w-synth 0 > $L/ft_seathru_noanchor.log 2>&1; $COOL
$PY finetune.py --name ft_uieb --init ft_seathru --steps 2000 --w-uieb 0.5 > $L/ft_uieb.log 2>&1; $COOL
$PY finetune.py --name ft_uieb_ref --init ft_seathru --steps 2000 --w-uieb 0.5 --w-ref 0.5 > $L/ft_uieb_ref.log 2>&1; $COOL

for n in ft_seathru ft_seathru_noanchor ft_uieb ft_uieb_ref; do
    $PY eval_real.py --ckpt $n --tag _$n > $L/eval_real_$n.log 2>&1
done
$PY eval_finetune.py beta_rgbd ft_seathru ft_seathru_noanchor ft_uieb ft_uieb_ref > $L/eval_finetune.log 2>&1
$PY fig_finetune.py ft_seathru ft_uieb ft_uieb_ref > $L/fig_finetune.log 2>&1
echo FTDONE > $L/ft_done
