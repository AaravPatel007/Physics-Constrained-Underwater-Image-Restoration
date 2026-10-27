#!/bin/bash
# Training queue (sequential on the single 4 GB GPU)
cd /home/aarav-patel/oe5510/final_project/src
PY=/home/aarav-patel/wec_rl/.venv/bin/python
$PY train.py --model beta --name beta_rgbd                       > ../results/logs/beta_rgbd.log 2>&1
$PY train.py --model beta --no-depth --name beta_rgb             > ../results/logs/beta_rgb.log 2>&1
$PY train.py --model beta --lambda-img 0 --name beta_coefonly    > ../results/logs/beta_coefonly.log 2>&1
$PY train.py --model beta --lambda-coef 0 --name beta_imgonly    > ../results/logs/beta_imgonly.log 2>&1
$PY train.py --model unet --bs 8 --epochs 25 --lr 1e-3 --name unet_rgbd > ../results/logs/unet_rgbd.log 2>&1
echo DONE > ../results/logs/queue_done
