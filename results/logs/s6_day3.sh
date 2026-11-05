#!/bin/bash
# Session 6: reproduce Aarav's touch-ups with his unmodified script (copied from origin/main, nothing merged)
cd ~/Downloads/MLOE---Group-4/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY day3_touchup.py cache && $PY day3_touchup.py train day3b_touchup_global_plus_local && $PY day3_touchup.py train day3a_touchup_global && $PY day3_touchup.py eval && echo S6_DAY3_DONE || echo S6_DAY3_FAILED
