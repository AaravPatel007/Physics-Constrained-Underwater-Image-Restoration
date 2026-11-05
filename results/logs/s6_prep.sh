#!/bin/bash
# session 6: all LSUI + EUVP pairs (waits for the EUVP parquet)
cd "$(dirname "$0")/../../src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
until [ -s ../data/external/euvp_dark/train.parquet ]; do sleep 10; done
nice -n 10 $PY prep_extra_pairs.py lsui euvp
echo S6_PREP_DONE
