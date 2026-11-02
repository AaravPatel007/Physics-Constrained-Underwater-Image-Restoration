#!/bin/bash
# Rebuild data/ from scratch: downloads, then prep (order matters: prep_nyu_more needs nyu_train.npz)
set -e
cd "$(dirname "$0")/../.."
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
$PY src/download_data.py nyu uieb seathru
$PY src/prep_nyu.py
$PY src/prep_nyu_more.py
$PY src/prep_uieb.py
$PY src/prep_seathru.py
echo DATA_CHAIN_DONE
