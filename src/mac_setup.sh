#!/bin/bash
# Rebuild the venv + data/ on a Mac (Apple Silicon, MPS) from scratch. Same steps as the Linux data chain,
# minus the CUDA venv. Raw parquet shards are deleted after prep to save disk (the .npz files are what the code reads).
#   bash src/mac_setup.sh 2>&1 | tee results/logs/mac_setup.log
set -e
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python torch torchvision numpy opencv-python-headless scikit-image \
    transformers huggingface_hub pyarrow matplotlib pillow scipy
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
cd src
../$PY download_data.py nyu uieb seathru
../$PY prep_nyu.py
../$PY prep_nyu_more.py
../$PY prep_seathru.py && git -C .. checkout -- results/depth_eval.json  # keep the CUDA-made depth numbers
rm -f ../data/nyu/data/train-*.parquet ../data/nyu/data/validation-*.parquet
echo MAC_SETUP_DONE; du -sh ../data; df -h ~ | tail -1
