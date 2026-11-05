#!/bin/bash
# session 6: background datasets not needed by the hand-off steps; everything resumes
cd "$(dirname "$0")/../../data/external"
PY=~/Downloads/Garment/piper_x/.venv_vla/bin/python
export OMP_NUM_THREADS=2
fetch() { # file md5 url : resume until the checksum passes (Zenodo drops connections)
  until [ "$(md5sum "$1" 2>/dev/null | cut -d' ' -f1)" = "$2" ]; do curl -sS -L -C - --retry 5 --retry-delay 30 -o "$1" "$3"; sleep 30; done; echo "OK $1"; }
(cd rcaustic && fetch R-CAUSTIC_v2.0.zip f94fe86eb9a25ecb56b99d9199bbeb2a https://zenodo.org/api/records/10728861/files/R-CAUSTIC_v2.0.zip/content) &
while kill -0 12257 2>/dev/null; do sleep 20; done   # let TUB (needed for the stress test) finish first
$PY -c "from huggingface_hub import snapshot_download as s; s('bhowmikabhimanyu/seathru-dataset',repo_type='dataset',local_dir='seathru_full',max_workers=2)" && echo SEATHRU_DONE
wait; echo S6_DOWNLOADS_DONE
