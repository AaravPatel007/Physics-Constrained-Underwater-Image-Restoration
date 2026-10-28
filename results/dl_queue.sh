#!/bin/bash
# Sea-thru streaming download queue: all of the small dives, every 2nd D2 frame, every 4th D1 frame.
PY=/home/aarav-patel/wec_rl/.venv/bin/python
cd "$(dirname "$0")/.."
$PY src/download_seathru_all.py --workers 8 --sites D3 D4 D5
$PY src/download_seathru_all.py --workers 8 --sites D2 --stride 2
$PY src/download_seathru_all.py --workers 8 --sites D1 --stride 4
echo ALLDONE
