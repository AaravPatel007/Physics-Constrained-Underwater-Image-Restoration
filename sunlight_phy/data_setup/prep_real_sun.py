"""Real evaluation data: the UIEB val split and the 24 Sea-thru frames used by src/eval_real.py.

UIEB: data/uieb/data/val-00000-of-00001-*.parquet (90 raw/reference pairs, as src/download_data.py uieb fetches).
Sea-thru: the 24 frames listed in results/real_eval_big.json (2 colour-chart frames of D1 + frames of D1-D5), linear
PNG + SfM range TIF, shrunk to 1/4 resolution exactly like src/prep_seathru.py, written to
data/seathru_small_sun.npz (names, lin_i, gt_i). The DAv2 part of prep_seathru.py is skipped (the evaluation uses
the SfM range) so results/depth_eval.json is not rewritten. The full-size files are deleted afterwards.

  cd src && python ../sunlight_phy/data_setup/prep_real_sun.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_hf import DATA, fetch, list_files  # noqa: E402

SCALE = 4
REF = HERE.parents[1] / "results" / "real_eval_big.json"


def main():
    uieb = [f for f in list_files("Hikari0608/UIEB", "data") if "/val-" in f]
    fetch("Hikari0608/UIEB", uieb, DATA / "uieb")

    want = list(json.load(open(REF))["seathru"]["frames"])  # e.g. "D1/T_S03272"
    files = []
    for site in sorted({w.split("/")[0] for w in want}):
        tree = list_files("bhowmikabhimanyu/seathru-dataset", site)
        for w in [w for w in want if w.startswith(site + "/")]:
            stem = w.split("/")[1]
            files += [p for p in tree if ("/linearPNG/" in p and p.endswith(f"/{stem}.png")) or p.endswith(f"depth{stem}.tif")]
    raw = DATA / "seathru_eval_raw"
    fetch("bhowmikabhimanyu/seathru-dataset", files, raw, workers=4)

    names, arrays = [], {}
    for w in want:
        site, stem = w.split("/")
        png = next((raw / p for p in files if p.endswith(f"/{stem}.png")), None)
        tif = next((raw / p for p in files if p.endswith(f"depth{stem}.tif")), None)
        if png is None or tif is None:
            print("skip (image or range map not in the HF dataset)", w, flush=True)
            continue
        lin = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
        h, wd = lin.shape[0] // SCALE, lin.shape[1] // SCALE
        lin = cv2.resize(lin, (wd, h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(np.array(Image.open(tif), dtype=np.float32), (wd, h), interpolation=cv2.INTER_NEAREST)
        i = len(names)
        names.append(w)
        arrays[f"lin_{i}"], arrays[f"gt_{i}"] = lin, gt
        print(w, lin.shape, f"range {np.nanmin(gt[gt > 0]):.1f}-{np.nanmax(gt):.1f} m", flush=True)
    np.savez_compressed(DATA / "seathru_small_sun.npz", names=np.array(names), **arrays)
    shutil.rmtree(raw)
    print("REAL_DONE", len(names), "Sea-thru frames", flush=True)


if __name__ == "__main__":
    main()
