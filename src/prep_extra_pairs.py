"""Extra real (photo, reference) pairs for RefineNet stage 2, same format as data/uieb_train.npz
(prep_uieb.save_pairs: longest side 400, DAv2 metric-indoor depth).

  lsui : LSUI (Peng et al. 2023), data/external/lsui/LSUI/{input,GT}/ -> data/lsui_pairs.npz
  euvp : EUVP "Underwater Dark" (HF Ken1053/EUVP parquet)            -> data/euvp_pairs.npz

Both are random subsets (seed 0) to fit a ~18 GB disk; N_TEST of each is the held-out test split (see TEST_SPLIT,
used by finetune_refine.py and stress_test.py). References are enhanced photos, not ground truth (like UIEB).

  python src/prep_extra_pairs.py lsui euvp
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from prep_uieb import resize_pair, save_pairs

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
N_KEEP = {"lsui": 2000, "euvp": 2500}  # shortcut: subsets for disk; use all (4279 / 5550) on the 120 GB laptop
TEST_SPLIT = {"lsui": 400, "euvp": 500}  # last N of the stored order = held-out test, never trained on


def rgb(x):
    return np.array(Image.open(x).convert("RGB"))


def lsui():
    gts = {p.stem: p for p in (EXT / "lsui").rglob("GT/*") if p.suffix.lower() in (".jpg", ".png")}
    ins = {p.stem: p for p in (EXT / "lsui").rglob("input/*") if p.suffix.lower() in (".jpg", ".png")}
    keys = sorted(set(gts) & set(ins))
    assert keys, "LSUI input/GT folders not found under data/external/lsui"
    pick = np.random.default_rng(0).permutation(len(keys))[:N_KEEP["lsui"]]
    pairs = [resize_pair(rgb(ins[keys[i]]), rgb(gts[keys[i]])) for i in pick]
    print(f"lsui: {len(keys)} pairs found, {len(pairs)} kept", flush=True)
    save_pairs([a for a, _ in pairs], [b for _, b in pairs], ROOT / "data" / "lsui_pairs.npz")


def euvp():
    rows = pq.read_table(EXT / "euvp_dark" / "train.parquet").to_pylist()
    pick = np.random.default_rng(0).permutation(len(rows))[:N_KEEP["euvp"]]
    pairs = [resize_pair(rgb(io.BytesIO(rows[i]["input_image"]["bytes"])), rgb(io.BytesIO(rows[i]["edited_image"]["bytes"])),
                         upscale=False) for i in pick]
    print(f"euvp: {len(rows)} pairs found, {len(pairs)} kept", flush=True)
    save_pairs([a for a, _ in pairs], [b for _, b in pairs], ROOT / "data" / "euvp_pairs.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["lsui", "euvp"])
    for w in ap.parse_args().which:
        globals()[w]()
