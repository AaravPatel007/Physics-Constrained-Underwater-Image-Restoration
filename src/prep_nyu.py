"""Decode NYU-v2 parquet shards into compact arrays at 240x320.

Train: 6 shards (different rooms), every 2nd frame (consecutive video frames are near duplicates).
Test: the official 654-image NYU test split (rooms never seen in training).
Depth holes (0) are inpainted. Output: data/nyu_{train,test}.npz with rgb uint8 (N,H,W,3), depth float16 metres.
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
H, W = 240, 320


def decode(files, step):
    rgbs, depths = [], []
    for f in files:
        tab = pq.read_table(f).to_pylist()
        for row in tab[::step]:
            rgb = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
            d = np.array(Image.open(io.BytesIO(row["depth_map"]["bytes"])), dtype=np.float32)
            # crop the white/black border NYU frames have, then resize
            rgb, d = rgb[8:-8, 8:-8], d[8:-8, 8:-8]
            hole = (d < 0.1).astype(np.uint8)
            if hole.mean() > 0.3:
                continue
            if hole.any():
                scale = d.max()
                d = cv2.inpaint((d / scale * 65535).astype(np.uint16), hole, 5, cv2.INPAINT_NS).astype(np.float32) / 65535 * scale
            rgbs.append(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA))
            depths.append(cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float16))
        print(f.name, len(rgbs), flush=True)
    return np.stack(rgbs), np.stack(depths)


if __name__ == "__main__":
    files = sorted((DATA / "nyu" / "data").glob("*.parquet"))
    for split, fs, step in [("train", [f for f in files if "train" in f.name], 2),
                            ("test", [f for f in files if "validation" in f.name], 1)]:
        rgb, depth = decode(fs, step)
        np.savez(DATA / f"nyu_{split}.npz", rgb=rgb, depth=depth)
        print(split, rgb.shape, float(depth.min()), float(depth.max()))
