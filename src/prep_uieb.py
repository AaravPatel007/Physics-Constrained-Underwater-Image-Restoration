"""Prepare the UIEB *train* split for fine-tuning: resize (longest side 400, sides divisible by 8, as in
eval_real.uieb) and add Depth Anything V2 metric-indoor depth, since UIEB has no range maps.
The 90 val pairs used for evaluation are a separate parquet file and are not touched.

Writes data/uieb_train.npz: raw_i, ref_i (uint8 sRGB), z_i (float16 metres).
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import depth as dav2

ROOT = Path(__file__).resolve().parents[1]


def main():
    raws, refs = [], []
    for f in sorted((ROOT / "data" / "uieb" / "data").glob("train-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            a = np.array(Image.open(io.BytesIO(r["raw"]["bytes"])).convert("RGB"))
            b = np.array(Image.open(io.BytesIO(r["gt"]["bytes"])).convert("RGB"))
            s = 400 / max(a.shape[:2])
            size = (int(a.shape[1] * s) // 8 * 8, int(a.shape[0] * s) // 8 * 8)
            raws.append(cv2.resize(a, size, interpolation=cv2.INTER_AREA))
            refs.append(cv2.resize(b, size, interpolation=cv2.INTER_AREA))
        print(f.name, len(raws), flush=True)
    depths = dav2.predict(raws, "metric-indoor", bs=1)
    arrays = {}
    for i, (a, b, z) in enumerate(zip(raws, refs, depths)):
        arrays[f"raw_{i}"], arrays[f"ref_{i}"], arrays[f"z_{i}"] = a, b, np.clip(z, 0.1, 30).astype(np.float16)
    np.savez_compressed(ROOT / "data" / "uieb_train.npz", n=len(raws), **arrays)
    print("saved", len(raws), flush=True)


if __name__ == "__main__":
    main()
