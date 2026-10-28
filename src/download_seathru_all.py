"""Stream every Sea-thru frame (linear RAW PNG + SfM range map) from Hugging Face, shrink it to
training size and delete the original, so the 38 GB dataset costs ~1 GB of disk.

Each frame -> data/seathru_train/<site>/<stem>.npz  (lin float16 HxWx3, z float16 HxW, metres; 0 = no depth)
Resumable: frames already written are skipped.

  python src/download_seathru_all.py [--workers 4] [--sites D1 D2 D3 D4 D5] [--stride 1]
Consecutive frames are near-duplicates, so --stride N keeps every N-th frame of each dive.
"""
import argparse
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "bhowmikabhimanyu/seathru-dataset"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "seathru_train"
H = 300  # stored height; training crops are 240 high


def pairs(sites):
    names = [s.rfilename for s in HfApi().dataset_info(REPO).siblings]
    depth = {n.split("/")[0] + "/" + n.split("/")[-1][5:-4]: n for n in names if n.endswith(".tif")}
    for n in sorted(names):
        if "/linearPNG/" in n and n.endswith(".png") and n.split("/")[0] in sites:
            key = n.split("/")[0] + "/" + n.split("/")[-1][:-4]
            if key in depth:
                yield key, n, depth[key]


def one(job, tmp_root):
    key, png, tif = job
    out = OUT / f"{key}.npz"
    if out.exists():
        return key, "skip"
    tmp = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        p = hf_hub_download(REPO, png, repo_type="dataset", local_dir=tmp)
        d = hf_hub_download(REPO, tif, repo_type="dataset", local_dir=tmp)
        lin = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
        w = int(round(lin.shape[1] * H / lin.shape[0]))
        lin = cv2.resize(lin, (w, H), interpolation=cv2.INTER_AREA)
        z = np.nan_to_num(np.array(Image.open(d), dtype=np.float32), nan=0.0)
        z = cv2.resize(z, (w, H), interpolation=cv2.INTER_NEAREST)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".tmp.npz"), lin=lin.astype(np.float16), z=z.astype(np.float16))
        out.with_suffix(".tmp.npz").rename(out)
        return key, f"ok valid={np.mean(z > 0):.2f}"
    except Exception as e:  # keep going; a rerun retries the failures
        return key, f"FAIL {e!r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sites", nargs="+", default=["D1", "D2", "D3", "D4", "D5"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    jobs = list(pairs(a.sites))[:: a.stride][: a.limit or None]
    tmp_root = ROOT / "data" / "_seathru_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} frames", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (key, msg) in enumerate(ex.map(lambda j: one(j, tmp_root), jobs)):
            print(f"[{i + 1}/{len(jobs)}] {key} {msg}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
