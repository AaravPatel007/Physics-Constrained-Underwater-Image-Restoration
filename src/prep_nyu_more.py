"""Stream 30 more NYU-v2 train shards (download one, decode at 240x320, delete it) and write
data/nyu_train_big.npz = new frames + all of nyu_train.npz.

The old frames go LAST and new shards are taken only from indices < 137, so the last 288 frames
(shards 137 and 148) stay exactly the validation set used for beta_rgbd (train.py --nval 288).
"""
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from prep_nyu import decode

DATA = Path(__file__).resolve().parents[1] / "data"
N_NEW = 30


def main():
    files = sorted(f.rfilename for f in HfApi().dataset_info("vikhyatk/nyu_depth_v2").siblings if "/train-" in f.rfilename)
    have = {int(p.name.split("-")[1]) for p in (DATA / "nyu" / "data").glob("train-*.parquet")}
    pool = [i for i in range(1, 137) if i not in have]
    pick = [pool[int(j)] for j in np.linspace(0, len(pool) - 1, N_NEW)]
    print("new shards:", pick, flush=True)

    tmp = DATA / "nyu_tmp"
    rgbs, depths = [], []
    for i in pick:
        p = Path(hf_hub_download("vikhyatk/nyu_depth_v2", files[i], repo_type="dataset", local_dir=tmp))
        r, d = decode([p], 2)
        rgbs.append(r), depths.append(d)
        p.unlink()

    old = np.load(DATA / "nyu_train.npz")
    rgb = np.concatenate(rgbs + [old["rgb"]])
    del rgbs
    depth = np.concatenate(depths + [old["depth"]])
    del depths
    np.savez(DATA / "nyu_train_big.npz", rgb=rgb, depth=depth)
    print("train_big", rgb.shape, f"{len(rgb) - len(old['rgb'])} new frames", flush=True)


if __name__ == "__main__":
    main()
