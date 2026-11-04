"""src/prep_nyu_more.py with a faster download: same 30 shards (same deterministic pick), same decoding (prep_nyu.decode,
every 2nd frame), same order (new frames first, the old nyu_train.npz frames last, so the last 288 frames are the
same validation set). Only difference: shards come through fetch_hf.fetch (parallel curl, 6 at a time) and each
batch is decoded and deleted before the next, to fit the disk.

Expected result (results/logs/beta_rgbd_big.log): 7 140 train + 288 val = 7 428 frames.

  cd src && python ../sunlight_phy/data_setup/prep_nyu_more_local.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from fetch_hf import DATA, NYU, fetch, nyu_train_files  # noqa: E402
from prep_nyu import decode  # noqa: E402

N_NEW = 30


def main():
    train, _ = nyu_train_files()
    have = {int(p.name.split("-")[1]) for p in (DATA / "nyu" / "data").glob("train-*.parquet")}
    pool = [i for i in range(1, 137) if i not in have]
    pick = [pool[int(j)] for j in np.linspace(0, len(pool) - 1, N_NEW)]
    print("new shards:", pick, flush=True)
    tmp = DATA / "nyu_tmp"
    rgbs, depths = [], []
    for b in range(0, N_NEW, 6):
        paths = fetch(NYU, [train[i] for i in pick[b:b + 6]], tmp)
        for p in paths:  # in pick order, like the original loop
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
