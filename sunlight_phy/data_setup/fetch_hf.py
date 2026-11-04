"""Parallel, resumable download of Hugging Face dataset files with curl (standard library only).

Why: on this machine's Wi-Fi one HF connection gives ~0.5 MB/s and hf_hub_download stalled; NYU shards are ~220 MB
each. curl -C - resumes after drops, and several connections in parallel multiply the throughput.

  python3 fetch_hf.py nyu-keep      # the 19 train + 3 test shards baseline_v1 was built from -> data/nyu/data/
  (prep_nyu_more_local.py uses fetch() for the 30 extra shards)
"""
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
NYU = "vikhyatk/nyu_depth_v2"
NYU_KEEP = [0, 5, 16, 26, 27, 38, 49, 52, 60, 71, 78, 82, 93, 104, 115, 126, 130, 137, 148]  # results/logs/dl_nyu*.log


def list_files(repo, path=""):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        return sorted(e["path"] for e in json.load(r) if e["type"] == "file")


def fetch_one(repo, f, dest):
    out = Path(dest) / f
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    if out.exists():
        return out
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"
    for _ in range(30):
        r = subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "10", "--retry-all-errors", "--connect-timeout", "30",
                            "--speed-limit", "20000", "--speed-time", "60", "-o", str(part), url])
        if r.returncode in (0, 33):  # 33 = range not satisfiable = already complete
            part.rename(out)
            print("ok", out.name, f"{out.stat().st_size / 1e6:.0f} MB", flush=True)
            return out
    raise RuntimeError(f"failed: {f}")


def fetch(repo, files, dest, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda f: fetch_one(repo, f, dest), files))


def nyu_train_files():
    files = list_files(NYU, "data")
    train = sorted(f for f in files if "/train-" in f)
    val = sorted(f for f in files if "/validation-" in f)
    assert len(train) == 157 and len(val) == 3, (len(train), len(val))
    return train, val


if __name__ == "__main__":
    if sys.argv[1] == "nyu-keep":
        train, val = nyu_train_files()
        fetch(NYU, [train[i] for i in NYU_KEEP] + val, DATA / "nyu")
        print("NYU_KEEP_DONE", flush=True)
