"""Download large PyPI wheels in parallel byte ranges (standard library + curl), checking each sha256.

Why: on this machine's network one connection to files.pythonhosted.org gives ~0.3 MB/s (uv / pip), while 8
parallel range requests give ~6.5 MB/s. The CUDA wheels torch 2.11+cu128 needs are ~2.9 GB.
Install afterwards with:  uv pip install --find-links <dest> torch==2.11.0+cu128 ...

  python3 fetch_wheels.py <dest_dir> pkg==ver [pkg==ver ...]
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def pick(pkg, ver):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60) as r:
        files = json.load(r)["urls"]
    ok = [f for f in files if f["filename"].endswith(".whl") and "x86_64" in f["filename"] and "manylinux" in f["filename"]
          and ("py3-none" in f["filename"] or "cp312-cp312" in f["filename"])]
    return sorted(ok, key=lambda f: f["filename"])[0]


def get_range(url, out, a, b):
    for _ in range(20):
        if out.exists() and out.stat().st_size == b - a + 1:
            return
        r = subprocess.run(["curl", "-sS", "--retry", "5", "--retry-all-errors", "-r", f"{a}-{b}", "-o", str(out), url])
        if r.returncode == 0 and out.stat().st_size == b - a + 1:
            return
    raise RuntimeError(f"range {a}-{b} of {url}")


def fetch(spec, dest, parts=8):
    pkg, ver = spec.split("==")
    f = pick(pkg, ver)
    out = dest / f["filename"]
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == f["digests"]["sha256"]:
        return out
    n, step = f["size"], -(-f["size"] // parts)
    chunks = [(dest / f"{f['filename']}.{i}", i * step, min(n, (i + 1) * step) - 1) for i in range(parts)]
    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(lambda c: get_range(f["url"], *c), chunks))
    with open(out, "wb") as o:
        for c, _, _ in chunks:
            o.write(c.read_bytes())
            c.unlink()
    assert hashlib.sha256(out.read_bytes()).hexdigest() == f["digests"]["sha256"], f"sha256 mismatch {out.name}"
    print("ok", out.name, f"{n / 1e6:.0f} MB", flush=True)
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:  # one wheel at a time, 8 ranges each
        fetch(spec, dest)
    print("WHEELS_DONE", flush=True)
