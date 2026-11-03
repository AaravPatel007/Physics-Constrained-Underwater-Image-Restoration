"""LSUI (Peng et al., U-shape Transformer, 2023; 4279 underwater photos + chosen references) for the Day 3 touch-up.

Reads data/lsui/LSUI.zip directly (no extraction). Each pair is resized like UIEB (long side 400, multiple of 8).
  overlap : finds LSUI photos that are also in UIEB (correlation of 48x48 grey thumbnails, so colour casts do not
            count) and drops every match. Checked by eye: true duplicates score >= 0.96, look-alikes <= 0.95; the
            closest LSUI photo to any UIEB val photo scores 0.80 (no val overlap).
  build   : DAv2 depth -> baseline_v1_60ep_bigdata + Day 1b physics -> stores raw / reference / physics output
            (uint8, compressed) in data/lsui/lsui_day1b.npz; depth is not kept.

  python prep_lsui.py
"""
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import depth as dav2
from day3_touchup import load_split, restore_srgb, u8
from eval_synth import load

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "lsui" / "LSUI.zip"
OUT = ROOT / "data" / "lsui" / "lsui_day1b.npz"
REPORT = ROOT / "results" / "review2_day3_touchup" / "lsui_overlap.json"
MATCH = 0.955  # grey-thumbnail correlation above which two photos count as the same photo


def resize(a):
    s = 400 / max(a.shape[:2])
    size = (int(a.shape[1] * s) // 8 * 8, int(a.shape[0] * s) // 8 * 8)
    return cv2.resize(a, size, interpolation=cv2.INTER_AREA)


def thumb(a):
    x = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def read(z, name):
    return np.array(Image.open(io.BytesIO(z.read(name))).convert("RGB"))


def main():
    z = zipfile.ZipFile(ZIP)
    ids = sorted(int(Path(n).stem) for n in z.namelist() if n.startswith("LSUI/input/") and n.endswith(".jpg"))
    c, _, _ = load_split()
    uieb = {f"val_{i}": thumb(c[f"va_raw_{i}"]) for i in range(90)}
    uieb.update({f"train_{i}": thumb(c[f"tr_raw_{i}"]) for i in range(int(c["n_tr"]))})
    keys = list(uieb)
    U = np.stack([uieb[k] for k in keys])
    net = load("baseline_v1_60ep_bigdata")[0]
    out, report = {}, {"n_lsui": len(ids), "threshold": MATCH, "matches": {}, "dropped": []}
    kept = 0
    for b in range(0, len(ids), 100):
        chunk = []
        for k in ids[b:b + 100]:
            raw, ref = resize(read(z, f"LSUI/input/{k}.jpg")), read(z, f"LSUI/GT/{k}.jpg")
            ref = cv2.resize(ref, raw.shape[1::-1], interpolation=cv2.INTER_AREA)
            s = U @ thumb(raw)
            j = int(s.argmax())
            report["closest_val"] = max(report.get("closest_val", 0.0), float(s[:90].max()))
            if s[j] > MATCH:
                report["matches"][str(k)] = [keys[j], float(s[j])]
                report["dropped"].append(k)
                continue
            chunk.append((k, raw, ref))
        depths = dav2.predict([r for _, r, _ in chunk], "metric-indoor", bs=1)
        with torch.no_grad():
            for (k, raw, ref), zz in zip(chunk, depths):
                out[f"raw_{kept}"], out[f"ref_{kept}"] = raw, ref
                out[f"d1b_{kept}"] = u8(restore_srgb(net, raw, zz))
                out[f"id_{kept}"] = np.int32(k)
                kept += 1
        print(f"{b + len(ids[b:b + 100])}/{len(ids)} kept {kept}", flush=True)
    np.savez_compressed(OUT, n=kept, **out)
    report["n_kept"] = kept
    report["n_matching_uieb"] = len(report["matches"])
    json.dump(report, open(REPORT, "w"), indent=1)
    print(OUT, OUT.stat().st_size / 1e9, "GB;", report["n_matching_uieb"], "match UIEB,", len(report["dropped"]), "dropped")


if __name__ == "__main__":
    main()
