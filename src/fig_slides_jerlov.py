"""Per-Jerlov-type PSNR of every method (fig8 style), merging the original (synth_eval.json) and longer-run
(synth_eval_big.json, PSNR only) evaluations: same 654 rooms, same waters.
-> figures/4_presentation_slides/jerlov_all_methods.png. Run after eval_synth.py and eval_synth_big.py."""
import json
from pathlib import Path

import numpy as np

from physics import JERLOV_N
from vizstyle import AXIS, INK2, SERIES, plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "4_presentation_slides" / "jerlov_all_methods.png"

S = json.load(open(ROOT / "results" / "synth_eval.json"))["jerlov"]
B = json.load(open(ROOT / "results" / "synth_eval_big.json"))["jerlov"]
types = list(JERLOV_N)
assert all(abs(S[t]["raw"]["psnr"] - B[t]["raw"]) < 1e-9 for t in types), "test sets differ"


def psnr(src, t, m):
    v = src[t][m]
    return v["psnr"] if isinstance(v, dict) else v


rows = [(S, "raw", "#c3c2b7", "o", "-", "Raw underwater"),
        (S, "gray-world", SERIES[2], "^", "-", "Gray-world"),
        (S, "unet_rgbd", SERIES[1], "s", "-", "UNet black box"),
        (S, "beta_rgbd", "#86b6ef", "D", "-", "BetaNet, original"),
        (B, "beta_rgbd_big", SERIES[0], "D", "-", "BetaNet, longer run"),
        (S, "oracle (true coefs)", INK2, None, "--", "Oracle (true coefficients)")]

x = np.arange(len(types))
fig, ax = plt.subplots(figsize=(12, 4.8))
for src, m, c, mk, ls, lab in rows:
    ax.plot(x, [psnr(src, t, m) for t in types], color=c, marker=mk, markersize=8, ls=ls, lw=2, label=lab)
ax.set_xticks(x, ["I", "IA", "IB", "II", "III", "1", "3", "5", "7", "9"])
ax.set_xlim(-0.3, 9.3)
ax.axvline(4.5, color=AXIS, lw=1)
ax.text(2, ax.get_ylim()[1], "open ocean", ha="center", va="bottom", color=INK2)
ax.text(7, ax.get_ylim()[1], "coastal", ha="center", va="bottom", color=INK2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=11)
ax.set_ylabel("PSNR (dB)")
ax.set_xlabel("Jerlov water type (clear → murky),  200 test rooms each")
fig.savefig(OUT)
print(OUT)
