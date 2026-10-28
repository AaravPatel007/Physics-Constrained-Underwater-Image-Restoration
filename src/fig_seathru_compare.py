"""Real Sea-thru frames: original BetaNet (beta_rgbd) vs longer run (beta_rgbd_big), both + gray-world, SfM range.
Left: the paper's chart frame of each dive D1-D5 (fixed before looking at results), with classical Sea-thru for comparison.
Right: predicted beta_D (red) for all 24 frames, grouped by dive (no ground truth exists for real water).
Reads results/real_seathru_outputs{,_big}.npz and results/real_eval{,_big}.json (written by eval_real.py)."""
import json
from pathlib import Path

import numpy as np

from fig_results import srgb
from vizstyle import INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FRAMES = ["D1/T_S03272", "D2/T_S03647", "D3/T_S04910", "D4/RGT_0209", "D5/LFT_3374"]
DIVE = {"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4 (turbid)", "D5": "D5 (clear)"}
M = "BetaNet + gray-world [SfM range]"


def main():
    old, new = np.load(RES / "real_seathru_outputs.npz"), np.load(RES / "real_seathru_outputs_big.npz")
    eo, en = json.load(open(RES / "real_eval.json"))["seathru"], json.load(open(RES / "real_eval_big.json"))["seathru"]
    cols = [("Raw underwater", old, "raw", None), ("BetaNet, original + gray-world", old, M, eo),
            ("BetaNet, longer run + gray-world", new, M, en),
            ("Sea-thru classical (comparison, not ground truth)", old, "Sea-thru (classical, SfM range)", eo)]

    n = len(FRAMES)
    fig = plt.figure(figsize=(20, 2.9 * n))
    gs = fig.add_gridspec(n, 6, width_ratios=[1, 1, 1, 1, 0.12, 1.6])
    for i, fr in enumerate(FRAMES):
        for c, (title, src, key, ev) in enumerate(cols):
            ax = fig.add_subplot(gs[i, c])
            no_axes(ax)
            ax.imshow(srgb(src[f"{fr}|{key}"]))
            if i == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{DIVE[fr[:2]]}\n{fr[3:]}", fontsize=10, color=INK2)
            note = []
            if ev is not None and key.startswith("BetaNet"):
                note.append(f"β_D red {ev['frames'][fr]['SfM range']['beta_D'][0]:.2f} /m")
            if fr in eo["chart"]:
                note.append(f"grey error {(ev or eo)['chart'][fr][key]:.1f}°")
            if note:
                ax.text(0.02, 0.03, " · ".join(note), transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))

    ax = fig.add_subplot(gs[:, 5])
    dives = sorted(DIVE)
    for j, (ev, lab) in enumerate([(eo, "BetaNet, original"), (en, "BetaNet, longer run")]):
        for d_i, dv in enumerate(dives):
            v = [f["SfM range"]["beta_D"][0] for name, f in ev["frames"].items() if name.startswith(dv)]
            x = d_i + (j - 0.5) * 0.32
            ax.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v, s=30, color=SERIES[j],
                       edgecolor="white", lw=1, zorder=3, label=lab if d_i == 0 else None)
            ax.plot([x - 0.13, x + 0.13], [np.mean(v)] * 2, color=SERIES[j], lw=2.5)
    ax.set_xticks(range(len(dives)), [DIVE[d] for d in dives], fontsize=9)
    ax.set_ylabel("predicted β_D, red channel (1/m)")
    ax.set_title("Water read-out per dive (dots = frames, bar = mean)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.02, 0.02, "No ground-truth β exists for real dives.\nExpected: turbid D4 highest, clear D5 lowest.",
            transform=ax.transAxes, fontsize=9, color=MUTED)

    fig.suptitle("Real Sea-thru dives (SfM range): original vs longer-trained BetaNet", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = ROOT / "figures" / "3_new_runs" / "seathru_original_vs_longer.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
