"""Day 3 on real Sea-thru dives (SfM range, never used to train the touch-up): raw | physics (v1 + Day 1b) |
physics + touch-up (day3a, chosen) | control: touch-up on raw, no physics | classical Sea-thru (comparison, not
ground truth). Frames fixed in advance:
  page 1: the chart frame of each dive (as in fig_seathru_compare.py) + the second D1 chart frame, grey error shown;
  page 2: the first frame of each dive that is not on page 1.
Output: figures/8_review2_day3_touchup/seathru_page_1.png, seathru_page_2.png

  python fig_day3_seathru.py
"""
from pathlib import Path

import numpy as np
import torch

from day3_touchup import load_touchup, run_full, u8
from eval_real import CHARTS, chart_error, predict, t
from eval_synth import load
from fig_results import srgb
from physics import linear_to_srgb, srgb_to_linear
from safe_restore import CURRENT_RECIPE, safe_restore
from vizstyle import INK2, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "8_review2_day3_touchup"
PAGE1 = ["D1/T_S03272", "D1/T_S03425", "D2/T_S03647", "D3/T_S04910", "D4/RGT_0209", "D5/LFT_3374"]
DIVE = {"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4 (turbid)", "D5": "D5 (clear)"}
COLS = ["Raw underwater", "Physics (v1 + Day 1b)", "Physics + touch-up (chosen)",
        "Control: touch-up on raw (no physics)", "Sea-thru classical (comparison)"]


def lin_of(srgb_img):
    return srgb_to_linear(t(srgb_img))[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def frame(net, tu, ctrl, d, cl, names, name):
    i = names.index(name)
    lin = np.clip(d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5), 0, 1)
    z = d[f"gt_{i}"].copy()
    z[~(z > 0.1)] = np.nanmax(z)
    I, zt = t(lin), t(z[..., None])
    p, _ = predict(net, I, zt)
    J, _ = safe_restore(I, zt, p, **CURRENT_RECIPE)
    phys = linear_to_srgb(J)[0].permute(1, 2, 0).cpu().numpy()
    raw = srgb(lin)
    ims = [raw, phys, run_full(tu, u8(phys), u8(raw)), run_full(ctrl, u8(raw), u8(raw)), srgb(cl[f"st_{i}"])]
    err = [chart_error(lin_of(im), CHARTS[name]) for im in ims] if name in CHARTS else None
    return ims, err


def page(rows, net, tu, ctrl, d, cl, names, title, out):
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(3.7 * len(COLS), 2.6 * len(rows)))
    for r, name in enumerate(rows):
        ims, err = frame(net, tu, ctrl, d, cl, names, name)
        for c, im in enumerate(ims):
            ax = axs[r, c]
            no_axes(ax), ax.imshow(np.clip(im, 0, 1))
            if r == 0:
                ax.set_title(COLS[c], fontsize=10)
            if err:
                ax.text(0.02, 0.03, f"grey error {err[c]:.1f}°", transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
        axs[r, 0].set_ylabel(f"{DIVE[name[:2]]}\n{name[3:]}", fontsize=10, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(out)


def main():
    net = load("baseline_v1_60ep_bigdata")[0]
    tu, ctrl = load_touchup("day3a_touchup_global")[0], load_touchup("day3c_control_touchup_on_raw")[0]
    d, cl = np.load(ROOT / "data" / "seathru_small.npz"), np.load(ROOT / "results" / "seathru_classical_real.npz")
    names = [str(n) for n in d["names"]]
    page2 = [next(n for n in names if n.startswith(dv) and n not in PAGE1) for dv in DIVE]
    page(PAGE1, net, tu, ctrl, d, cl, names, "Real Sea-thru dives (never seen by the touch-up): chart frame of each dive. "
         "Grey error = how far the grey chart is from neutral, lower = better", FIG / "seathru_page_1.png")
    page(page2, net, tu, ctrl, d, cl, names, "Real Sea-thru dives: first other frame of each dive (fixed in advance). "
         "No ground truth exists for these", FIG / "seathru_page_2.png")


if __name__ == "__main__":
    main()
