"""The two architecture diagrams (SVG + 300-dpi PNG), drawn with matplotlib so they are reproducible.

  diagram1_betanet_vs_v1   BetaNet (12 outputs) vs SunDepthBetaNet v1 (10 outputs)
  diagram2_betanet_vs_v2   BetaNet (12 outputs) vs SunDepthBetaNet v2 (16 outputs)

  python sunlight_depth_v2/figures/architecture/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "svg.fonttype": "none"})

INK, INK2 = "#1f2933", "#52606d"
SHARED_FC, SHARED_EC = "#eef2f7", "#52606d"     # unchanged components
NEW_FC, NEW_EC = "#fde6d8", "#d9622b"           # changed / new learned outputs
PHYS_FC, PHYS_EC = "#ffffff", "#d9622b"         # physics computation (no learned weights)
OLD_FC, OLD_EC = "#e3ecf9", "#2a6fd6"           # the directly predicted E of BetaNet

# column layout (x, width) in diagram units
X_IN, W_IN = 0.0, 12.0
X_BB, W_BB = 15.0, 12.5
X_HD, W_HD = 30.5, 13.0
X_OUT, W_OUT = 47.0, 12.0
X_PH, W_PH = 64.0, 20.0
X_RS, W_RS = 89.0, 15.5
X_J, W_J = 108.0, 11.0
FIG_W = 13.3  # inches; the height follows from the drawing so that 1 unit is the same in x and y
BOX_H, GAP = 2.3, 0.55


def box(ax, x, y, w, h, text, sub=None, fc=SHARED_FC, ec=SHARED_EC, ls="-", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc=fc, ec=ec, lw=1.3, ls=ls))
    cy = y + h / 2 + (0.55 if sub else 0)
    ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.8, sub, ha="center", va="center", fontsize=8.5, color=INK2)


def arrow(ax, p, q, color=INK2, ls="-", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def row(ax, y0, title, outputs, physics=None, physics_inputs=(), kd_from_bd=False):
    """outputs: list of (label, n, kind) with kind in {'shared', 'new', 'old'}; the row is centred on y0."""
    n = len(outputs)
    h_out = n * BOX_H + (n - 1) * GAP
    top = y0 + h_out / 2
    total = sum(k for _, k, _ in outputs)
    ax.text(X_IN, top + 2.0, title, ha="left", va="bottom", fontsize=12, weight="bold", color=INK)
    box(ax, X_IN, y0 - 2.4, W_IN, 4.8, "RGB-D input", r"image $I$, range $z$")
    box(ax, X_BB, y0 - 2.4, W_BB, 4.8, "ResNet-18", "4-channel stem")
    box(ax, X_HD, y0 - 2.4, W_HD, 4.8, "MLP head", rf"512 $\rightarrow$ 256 $\rightarrow$ {total}",
        fc=NEW_FC if total != 12 else SHARED_FC, ec=NEW_EC if total != 12 else SHARED_EC)
    arrow(ax, (X_IN + W_IN, y0), (X_BB, y0))
    arrow(ax, (X_BB + W_BB, y0), (X_HD, y0))
    ys = {}
    for i, (lab, k, kind) in enumerate(outputs):
        y = top - BOX_H - i * (BOX_H + GAP)
        fc, ec = {"shared": (SHARED_FC, SHARED_EC), "new": (NEW_FC, NEW_EC), "old": (OLD_FC, OLD_EC)}[kind]
        box(ax, X_OUT, y, W_OUT, BOX_H, rf"{lab}  ({k})", fc=fc, ec=ec, fs=10.5)
        ys[lab] = y + BOX_H / 2
        arrow(ax, (X_HD + W_HD, y0), (X_OUT, ys[lab]), lw=0.9)
    # restoration
    yr = y0
    box(ax, X_RS, yr - 2.4, W_RS, 4.8, "Restoration", "closed-form inversion", bold=False)
    box(ax, X_J, yr - 2.4, W_J, 4.8, r"$\hat{J}$", "restored image", fs=13)
    arrow(ax, (X_RS + W_RS, yr), (X_J, yr))
    # water coefficients -> restoration (one bracket for the three)
    water = [ys[k] for k in ys if k.startswith((r"$\beta_D$", r"$\beta_B$", r"$B_\infty$"))]
    xb = X_OUT + W_OUT + 1.0
    ax.plot([X_OUT + W_OUT, xb], [water[0]] * 2, color=INK2, lw=0.9)
    ax.plot([X_OUT + W_OUT, xb], [water[-1]] * 2, color=INK2, lw=0.9)
    ax.plot([X_OUT + W_OUT, xb], [water[1]] * 2, color=INK2, lw=0.9)
    ax.plot([xb, xb], [water[-1], water[0]], color=INK2, lw=0.9)
    ym = (water[0] + water[-1]) / 2
    ax.plot([xb, X_PH - 1.2], [ym, ym], color=INK2, lw=0.9)
    arrow(ax, (X_PH - 1.2, ym), (X_RS, yr + 1.2), lw=0.9)
    if physics is None:  # E straight to the restoration
        arrow(ax, (X_OUT + W_OUT, ys[r"$E$"]), (X_RS, yr - 1.2), color=OLD_EC, lw=1.3)
        return top - h_out
    yp = sum(ys[k] for k in physics_inputs) / len(physics_inputs)
    hp = max(3.6, 1.4 + 1.9 * len(physics))
    box(ax, X_PH, yp - hp / 2, W_PH, hp, "", fc=PHYS_FC, ec=PHYS_EC, ls=(0, (4, 2)))
    for j, line in enumerate(physics):
        ax.text(X_PH + W_PH / 2, yp + (len(physics) - 1) * 0.95 - j * 1.9, line, ha="center", va="center",
                fontsize=10.5, color=INK)
    for k in physics_inputs:
        arrow(ax, (X_OUT + W_OUT, ys[k]), (X_PH, yp + (ys[k] - yp) * 0.5), color=NEW_EC, lw=1.2)
    if kd_from_bd:  # v1: K_d is computed from the predicted beta_D
        arrow(ax, (X_OUT + W_OUT + 1.0, ys[r"$\beta_D$"]), (X_PH + 2.0, yp + hp / 2), color=NEW_EC, lw=1.0,
              ls=(0, (3, 2)), rad=-0.25)
    arrow(ax, (X_PH + W_PH, yp), (X_RS, yr - 1.2), color=NEW_EC, lw=1.3)
    ax.text((X_PH + W_PH + X_RS) / 2 + 0.3, (yp + yr - 1.2) / 2 + 0.55, r"$E$", fontsize=10.5, color=NEW_EC,
            ha="center", va="bottom")
    return min(top - h_out, yp - hp / 2)


def legend(ax, y):
    items = [(SHARED_FC, SHARED_EC, "-", "unchanged"), (OLD_FC, OLD_EC, "-", "sunlight predicted directly"),
             (NEW_FC, NEW_EC, "-", "new or changed output"), (PHYS_FC, PHYS_EC, (0, (4, 2)), "physics (no learned weights)")]
    x = 0.0
    for fc, ec, ls, lab in items:
        ax.add_patch(FancyBboxPatch((x, y - 0.6), 2.4, 1.2, boxstyle="round,pad=0,rounding_size=0.3", fc=fc, ec=ec,
                                    lw=1.2, ls=ls))
        ax.text(x + 3.0, y, lab, va="center", fontsize=9.5, color=INK2)
        x += 3.0 + 0.62 * len(lab) + 2.5


def draw(name, bottom_title, bottom_outputs, physics, physics_inputs, kd_from_bd, height):
    fig, ax = plt.subplots(figsize=(FIG_W, 6))
    base = [(r"$\beta_D$", 3, "shared"), (r"$\beta_B$", 3, "shared"), (r"$B_\infty$", 3, "shared"), (r"$E$", 3, "old")]
    nb = len(bottom_outputs)
    yb = 0.0
    yt = yb + (nb * BOX_H + (nb - 1) * GAP) / 2 + 4.0 + 4 * BOX_H / 2 + 3 * GAP / 2 + 3.2
    row(ax, yt, "Original BetaNet  (12 outputs)", base)
    low = row(ax, yb, bottom_title, bottom_outputs, physics, physics_inputs, kd_from_bd)
    ylo = low - 2.0
    legend(ax, ylo)
    x0, x1 = -0.5, X_J + W_J + 0.5
    y0, y1 = ylo - 1.2, yt + 4 * BOX_H / 2 + 2 * GAP + 3.8
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_position([0, 0, 1, 1])
    fig.set_size_inches(FIG_W, FIG_W * (y1 - y0) / (x1 - x0))
    ax.axis("off")
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def main():
    draw("diagram1_betanet_vs_v1", "SunDepthBetaNet v1  (10 outputs)",
         [(r"$\beta_D$", 3, "shared"), (r"$\beta_B$", 3, "shared"), (r"$B_\infty$", 3, "shared"), (r"$d_s$", 1, "new")],
         [r"$E_0 = [1, 1, 1]$  (fixed)", r"$K_d = 0.8\,\beta_D$", r"$E = E_0\, e^{-K_d\, d_s}$"], [r"$d_s$"], True, None)
    draw("diagram2_betanet_vs_v2", "SunDepthBetaNet v2  (16 outputs)",
         [(r"$\beta_D$", 3, "shared"), (r"$\beta_B$", 3, "shared"), (r"$B_\infty$", 3, "shared"),
          (r"$E_0$", 3, "new"), (r"$K_d$", 3, "new"), (r"$d_s$", 1, "new")],
         [r"$E = E_0 \odot e^{-K_d\, d_s}$"], [r"$E_0$", r"$K_d$", r"$d_s$"], False, None)


if __name__ == "__main__":
    main()
