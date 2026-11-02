"""Slide-shaped figures and LaTeX equations for the formulation slides (why, physics, data, depth).

The Part 1 figures are wide strips for documents; these re-cut the same content into ~3:2 figures that
fit next to text on a 16:9 slide, using different scenes from the Part 1 figures.
Notation follows Akkaynak & Treibitz: beta^D_c (attenuation), beta^B_c (backscatter), c = R, G, B.
Output: figures/4_presentation_slides/ (+ equations/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from physics import JERLOV_BETA, add_water, linear_to_srgb, sample_water, srgb_to_linear
from vizstyle import CH, CH_MARK, INK, INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "4_presentation_slides"
RES = ROOT / "results"
FORMATION_ROOM, JERLOV_ROOM = 126, 434  # NYU test: colourful classroom, red bunk bed
CORAL = "#c8553d"


def scene(i, scale=1.0):
    d = np.load(ROOT / "data" / "nyu_test.npz")
    J = srgb_to_linear(torch.from_numpy(d["rgb"][i]).permute(2, 0, 1)[None].float() / 255)
    z = torch.from_numpy(d["depth"][i].astype(np.float32))[None, None] * scale
    return J, z


def show(x):
    return linear_to_srgb(x)[0].permute(1, 2, 0).numpy().clip(0, 1)


def to_display(lin):
    return linear_to_srgb(torch.from_numpy(np.asarray(lin, np.float32))).numpy().clip(0, 1)


def crop32(im):
    """Centre-crop to 3:2 so photos from different datasets tile evenly."""
    h, w = im.shape[:2]
    if w / h > 1.5:
        c = int(h * 1.5)
        return im[:, (w - c) // 2:(w - c) // 2 + c]
    c = int(w / 1.5)
    return im[(h - c) // 2:(h - c) // 2 + c]


def grid(panels, rows, cols, size, name, cmap_label=None, vmax=None):
    fig, axs = plt.subplots(rows, cols, figsize=size)
    h = None
    for ax, (im, title, sub) in zip(axs.ravel(), panels):
        if im.ndim == 2:
            h = ax.imshow(im, cmap="Blues", vmin=0, vmax=vmax)
        else:
            ax.imshow(im)
        no_axes(ax)
        if title:
            ax.set_title(title, fontsize=15)
        if sub:
            ax.text(0.0, -0.04, sub, transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=13)
    has_sub = any(p[2] for p in panels)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.05, hspace=0.32 if has_sub else 0.18)
    if cmap_label:
        cb = fig.colorbar(h, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.set_label(cmap_label, color=INK2, fontsize=13)
        cb.ax.tick_params(colors=MUTED, labelsize=12)
        cb.outline.set_visible(False)
    fig.savefig(OUT / name)
    plt.close(fig)


def raw_photos():
    """3 x 2 real raw photos from different sites: each water gives its own colour cast."""
    u = np.load(RES / "uieb_outputs.npz")  # stored in display (sRGB) space
    s = np.load(RES / "real_seathru_outputs.npz")  # stored linear
    ims = [u["0|raw"].astype(np.float32), u["5|raw"].astype(np.float32),
           to_display(s["D5/LFT_3384|raw"]), u["6|raw"].astype(np.float32),
           to_display(s["D1/T_S03272|raw"]), u["4|raw"].astype(np.float32)]
    grid([(crop32(np.clip(im, 0, 1)), "", "") for im in ims], 2, 3, (9.4, 4.4), "raw_underwater_photos.png")


def red_light_curve():
    r = np.linspace(0, 15, 200)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for c, b, name in zip("RGB", JERLOV_BETA["I"], ["Red", "Green", "Blue"]):
        ax.plot(r, np.exp(-b * r) * 100, color=CH[c], marker=CH_MARK[c], markevery=25, markersize=7)
        ax.annotate(name, (15, np.exp(-b * 15) * 100), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=13, color=INK2)
    red5 = np.exp(-JERLOV_BETA["I"][0] * 5) * 100
    ax.plot([5, 5], [0, red5], color=MUTED, lw=1, ls="--")
    ax.annotate(f"{red5:.0f} % of red\nleft at 5 m", (5, red5), xytext=(12, 10), textcoords="offset points",
                fontsize=13, color=CH["R"], fontweight="bold")
    ax.set_title("Light left after z metres, clearest ocean", fontsize=14)
    ax.set_xlabel("distance through water (m)", fontsize=13)
    ax.set_ylabel("% surviving", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 105)
    fig.savefig(OUT / "red_light_curve.png")
    plt.close(fig)


def image_formation():
    """2 x 3: what goes in (J, J*E, z) on top, what comes out (direct, backscatter, photo) below."""
    J, z = scene(FORMATION_ROOM, 0.8)
    w = sample_water(1, water_type="3C")
    w["beta_B"] = w["beta_B"] * torch.tensor([[1.3, 0.8, 0.7]])
    w["B_inf"] = torch.tensor([[0.05, 0.32, 0.38]])
    w["E"] = torch.exp(-w["beta_D"] * 4.0)
    w["E"] = w["E"] / w["E"].amax(1, keepdim=True)
    I, D, B = add_water(J, z, w)
    zmax = float(z.max())
    panels = [(show(J), r"Clean scene  $J$", "what we want back"),
              (show(J * w["E"][:, :, None, None]), r"Sunlit at depth  $J\cdot E$", "red lost on the way down"),
              (z[0, 0].numpy(), r"Range  $z$", f"0 – {zmax:.0f} m"),
              (show(D), "Direct signal", r"fades with $z$  ($\beta^D$)"),
              (show(B * 1.6), "Backscatter (×1.6)", r"haze grows with $z$  ($\beta^B$)"),
              (show(I), r"Underwater photo  $I$", "= direct + backscatter")]
    grid(panels, 2, 3, (10, 6.9), "image_formation_grid.png")


def jerlov_waters():
    """2 x 3: one room in air and in five Jerlov waters, clear ocean to murky coast."""
    J, z = scene(JERLOV_ROOM, 1.0)
    names = [("I", "Ocean I (clearest)"), ("III", "Ocean III"), ("3C", "Coastal 3"),
             ("5C", "Coastal 5"), ("9C", "Coastal 9 (murkiest)")]
    panels = [(show(J), "In air (NYU room)", "+ Kinect depth")]
    gen = torch.Generator().manual_seed(3)
    for t, label in names:
        w = sample_water(1, gen=gen, water_type=t)
        I, _, _ = add_water(J, z, w)
        b = JERLOV_BETA[t]
        panels.append((show(I), label, rf"$\beta^D$  R {b[0]:.2f}  G {b[1]:.2f}  B {b[2]:.2f} /m"))
    grid(panels, 2, 3, (10, 6.9), "jerlov_six_waters.png")


def sunlight_fix():
    """2 x 3 real dives: raw, v1 (no sunlight colour E), v2 (with E). Both BetaNet columns are the
    network's own output (no gray-world), so the only difference is E."""
    o2 = np.load(RES / "real_seathru_outputs.npz")
    o1 = np.load(RES / "v1_no_illuminant" / "real_seathru_outputs.npz")
    panels = []
    for f in ["D1/T_S02951", "D4/RGT_0229"]:
        first = not panels
        panels += [(to_display(o2[f"{f}|raw"]), "Raw photo" if first else "", ""),
                   (to_display(o1[f"{f}|BetaNet [SfM range]"]), "v1: without E" if first else "", ""),
                   (to_display(o2[f"{f}|BetaNet [SfM range]"]), "v2: with E" if first else "", "")]
    grid(panels, 2, 3, (10, 4.75), "sunlight_colour_fix.png")


def depth_maps():
    """2 x 2 on one colour scale: photo, SfM ground truth, DAv2 metric-indoor and -outdoor as is."""
    d = np.load(ROOT / "data" / "seathru_small.npz")
    i = list(d["names"]).index("D5/LFT_3384")
    gt = d[f"gt_{i}"].copy()
    gt[gt <= 0.1] = np.nan
    panels = [(to_display(d[f"lin_{i}"]), "Real dive (Sea-thru D5)", ""),
              (gt, "True range (SfM)", ""),
              (d[f"dav2_metric-indoor_{i}"], "DAv2 indoor model", ""),
              (d[f"dav2_metric-outdoor_{i}"], "DAv2 outdoor model", "")]
    grid(panels, 2, 2, (8, 5.6), "depth_maps_dav2.png", cmap_label="metres", vmax=np.nanpercentile(gt, 99))


def depth_scale_per_frame():
    """Per-frame scale error of DAv2 metric depth: median(predicted / true) on each of the 24 frames."""
    d = np.load(ROOT / "data" / "seathru_small.npz")
    ev = json.load(open(RES / "depth_eval.json"))["summary"]
    rows = [("metric-indoor", "Indoor model", SERIES[0]), ("metric-outdoor", "Outdoor model", CORAL)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for k, (m, label, col) in enumerate(rows):
        r = []
        for i in range(len(d["names"])):
            g, p = d[f"gt_{i}"], d[f"dav2_{m}_{i}"]
            v = g > 0.1
            r.append(np.median(p[v] / g[v]))
        y = k + rng.uniform(-0.12, 0.12, len(r))
        ax.scatter(r, y, s=46, color=col, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(9.5, k, f"AbsRel {ev[m]['absrel']:.2f}", va="center", fontsize=13, color=col, fontweight="bold")
    ax.axvline(1, color=INK, lw=1.2, zorder=2)
    ax.text(1.04, 1.45, "correct scale", fontsize=12, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["¼×", "½×", "1×", "2×", "4×", "8×"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.2, 9)
    ax.set_yticks([0, 1], [r[1] for r in rows])
    ax.set_ylim(-0.5, 1.7)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("predicted depth ÷ true depth, one dot per real frame (24)", fontsize=12)
    fig.savefig(OUT / "depth_scale_per_frame.png")
    plt.close(fig)


STEP_FRAME = "D4/RGT_0191"  # real Red Sea frame with SfM range and a coloured slate; v2 restores it to neutral


def real_frame(name):
    """The exact inputs eval_real.py used: photo scaled by its 99.5th percentile, holes in z set to max."""
    d = np.load(ROOT / "data" / "seathru_small.npz")
    i = list(d["names"]).index(name)
    lin = np.clip(d[f"lin_{i}"] / np.percentile(d[f"lin_{i}"], 99.5), 0, 1)
    z = d[f"gt_{i}"].copy()
    z[~(z > 0.1)] = np.nanmax(z)
    p = json.load(open(RES / "real_eval.json"))["seathru"]["frames"][name]["SfM range"]
    return lin, z, {k: np.array(v) for k, v in p.items()}


def restoration_steps():
    """2 x 3 on a real photo, using BetaNet's own 12 numbers for it: undo the model one term at a time."""
    I, z, p = real_frame(STEP_FRAME)
    zz = z[..., None]
    B = p["B_inf"] * (1 - np.exp(-p["beta_B"] * zz))
    dehazed = np.clip(I - B, 0, None)
    unfaded = dehazed * np.exp(np.minimum(p["beta_D"] * zz, np.log(50.0)))
    J = unfaded / p["E"]
    s = 1 / np.percentile(J, 99)  # one display exposure for the last two steps
    panels = [(to_display(I), r"1. Raw photo  $I$", "real Red Sea dive (Sea-thru D4)"),
              (z, r"2. Range  $z$  (SfM)", f"{np.nanmin(z):.1f} – {np.nanmax(z):.1f} m"),
              (to_display(B), r"3. Predicted haze", r"$B^\infty(1-e^{-\beta^B z})$"),
              (to_display(dehazed), "4. Haze removed", r"$I - $ haze"),
              (to_display(unfaded * s), "5. Fading undone", r"$\times\, e^{+\beta^D z}$"),
              (to_display(J * s), r"6. Sunlight colour removed", r"$\div\, E$   $=\hat{J}$")]
    grid(panels, 2, 3, (10, 5.7), "real_restoration_steps.png")


def architecture():
    """Inputs (4 real channels) -> ResNet-18 stages -> head -> 12 outputs, with tensor sizes."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    I, z, _ = real_frame(STEP_FRAME)
    I = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(I).permute(2, 0, 1)[None], (240, 360),
                                                   mode="area")[0].permute(1, 2, 0))
    z = np.asarray(torch.nn.functional.interpolate(torch.from_numpy(z)[None, None], (240, 360))[0, 0])
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    ax.set_xlim(0, 13.5), ax.set_ylim(-0.1, 3.9)
    no_axes(ax)
    tint = {"R": [1, 0.25, 0.25], "G": [0.25, 0.75, 0.25], "B": [0.25, 0.45, 1]}
    chans = [(np.sqrt(I[..., k])[..., None] * np.array(tint[c])) for k, c in enumerate("RGB")]
    zc = plt.get_cmap("Blues")(np.clip(z / np.nanmax(z), 0, 1))[..., :3]
    for k, im in enumerate([zc] + chans[::-1]):  # back to front
        x0, y0 = 0.1 + 0.22 * (3 - k), 0.85 + 0.22 * (3 - k)
        ax.imshow(np.clip(im, 0, 1), extent=(x0, x0 + 1.9, y0, y0 + 1.27), zorder=k + 1)
        ax.add_patch(Rectangle((x0, y0), 1.9, 1.27, fill=False, ec="white", lw=1.2, zorder=k + 1))
    ax.text(1.4, 0.45, "4 × 240 × 320", ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(1.4, 0.05, r"$\sqrt{I}$ (R, G, B)  +  $z/10$", ha="center", fontsize=13, color=INK2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=MUTED))

    blue, blue_ec = "#d9e9f7", SERIES[0]
    stages = [("stem", 64, "120×160"), ("stage 1", 64, "60×80"), ("stage 2", 128, "30×40"),
              ("stage 3", 256, "15×20"), ("stage 4", 512, "8×10")]
    x, x_start, yc = 3.4, 3.4, 1.95
    arrow((3.0, yc), (x - 0.05, yc))
    for k, (name, c, hw) in enumerate(stages):
        h = 1.0 + 1.4 * (len(stages) - k) / len(stages)
        w = 0.55 + 0.3 * c / 512
        ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=blue, ec=blue_ec, lw=1.3))
        ax.text(x + w / 2, yc, str(c), ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + w / 2, yc + h / 2 + 0.1, name, ha="center", va="bottom", fontsize=11.5, color=INK)
        ax.text(x + w / 2, yc - h / 2 - 0.1, hw, ha="center", va="top", fontsize=11, color=INK2)
        x += w + 0.32
    ax.text((x_start + x - 0.32) / 2, 3.65, "ResNet-18 (ImageNet weights): 11.2 M parameters", ha="center",
            fontsize=13.5, fontweight="bold", color=SERIES[0])
    arrow((x - 0.28, yc), (x + 0.02, yc))
    x += 0.05
    hx = x
    for name, n in [("avg\npool", "512"), ("FC", "256"), ("FC", "12")]:
        ax.add_patch(FancyBboxPatch((x, yc - 0.42), 0.62, 0.84, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="#f3f4f4", ec="#9aa3a8", lw=1.2))
        ax.text(x + 0.31, yc, name, ha="center", va="center", fontsize=11, color=INK)
        ax.text(x + 0.31, yc - 0.55, n, ha="center", va="top", fontsize=11, color=INK2)
        x += 0.62 + 0.2
        arrow((x - 0.2, yc), (x, yc))
    ax.text((hx + x - 0.2) / 2, yc + 0.62, "head: 0.13 M", ha="center", fontsize=12.5, fontweight="bold", color=INK2)
    ox, ow = x + 0.05, 13.45 - x - 0.05
    ax.add_patch(FancyBboxPatch((ox, 0.25), ow, 3.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#fbe9e4", ec=CORAL, lw=1.5))
    ax.text(ox + ow / 2, 3.2, "12 numbers", ha="center", fontsize=13.5, fontweight="bold", color=CORAL)
    ax.text(ox + ow / 2, 2.88, "4 quantities × (R, G, B)", ha="center", fontsize=11, color=CORAL)
    outs = [(r"$\beta^D$", "exp", "attenuation, /m"), (r"$\beta^B$", "exp", "backscatter, /m"),
            (r"$B^\infty$", "sigmoid", "water colour, 0–1"), (r"$E$", "sigmoid ÷ max", "sunlight colour")]
    for k, (sym, act, what) in enumerate(outs):
        yy = 2.35 - 0.56 * k
        ax.text(ox + 0.15, yy, sym, fontsize=17, va="center", color=INK)
        ax.text(ox + 0.72, yy + 0.11, what, fontsize=11.5, va="center", color=INK)
        ax.text(ox + 0.72, yy - 0.15, act, fontsize=10.5, va="center", color=INK2)
    fig.savefig(OUT / "betanet_architecture.png")
    plt.close(fig)


def pipeline():
    """Compact BetaNet pipeline with slide-sized text (fig4 is too small once shrunk onto a slide)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.set_xlim(0, 11), ax.set_ylim(-0.35, 3.7)
    no_axes(ax)

    def box(x, y, w, h, t, sub, fc="#ffffff", ec="#c3c2b7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2 + 0.2, t, ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=12, color=INK2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=MUTED))

    blue, blue_ec = "#d9e9f7", SERIES[0]
    box(0.05, 2.35, 2.0, 1.1, "Photo  I", "RGB, linear")
    box(0.05, 0.75, 2.0, 1.1, "Range  z", "SfM, stereo, DAv2")
    box(2.6, 1.5, 1.95, 1.3, "BetaNet", "ResNet-18", fc=blue, ec=blue_ec)
    box(5.0, 1.5, 2.1, 1.3, "12 numbers", r"$\beta^D,\ \beta^B,\ B^\infty,\ E$", fc="#fbe9e4", ec=CORAL)
    box(7.55, 1.5, 1.7, 1.3, "Physics", "closed form", fc=blue, ec=blue_ec)
    box(9.65, 1.5, 1.3, 1.3, "Clean", r"image $\hat{J}$")
    arrow((2.05, 2.9), (2.6, 2.45))
    arrow((2.05, 1.3), (2.6, 1.85))
    arrow((4.55, 2.15), (5.0, 2.15))
    arrow((7.1, 2.15), (7.55, 2.15))
    arrow((9.25, 2.15), (9.65, 2.15))
    ax.annotate("", xy=(8.4, 1.5), xytext=(1.05, 0.75), arrowprops=dict(
        arrowstyle="-|>", color=MUTED, lw=1.4, connectionstyle="arc3,rad=0.22", linestyle=(0, (4, 3))))
    ax.text(4.6, -0.28, "z also goes into the physics layer", ha="center", fontsize=12, color=INK2)
    ax.text(6.05, 3.05, "output 1: what the water is", ha="center", fontsize=13, color=CORAL)
    ax.text(10.3, 3.05, "output 2", ha="center", fontsize=13, color=INK2)
    fig.savefig(OUT / "pipeline_slide.png")
    plt.close(fig)


EQS = {
    # earlier single-beta model (e.g. the one UWCNN's data are built on)
    "eq_old_model": r"$I_c = J_c\,e^{-\beta_c z} + B^\infty_c\big(1-e^{-\beta_c z}\big)$",
    "eq_revised": r"$I_c = \underbrace{J_c\,e^{-\beta^D_c\,z}}_{\textrm{direct signal}}"
                  r" + \underbrace{B^\infty_c\big(1-e^{-\beta^B_c\,z}\big)}_{\textrm{backscatter}},"
                  r"\qquad c\in\{R,G,B\}$",
    "eq_matrix": r"$\underbrace{\begin{bmatrix} I_R\\ I_G\\ I_B\end{bmatrix}}_{\textrm{pixel colour}}="
                 r"\underbrace{\begin{bmatrix} e^{-\beta^D_R z}&0&0\\0&e^{-\beta^D_G z}&0\\0&0&e^{-\beta^D_B z}\end{bmatrix}}_{T_D(z)\textrm{: fading}}"
                 r"\underbrace{\textcolor[HTML]{C8553D}{\begin{bmatrix} E_R&0&0\\0&E_G&0\\0&0&E_B\end{bmatrix}}}_{\textcolor[HTML]{C8553D}{\textrm{sunlight colour (new)}}}"
                 r"\underbrace{\begin{bmatrix} J_R\\ J_G\\ J_B\end{bmatrix}}_{\textrm{clean}}"
                 r"+\underbrace{\begin{bmatrix} 1-e^{-\beta^B_R z}&0&0\\0&1-e^{-\beta^B_G z}&0\\0&0&1-e^{-\beta^B_B z}\end{bmatrix}}_{\mathbf{1}-T_B(z)\textrm{: haze build-up}}"
                 r"\underbrace{\begin{bmatrix} B^\infty_R\\ B^\infty_G\\ B^\infty_B\end{bmatrix}}_{\textrm{water colour}}$",
    "eq_inverse": r"$\hat{\mathbf{J}} = \textcolor[HTML]{C8553D}{\mathrm{diag}(\mathbf{E})^{-1}}\;T_D(z)^{-1}"
                  r"\Big(\mathbf{I}-\big(\mathbf{1}-T_B(z)\big)\,\mathbf{B}^\infty\Big)$",
    "eq_problem": r"$\hat{\theta} = f_w(I, z),\qquad f_w:\ \mathbb{R}^{4\times 240\times 320}\rightarrow\mathbb{R}^{12}$",
    "eq_theta": r"$\theta = \big[\,\boldsymbol{\beta}^D,\ \boldsymbol{\beta}^B,\ \mathbf{B}^\infty,\ "
                r"\textcolor[HTML]{C8553D}{\mathbf{E}}\,\big]\in\mathbb{R}^{3\times 4}$",
}


def equations():
    """pdflatex (standalone) -> transparent PNG. Bypasses matplotlib's usetex, which needs cm-super."""
    import subprocess
    import tempfile
    eq_dir = OUT / "equations"
    eq_dir.mkdir(exist_ok=True)
    head = (r"\documentclass[border=3pt,varwidth=false]{standalone}\usepackage{amsmath,amssymb,bm}\usepackage{xcolor}"
            r"\begin{document}\color[HTML]{1D2B36}\Large ")
    for name, tex in EQS.items():
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "eq.tex").write_text(head + tex + r"\end{document}")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"], cwd=tmp,
                           check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["pdftocairo", "-png", "-transp", "-r", "600", "-singlefile", "eq.pdf",
                            str(eq_dir / name)], cwd=tmp, check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--eq-only" not in sys.argv:
        raw_photos()
        red_light_curve()
        image_formation()
        jerlov_waters()
        sunlight_fix()
        depth_maps()
        depth_scale_per_frame()
        pipeline()
        restoration_steps()
        architecture()
    equations()
