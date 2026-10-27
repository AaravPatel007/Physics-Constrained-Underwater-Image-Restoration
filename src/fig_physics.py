"""Explainer figures: image-formation decomposition, the 10 Jerlov waters, attenuation vs range."""
from pathlib import Path

import numpy as np
import torch

from physics import JERLOV_BETA, JERLOV_N, add_water, linear_to_srgb, sample_water, srgb_to_linear
from vizstyle import CH, CH_MARK, INK2, MUTED, SERIES, no_axes, plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
d = np.load(ROOT / "data" / "nyu_test.npz")
SCENE = 37


def scene(i):
    J = srgb_to_linear(torch.from_numpy(d["rgb"][i]).permute(2, 0, 1)[None].float() / 255)
    z = torch.from_numpy(d["depth"][i].astype(np.float32))[None, None] * 1.5
    return J, z


def show(x):
    return linear_to_srgb(x)[0].permute(1, 2, 0).numpy()


def decomposition():
    J, z = scene(SCENE)
    w = sample_water(1, water_type="3C")
    w["beta_B"] = w["beta_B"] * torch.tensor([[1.3, 0.8, 0.7]])
    w["B_inf"] = torch.tensor([[0.05, 0.32, 0.38]])
    w["E"] = torch.exp(-w["beta_D"] * 4.0)
    w["E"] = w["E"] / w["E"].amax(1, keepdim=True)
    I, D, B = add_water(J, z, w)
    fig, axs = plt.subplots(1, 6, figsize=(20, 3.4))
    panels = [(show(J), "Clean scene  J", "what we want back"),
              (show(J * w["E"][:, :, None, None]), "Lit by sunlight at depth", r"$J\cdot E$ : red already lost on the way down"),
              (z[0, 0].numpy(), "Range  z  (metres)", "camera-to-object distance"),
              (show(D), "Direct signal", r"$J E\,e^{-\beta_D z}$ : colours fade with z"),
              (show(B * 1.6), "Backscatter (x1.6)", r"$B_\infty(1-e^{-\beta_B z})$ : haze grows with z"),
              (show(I), "Underwater photo  I", "= direct + backscatter")]
    for ax, (im, t, sub) in zip(axs, panels):
        if im.ndim == 2:
            h = ax.imshow(im, cmap="Blues")
            cb = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            cb.outline.set_visible(False)
        else:
            ax.imshow(im)
        no_axes(ax)
        ax.set_title(t, fontsize=12)
        ax.text(0.0, -0.08, sub, transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.12)
    fig.suptitle("How water changes a photo (Akkaynak & Treibitz revised model)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.savefig(FIG / "fig1_image_formation.png")
    plt.close(fig)


def jerlov_grid():
    J, z = scene(SCENE)
    gen = torch.Generator().manual_seed(3)
    fig, axs = plt.subplots(2, 6, figsize=(17, 5.4))
    axs[0, 0].imshow(show(J))
    axs[0, 0].set_title("In air", fontsize=11)
    axs[1, 0].imshow(z[0, 0].numpy(), cmap="Blues")
    axs[1, 0].set_title("Range map", fontsize=11)
    names = {"I": "Ocean I (clearest)", "IA": "Ocean IA", "IB": "Ocean IB", "II": "Ocean II",
             "III": "Ocean III", "1C": "Coastal 1", "3C": "Coastal 3", "5C": "Coastal 5",
             "7C": "Coastal 7", "9C": "Coastal 9 (murkiest)"}
    for k, (t, ax) in enumerate(zip(JERLOV_N, [*axs[0, 1:], *axs[1, 1:]])):
        w = sample_water(1, gen=gen, water_type=t)
        I, _, _ = add_water(J, z, w)
        ax.imshow(show(I))
        b = JERLOV_BETA[t]
        ax.set_title(names[t], fontsize=11)
        ax.text(0.0, -0.04, rf"$\beta_D$ R {b[0]:.2f}  G {b[1]:.2f}  B {b[2]:.2f}", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK2)
    for ax in axs.ravel():
        no_axes(ax)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.08, hspace=0.3)
    fig.suptitle("Same scene, ten Jerlov water types: the network must tell these apart", x=0.01,
                 ha="left", fontsize=14, fontweight="bold")
    fig.savefig(FIG / "fig2_jerlov_types.png")
    plt.close(fig)


def attenuation_curves():
    r = np.linspace(0, 15, 200)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    names = dict(zip("RGB", ["Red", "Green", "Blue"]))
    panels = [(axs[0], "I", "Clear ocean (Jerlov I): red dies first -> blue cast", 15),
              (axs[1], "9C", "Murky coastal (Jerlov 9C): blue dies first -> yellow cast", 0.9)]
    for ax, t, title, xl in panels:
        for c, b in zip("RGB", JERLOV_BETA[t]):
            y = np.exp(-b * r) * 100
            ax.plot(r, y, color=CH[c], marker=CH_MARK[c], markevery=25, markersize=7)
            ax.annotate(names[c], (xl, np.exp(-b * xl) * 100), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK2)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("distance through water (m)")
        ax.set_xlim(0, 17.5)
    axs[0].set_ylabel("light surviving (%)")
    fig.suptitle("Each colour fades at its own rate, and the rates depend on the water", x=0.01,
                 ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(FIG / "fig3_attenuation_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    decomposition()
    jerlov_grid()
    attenuation_curves()
