"""fig21: BetaNet loss vs epoch (train loss, val coefficient error, val image error) + learning-rate schedule.
Adds the longer run (beta_rgbd_big: 2.6x data, 60 epochs, less murky water) when its log exists."""
import json
from pathlib import Path

import numpy as np
import torch

from vizstyle import INK2, MUTED, SERIES, plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("beta_rgbd", "BetaNet RGB-D (2.9k images, 40 epochs)"),
        ("beta_rgbd_big", "BetaNet RGB-D, longer (7.4k images, 60 epochs,\nless murky water; resumed at epoch 47)")]


def onecycle(epochs, max_lr=5e-4, spe=100):
    """Learning rate at the end of each epoch for train.py's OneCycle schedule."""
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=max_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=epochs * spe, pct_start=0.05)
    lr = []
    for s in range(epochs * spe):
        if s % spe == spe - 1:
            lr.append(opt.param_groups[0]["lr"])
        opt.step(), sched.step()
    return np.array(lr)


def main():
    runs = [(n, lab, json.load(open(ROOT / "results" / n / "log.json")))
            for n, lab in RUNS if (ROOT / "results" / n / "log.json").exists()]
    panels = [("train_loss", "Training loss", "loss (coefficient + image)"),
              ("val_coef", "Validation coefficient error", "coefficient error (log-L1)"),
              ("val_img_l1", "Validation image error", "image error (mean |restored − clean|)"),
              ("val_psnr", "Validation PSNR", "PSNR (dB)")]
    fig, grid = plt.subplots(2, 3, figsize=(17, 9))
    axes = grid.ravel()
    for c, (name, label, log) in enumerate(runs):
        ep = np.array([r["epoch"] for r in log])
        for ax, (k, title, ylab) in zip(axes, panels):
            y = np.array([r[k] for r in log])
            ax.plot(ep, y, color=SERIES[c], lw=2, label=label)
            ax.annotate(f"{y[-1]:.3g}", (ep[-1], y[-1]), xytext=(-8, 8), textcoords="offset points",
                        color=INK2, fontsize=9, ha="right")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylab)
        axes[4].plot(ep, onecycle(len(ep)) * 1e4, color=SERIES[c], lw=2)
    for ax in axes[:4]:
        ax.axvline(10, color=MUTED, lw=0.8, ls=":")
    axes[0].annotate("elbow", (10, axes[0].get_ylim()[1]), xytext=(4, -14), textcoords="offset points",
                     color=MUTED, fontsize=9)
    axes[4].set_title("Learning rate (OneCycle)", fontsize=11)
    axes[4].set_xlabel("epoch")
    axes[4].set_ylabel("learning rate (× 10⁻⁴)")
    axes[5].axis("off")
    axes[5].legend(*axes[0].get_legend_handles_labels(), loc="center", fontsize=10)
    fig.suptitle("BetaNet training curves (same 288 validation frames for both runs)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = ROOT / "figures" / "3_new_runs" / "fig21_loss_curves.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
