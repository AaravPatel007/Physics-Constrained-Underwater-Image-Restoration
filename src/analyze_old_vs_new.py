"""Original vs longer-trained BetaNet (and each fine-tuned with the ft_nodive recipe): paired analysis + verdict figure.

  synthetic : 654 NYU test rooms, same water as eval_synth (seed 2026). Per-image PSNR and coefficient errors,
              paired differences with bootstrap 95 % CI, split by how deep the room is.
  UIEB      : per-image PSNR / dE00 from real_eval*.json (paired, 90 images).
  Sea-thru  : chart grey error (real_eval*.json); label-free physics checks on held-out real frames and the
              dive separation index (finetune_eval.json, written by eval_finetune.py).
  training  : convergence from the training logs.
Writes results/old_vs_new_analysis.json and figures/3_new_runs/old_vs_new_verdict.png.
"""
import json
from pathlib import Path

import numpy as np
import torch

from eval_synth import DEV, load
from physics import add_water, linear_to_srgb, restore, sample_water, srgb_to_linear
from train import load_split
from vizstyle import INK2, MUTED, SERIES, plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = {"beta_rgbd": ("Original", ""), "beta_rgbd_big": ("Longer run", "_big"),
          "ft_nodive": ("Original + fine-tune", "_ft_nodive"), "ft_big_nodive": ("Longer run + fine-tune", "_ft_big_nodive")}
GW, OWN = "BetaNet + gray-world", "BetaNet"


def boot(d, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "share_better": float((d > 0).mean())}


@torch.no_grad()
def synth(names):
    rgb, depth = load_split("test")
    water = sample_water(len(rgb), DEV, torch.Generator().manual_seed(2026))
    gen = torch.Generator().manual_seed(7)
    out = {n: {"psnr": [], "beta_D": [], "beta_B": [], "B_inf": [], "E": []} for n in names}
    zmax = []
    nets = {n: load(n)[0] for n in names}
    for b in range(0, len(rgb), 16):
        sl = slice(b, b + 16)
        J = srgb_to_linear(rgb[sl].to(DEV).permute(0, 3, 1, 2).float() / 255)
        z = depth[sl].to(DEV).float()[:, None]
        w = {k: v[sl] for k, v in water.items()}
        I = (add_water(J, z, w)[0] + 0.005 * torch.randn(J.shape, generator=gen).to(DEV)).clamp(0, 1)
        zmax += z.amax((1, 2, 3)).tolist()
        for n, net in nets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = {k: v.float() for k, v in net(I, z).items()}
            mse = ((linear_to_srgb(restore(I, z, p).clamp(0, 1)) - linear_to_srgb(J)) ** 2).mean((1, 2, 3))
            out[n]["psnr"] += (-10 * torch.log10(mse + 1e-10)).tolist()
            for k in ("beta_D", "beta_B", "E"):
                out[n][k] += (p[k].log() - w[k].log()).abs().mean(1).tolist()
            out[n]["B_inf"] += (p["B_inf"] - w["B_inf"]).abs().mean(1).tolist()
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}, np.array(zmax)


def main():
    names = [n for n in MODELS if (RES / n / "best.pt").exists()]
    S, zmax = synth(names)
    A = {"models": names, "synthetic": {}, "uieb": {}, "chart": {}, "dives": {}, "training": {}}

    # synthetic: per-model medians + paired differences vs the original
    terc = np.quantile(zmax, [1 / 3, 2 / 3])
    groups = {"near rooms (max depth < %.1f m)" % terc[0]: zmax < terc[0],
              "middle": (zmax >= terc[0]) & (zmax < terc[1]),
              "far rooms (max depth >= %.1f m)" % terc[1]: zmax >= terc[1]}
    for n in names:
        A["synthetic"][n] = {"psnr_mean": float(S[n]["psnr"].mean()),
                             **{f"median_err_{k}": float(np.median(S[n][k])) for k in ("beta_D", "beta_B", "B_inf", "E")}}
    for n in names[1:]:
        d = S[n]["psnr"] - S["beta_rgbd"]["psnr"]
        A["synthetic"][n]["psnr_vs_original"] = boot(d)
        A["synthetic"][n]["psnr_vs_original_by_depth"] = {g: boot(d[m]) for g, m in groups.items()}
        A["synthetic"][n]["beta_D_err_vs_original"] = boot(S["beta_rgbd"]["beta_D"] - S[n]["beta_D"])  # >0 = lower error
        A["synthetic"][n]["beta_B_err_vs_original"] = boot(S["beta_rgbd"]["beta_B"] - S[n]["beta_B"])

    # UIEB + chart
    ev = {n: json.load(open(RES / f"real_eval{MODELS[n][1]}.json")) for n in names}
    base_u = ev["beta_rgbd"]["uieb"]["per_image"]
    for n in names:
        u = ev[n]["uieb"]["per_image"]
        A["uieb"][n] = {m: {"psnr": float(np.mean(u[m]["psnr"])), "de2000": float(np.mean(u[m]["de2000"]))} for m in (GW, OWN)}
        if n != "beta_rgbd":
            A["uieb"][n]["psnr_gw_vs_original"] = boot(np.array(u[GW]["psnr"]) - np.array(base_u[GW]["psnr"]))
        ch = ev[n]["seathru"]["chart"]
        A["chart"][n] = {"own_E": float(np.mean([c[f"{OWN} [SfM range]"] for c in ch.values()])),
                         "gray_world": float(np.mean([c[f"{GW} [SfM range]"] for c in ch.values()]))}

    # real dives (label-free)
    fe = json.load(open(RES / "finetune_eval.json"))
    for n in names:
        if n not in fe:
            continue
        s = fe[n]["sites"]
        bd = {d: s[d]["beta_D"][0] for d in ("D1", "D2", "D3", "D4", "D5")}
        A["dives"][n] = {"separation": s["separation"], "beta_D_red": bd,
                         "D4_rank_of_5": int(sorted(bd, key=bd.get, reverse=True).index("D4") + 1),
                         "D5_rank_of_5": int(sorted(bd, key=bd.get, reverse=True).index("D5") + 1),
                         "heldout_colour_drift": fe[n]["heldout"]["range"], "D3_colour_drift": fe[n]["heldout_D3"]["range"],
                         "heldout_dark": fe[n]["heldout"]["dark"]}

    # convergence
    for n in ("beta_rgbd", "beta_rgbd_big"):
        log = json.load(open(RES / n / "log.json"))
        v = np.array([r["val_coef"] for r in log])
        e = np.arange(len(v))
        k = max(10, len(v) // 6)
        A["training"][n] = {"epochs": len(v), "val_coef_last": float(v[-1]), "val_coef_best": float(v.min()),
                            "val_coef_slope_last_%d_epochs_per_10" % k: float(np.polyfit(e[-k:], v[-k:], 1)[0] * 10),
                            "train_minus_val": float(log[-1]["train_loss"] - log[-1]["val_coef"])}
    json.dump(A, open(RES / "old_vs_new_analysis.json", "w"), indent=1)
    print(json.dumps(A, indent=1))
    figure(A, names, groups)


def figure(A, names, groups):
    lab = [MODELS[n][0] for n in names]
    col = [SERIES[i] for i in range(len(names))]
    fig, axs = plt.subplots(2, 3, figsize=(19, 10))

    ax = axs[0, 0]
    for j, n in enumerate(names[1:], 1):
        g = A["synthetic"][n]["psnr_vs_original_by_depth"]
        m = [g[k]["mean"] for k in groups]
        lo = [g[k]["mean"] - g[k]["ci95"][0] for k in groups]
        hi = [g[k]["ci95"][1] - g[k]["mean"] for k in groups]
        x = np.arange(3) + (j - 2) * 0.26
        ax.bar(x, m, 0.24, color=col[j], label=lab[j], yerr=[lo, hi], capsize=3, ecolor=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(3), ["near rooms", "middle", "far rooms"])
    ax.set_ylabel("PSNR change vs original (dB), 95 % CI")
    ax.set_title("Synthetic: gain over the original, by room depth", fontsize=11)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    keys = ["beta_D", "beta_B", "E"]
    for j, n in enumerate(names):
        ax.bar(np.arange(3) + (j - 1.5) * 0.2, [A["synthetic"][n][f"median_err_{k}"] * 100 for k in keys], 0.18,
               color=col[j], label=lab[j])
    ax.set_xticks(range(3), ["β_D", "β_B", "E"])
    ax.set_ylabel("median |log error| × 100 (≈ % error)")
    ax.set_title("Synthetic: coefficient error (lower is better)", fontsize=11)
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [A["chart"][n]["own_E"] for n in names], 0.34, color=col, alpha=0.45, label="network's own sunlight colour E")
    ax.bar(x + 0.18, [A["chart"][n]["gray_world"] for n in names], 0.34, color=col, label="+ gray-world")
    ax.axhline(8.5, color=MUTED, lw=1, ls="--")
    ax.text(len(names) - 0.5, 8.7, "gray-world alone", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, lab, fontsize=8)
    ax.set_ylabel("grey-patch error (°), mean of 2 chart frames")
    ax.set_title("Real chart frames: colour cast (lower is better)", fontsize=11)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.bar(x - 0.18, [A["uieb"][n][OWN]["psnr"] for n in names], 0.34, color=col, alpha=0.45, label="own E")
    ax.bar(x + 0.18, [A["uieb"][n][GW]["psnr"] for n in names], 0.34, color=col, label="+ gray-world")
    ax.axhline(17.72, color=MUTED, lw=1, ls="--")
    ax.text(len(names) - 0.5, 17.78, "gray-world alone", ha="right", fontsize=8, color=MUTED)
    ax.set_ylim(16.5, 19)
    ax.set_xticks(x, lab, fontsize=8)
    ax.set_ylabel("UIEB PSNR (dB), 90 images")
    ax.set_title("Real UIEB photos (higher is better)", fontsize=11)
    ax.legend(fontsize=8)

    dv = [n for n in names if n in A["dives"]]
    xd = np.arange(len(dv))
    cd = [col[names.index(n)] for n in dv]
    ax = axs[1, 1]
    ax.bar(xd, [A["dives"][n]["separation"] for n in dv], 0.6, color=cd)
    for i, n in enumerate(dv):
        ax.text(i, A["dives"][n]["separation"], f"D4 rank {A['dives'][n]['D4_rank_of_5']}/5", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(xd, [MODELS[n][0] for n in dv], fontsize=8)
    ax.set_ylabel("between-dive ÷ within-dive spread of log β_D")
    ax.set_title("Real dives: are the 5 waters told apart? (higher is better)", fontsize=11)

    ax = axs[1, 2]
    ax.bar(xd, [A["dives"][n]["D3_colour_drift"] for n in dv], 0.6, color=cd)
    ax.set_xticks(xd, [MODELS[n][0] for n in dv], fontsize=8)
    ax.set_ylabel("colour change with distance after restoration")
    ax.set_title("Unseen dive D3: colour drift with distance (lower is better)", fontsize=11)

    fig.suptitle("Original vs longer-trained BetaNet, with and without real-dive fine-tuning", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "3_new_runs" / "old_vs_new_verdict.png")


if __name__ == "__main__":
    main()
