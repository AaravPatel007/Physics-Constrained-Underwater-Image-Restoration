# Figures

| Folder | Figures | What it is | Script |
|---|---|---|---|
| `1_original_runs/` | fig0–fig14 | Part 1: BetaNet before fine-tuning (physics, method, synthetic + real results, ablations, plan) | `fig_physics.py`, `fig_method.py`, `fig_results.py`, `fig_plan.py` |
| `2_original_finetune/` | fig15–fig20 | Part 2: fine-tuning on real dives (before / collapsed / selected) | `fig_finetune.py` |
| `3_new_runs/` | fig21–fig23 | Longer training run `beta_rgbd_big` vs original: training curves (21), example images (22), synthetic results by water type + coefficient error (23) | `fig_loss_curves.py`, `fig_big.py` |
| `3_new_runs/` | `uieb_original_vs_longer.png` | UIEB real photos: original vs longer run (6 images + per-image PSNR scatter of all 90) | `fig_uieb_compare.py` |
| `3_new_runs/` | `seathru_original_vs_longer.png` | Real Sea-thru dives: original vs longer run (chart frame per dive + classical Sea-thru) and β_D (red) per dive | `fig_seathru_compare.py` |
| `3_new_runs/` | `old_vs_new_verdict.png` | Paired analysis: original vs longer run, each with/without fine-tuning (synthetic by depth, coefficients, chart, UIEB, dive separation, colour drift) | `analyze_old_vs_new.py` |

Filenames and fig numbers are unchanged; only the folders are new. Docs that cite "fig N" still apply.
