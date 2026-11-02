# Figures

| Folder | Figures | What it is | Script |
|---|---|---|---|
| `1_original_runs/` | fig0–fig14 | Part 1: BetaNet before fine-tuning (physics, method, synthetic + real results, ablations, plan) | `fig_physics.py`, `fig_method.py`, `fig_results.py`, `fig_plan.py` |
| `2_original_finetune/` | fig15–fig20 | Part 2: fine-tuning on real dives (before / collapsed / selected) | `fig_finetune.py` |
| `3_new_runs/` | fig21–fig23 | Longer training run `baseline_v1_60ep_bigdata` vs original: training curves (21), example images (22), synthetic results by water type + coefficient error (23) | `fig_loss_curves.py`, `fig_big.py` |
| `3_new_runs/` | `uieb_original_vs_longer.png` | UIEB real photos: original vs longer run (6 images + per-image PSNR scatter of all 90) | `fig_uieb_compare.py` |
| `3_new_runs/` | `seathru_original_vs_longer.png` | Real Sea-thru dives: original vs longer run (chart frame per dive + classical Sea-thru) and β_D (red) per dive | `fig_seathru_compare.py` |
| `3_new_runs/` | `old_vs_new_verdict.png` | Paired analysis: original vs longer run, each with/without fine-tuning (synthetic by depth, coefficients, chart, UIEB, dive separation, colour drift) | `analyze_old_vs_new.py` |
| `5_review2_day1_safe_restore/` | `fins_steps_<model>.png` | Review 2, Day 1 (no retraining): the fins photo (UIEB val #2) after each fix in turn (fog cap, red smoothing, careful white balance), red channel underneath | `fig_day1.py` (data: `eval_day1.py` → `results/review2_day1/`) |
| `5_review2_day1_safe_restore/` | `uieb_examples_<model>.png` | Review 2, Day 1: first 6 UIEB val images, current recipe vs all Day 1 fixes | `fig_day1.py` |
| `6_review2_day2/1_more_murky_water/` | `fins_compare.png`, `uieb_examples.png`, `summary.png` | Review 2, Day 2 run 1 (`day2a_added_murky_water`: all murky water kept, trained from scratch) vs the base model: fins photo, 6 UIEB images, and the water read per dive / dive separation + headline scores | `fig_day2.py` (data: `day2_queue.sh` → `results/review2_day2/<run>/`) |
| `6_review2_day2/2_low_red_water/` | same three | Day 2 run 2 (`day2b_added_deep_red_starved_water`: continued from the base model with deep, red-starved water added) | `fig_day2.py` |
| `6_review2_day2/3_camera_flaws/` | same three | Day 2 run 3 (`day2c_added_camera_flaws`: continued from the base model with camera flaws: noise, 8-bit, JPEG) | `fig_day2.py` |

`<model>` = `original` (= `baseline_v0_40ep`, old name `beta_rgbd`) or `longer_run` (= `baseline_v1_60ep_bigdata`, old name `beta_rgbd_big`, the model in use). See `docs/MODELS.md`.
In the Day 2 figures "current baseline" = `baseline_v1_60ep_bigdata`.
Filenames and fig numbers are unchanged; only the folders are new. Docs that cite "fig N" still apply.
