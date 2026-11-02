# Model names (read this first when resuming)

Last updated 2 Nov 2026 (end of Day 2). Folder = `results/<name>/best.pt`.

| Name | What it is | Status |
|---|---|---|
| `baseline_v0_40ep` (old name `beta_rgbd`, still works as a link) | The original BetaNet: 40 epochs, 1 100 NYU rooms, clean fake water. Used for every Part 1 / Part 2 figure of the 29 Oct review. | Old baseline. Kept for the old figures. |
| `baseline_v1_60ep_bigdata` (old name `beta_rgbd_big`, link) | Same network, 60 epochs, 7 100 NYU rooms, 40 % of the very murky water kept. Trained 28 Oct. | **The model we use now.** |
| **current model** = `baseline_v1_60ep_bigdata` **+ Day 1 fixes** | Same weights. The Day 1 fixes are applied when the photo is restored (`src/safe_restore.py`): fog cap, smoothing of the weak red channel, careful white balance. No retraining. | **What we show for Method 1.** |
| **Day 1b** (recipe change, same weights) | Day 1 fixes with a 3 px smoothing patch instead of 15 px (`safe_restore(..., **CURRENT_RECIPE)` in `src/safe_restore.py`; margin stays 3). Removes the pink halo around objects. Chosen on 200 UIEB train photos (`src/eval_day1b.py`). | **In use from 2 Nov (late) on** (ΔE 12.25 → 12.21; fins specks 4.4 → 5.8). |
| `day2a_added_murky_water` | Trained from scratch like v1, but ALL very murky water kept (100 % instead of 40 %). | Side result, not used. |
| `day2b_added_deep_red_starved_water` | v1 plus 20 more epochs where 40 % of fake water is deep with almost no red (fins-like). | Side result, not used. |
| `day2c_added_camera_flaws` | v1 plus 20 more epochs where half the fake photos get camera noise, 8-bit rounding and JPEG. | Side result, not used. |
| `day2d_added_murky_deep_red_camera_60ep` | Prepared, **not run yet**. From scratch like v1 (60 epochs, lr 5e-4) with all three changes together: all murky water, 40 % deep red-starved water, half the photos with camera flaws. | If it beats v1 on the real photos it replaces it; else side result. |

Old names (`beta_rgbd`, `beta_rgbd_big`) are links to the new folders, so old scripts, `finetune_eval.json` keys
and old figures keep working. New work should use the new names.

Naming rule going forward: `baseline_v<n>_<what>` for base models, `day<N><letter>_added_<what>` for experiments.

## Plain-language answers

**Is the current model "60 epochs + Day 1 fixes"?** Yes. 60 epochs on the big dataset, with the Day 1 clean-up
applied at restoration time. None of the three Day 2 runs replaced it.

**Day 1: what it tried and is it better?** The model was fine, the way we *used* its output was not. It
subtracted more fog than a photo had, which erased red and then the white-balance step blew the leftover noise
up into red specks. Day 1 caps the fog by the darkest pixels, smooths the weak red channel using the sharp green
channel, and caps the white-balance gain. Yes, it is better (90 UIEB val photos, v1 model):

| | Before | With Day 1 fixes |
|---|---|---|
| Image quality (PSNR) | 18.48 dB | 19.09 dB |
| Colour error vs reference (ΔE, lower = better) | 13.23 | 12.25 |
| Fins photo red specks | 10.8 | 4.4 |
| Fins photo colour error | 20.0 | 18.7 |

It cost nothing and needed no training. Limit: the fins photo is still blue-green, because that photo has almost
no red light left to recover.

**Day 2: what it tried and how it failed.** The idea: the fake training water might not cover real cases like
the fins photo, so train on more extreme water. Three separate experiments:
(a) more very murky water, (b) deep water with almost no red, (c) camera flaws (noise, 8-bit, JPEG).
- (a) changed almost nothing (dive separation 1.29 → 1.34, UIEB colour slightly worse).
- (b) learned the fake deep water (+0.7 dB there) but the real red-starved photos and the fins got slightly worse.
- (c) halved the red specks on its own (10.8 → 5.4) but, with the Day 1 fixes already on, ends level with v1;
  colour charts slightly worse.
None fixed the tint. Reading: the problem is not the range of fake water. In the fins photo the red is
essentially gone, and a model that only estimates water effects cannot bring back what the camera never
recorded. Day 2 is still useful for the talk: three explanations tested and ruled out.

**What next (Day 3):** a small learned "touch-up" network on top of the restored image, trained on the 800 UIEB
train photo/reference pairs, on `baseline_v1_60ep_bigdata` + Day 1 fixes. Not started.
