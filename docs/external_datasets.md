# External datasets: improving and stress-testing the pipeline

Written 4 Nov 2026. Downloads go into `data/external/<name>/` (git-ignored). Status: **got** = downloaded
here; **manual** = needs a person (login, password, form or blocked site); **partial** = sample only (disk).
Already in use and not repeated here: NYU-v2, UIEB (890 pairs), Sea-thru D1–D5, Depth Anything V2.

## A. Datasets that could improve the pipeline

| Dataset | What it is | Where it helps | Status |
|---|---|---|---|
| **LSUI** (Peng et al., TIP 2023) | 4,279 real underwater photos + reference images, more water types and lighting than UIEB | RefineNet stage 2 trains on 640 UIEB pairs today; LSUI gives ~5× more real pairs. Same caveat as UIEB: the references are chosen enhancements, not ground truth | see table C |
| **EUVP**, Underwater Dark subset | 5,550 paired photos (poor vs good quality), low light | Extra RefineNet pairs plus low-light cases. Its references are partly made by a GAN, so give it a lower weight than UIEB/LSUI | got (HF mirror `Ken1053/EUVP`, 190 MB) |
| **EUVP** full (ImageNet + Scenes subsets) | about 6,000 more pairs | Same as above | manual |
| **HICRD** (Heron Island Coral Reef) | 6,003 raw photos and 2,000 restored ones. **Measured diffuse attenuation K_d at 6 of 8 sites, plus the camera's spectral response** | The only real set whose references come from **measured water physics**. (1) Check BetaNet's β_D read-out against measured K_d. (2) The camera response is what blocks using colour (non-grey) chart patches today (SUMMARY §6.0) | manual |
| **DIODE** val | Indoor + **outdoor** RGB-D with laser metric depth (771 frames, 2.6 GB) | SUMMARY next step 1: outdoor/natural RGB-D for the synthetic water. This reduces the NYU indoor domain gap that pushes warm indoor colours onto the UNet and RefineNet outputs | see table C |
| **Hypersim** (processed, HF `omrastogi/Hypersim-Processed`) | Photoreal synthetic indoor RGB-D | What Cai & Metzler 2025 used to fine-tune DAv2 for underwater; a route to the DAv2 metric-scale fix | not fetched (large; only if the DAv2 fix is pursued) |
| **FLSea-VI** | Real underwater (Mediterranean and Red Sea) with metric depth, 64 GB | Real water + metric depth for the DAv2-scale fix and for label-free fine-tuning beyond Sea-thru | partial: 2 of 109 shards (~1 GB) |
| **SeaThru-NeRF** scenes | Real multi-view scenes (Red Sea, Caribbean, Pacific) with poses | Multi-view range lets you test the range-dependent β_D(z) idea (SUMMARY next step 2) | see table C |

## B. Datasets for stress-testing

| Dataset | What it stresses | How to use it | Status |
|---|---|---|---|
| **SQUID** (Berman et al., TPAMI 2020) | 57 stereo pairs, 4 sites (Red Sea + Mediterranean), **colour charts + metric distance maps** | SUMMARY next step 5. More chart frames for the grey-patch error, with real metric range (no DAv2) | partial: Satil site (6.2 GB, Red Sea). The other 3 sites are 40 GB |
| **TUB** (Turbid Underwater Baseline, 2026) | 33 scenes × 10 **measured turbidity levels (NTU)**, CC-BY-4.0 | Test of the "camera as a water sensor" claim: β_B and β_D should rise steadily with NTU. Also shows where restoration breaks down, matching the "coastal 9: even the oracle fails" finding | got (230 MB) |
| **RUIE** (Liu et al., TCSVT 2020) | UIQS: 3,630 photos in 5 visibility levels. **UCCS: 300 photos grouped by blue / blue-green / green cast**. UTTS: detection | UCCS is the fins-photo failure (blue-green tint) at scale. UIQS gives quality versus visibility level | see table C |
| **U45** | 45 standard hard photos: green, blue, hazy | Common no-reference benchmark (UCIQE/UIQM) for comparing with published methods | see table C |
| **UIEB Challenging-60** | 60 photos with no usable reference | Hard cases, judged by eye and no-reference metrics | manual |
| **EUVP Dark** | Low light | Check whether noise gets amplified (the red-speck problem) | got (shared with A) |
| **TURBID** (Codevilla et al., BMVC 2015) | Tank with milk, methylene blue and chlorophyll at increasing levels, plus a clear-water reference image | Controlled turbidity and colour change with a true clean reference | manual |
| **Optical Ocean Recipes** (Schöntag et al., 2025) | Tank with measured optical water properties for 3 Jerlov types + a drained-tank reference | Closest to real ground truth for β; the paper says "will be released" | manual (ask the authors) |

## C. Download log
Session 5, Mac with ~19 GB free. Everything goes under `data/external/` (git-ignored).

| Dataset | Status | Used for |
|---|---|---|
| LSUI | Got. 2,000 of 4,279 pairs kept in `data/lsui_pairs.npz`; raw folder deleted for disk | RefineNet stage 2, 400 held out for test |
| EUVP Dark | Got (HF `Ken1053/EUVP`). 2,500 of 5,550 pairs kept in `data/euvp_pairs.npz`; raw parquet deleted for disk | RefineNet stage 2, 500 held out for test |
| TUB | Got, 1,313 images; 8 test experiments used (320 images) | Stress test: turbidity |
| RUIE | Got (UCCS, UIQS, UTTS) | Stress test: colour casts, quality levels |
| U45 | Got | Stress test |
| TankImage-I | Got | Stress test: tank, three turbidity levels |
| FLSea sample, SeaThru-NeRF | Not downloaded (disk) | Needs the 120 GB laptop |
| SQUID other sites (40 GB), DIODE, FLSea full (64 GB) | Held back (disk) | Needs the 120 GB laptop |
| HICRD | Manual: CSIRO WebDAV | Needs a person |
| EUVP full, UFO-120 | Manual: IRV lab site returns 403 to scripts | Needs a person |
| UIEB C-60 | Manual: password, no redistribution | Needs a person |
| TURBID | Manual: no public link | Needs a person |
| Optical Ocean Recipes | Not released | Wait |
