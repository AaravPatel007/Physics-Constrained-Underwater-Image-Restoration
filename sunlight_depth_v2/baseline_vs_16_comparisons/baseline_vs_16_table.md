| metric | BetaNet (12, predicts E) | SunDepth v2 (16, E0, K_d, d_s) |
|---|---|---|
| PSNR [dB] ↑ | 20.41 | 20.16 |
| SSIM ↑ | 0.627 | 0.621 |
| CIEDE2000 ↓ | 13.76 | 13.98 |
| β_D median rel. error R/G/B [%] ↓ | 13.3 / 17.6 / 21.7 | 13.9 / 17.8 / 22.6 |
| β_B median rel. error R/G/B [%] ↓ | 15.6 / 17.6 / 19.8 | 18.2 / 19.0 / 20.6 |
| B∞ median rel. error R/G/B [%] ↓ | 9.6 / 12.1 / 15.2 | 10.9 / 12.3 / 16.7 |
| E median rel. error R/G/B [%] ↓ | 9.2 / 1.2 / 5.4 | 9.4 / 1.2 / 5.2 |
| sunlight E colour error, mean [°] ↓ | 4.05 | 3.92 |
| E0 colour error, mean [°] ↓ | n/a | 5.57 |
| K_d median rel. error R/G/B [%] ↓ | n/a | 15.7 / 20.4 / 25.0 |
| depth d_s MAE / RMSE [m] ↓ | n/a | 1.78 / 2.34 |
| latency, network + restore [ms] ↓ | 1.22 | 1.46 |
| UIEB CIEDE2000, own E ↓ | 14.02 | 14.91 |
| UIEB CIEDE2000, Day 1b ↓ | 12.67 | 12.95 |
| Sea-thru chart error, own E [°] ↓ | 18.4 | 15.5 |

Paired over the same 654 test scenes (v2 − BetaNet, 95 % bootstrap interval over scenes):

- PSNR [dB]: -0.253 [-0.402, -0.101], v2 better on 47 % of scenes
- SSIM: -0.006 [-0.009, -0.003], v2 better on 44 % of scenes
- CIEDE2000: +0.211 [-0.023, +0.450], v2 better on 48 % of scenes
