| metric | BetaNet (12, predicts E) | SunDepth v1 (10, d_s; K_d = 0.8 β_D) | SunDepth v2 (16, E0, K_d, d_s) |
|---|---|---|---|
| PSNR [dB] ↑ | 20.41 | 19.89 | 20.16 |
| SSIM ↑ | 0.627 | 0.629 | 0.621 |
| CIEDE2000 ↓ | 13.76 | 15.15 | 13.98 |
| β_D median rel. error R/G/B [%] ↓ | 13.3 / 17.6 / 21.7 | 13.0 / 17.1 / 21.8 | 13.9 / 17.8 / 22.6 |
| β_B median rel. error R/G/B [%] ↓ | 15.6 / 17.6 / 19.8 | 15.3 / 18.0 / 21.0 | 18.2 / 19.0 / 20.6 |
| B∞ median rel. error R/G/B [%] ↓ | 9.6 / 12.1 / 15.2 | 10.0 / 12.4 / 15.4 | 10.9 / 12.3 / 16.7 |
| E median rel. error R/G/B [%] ↓ | 9.2 / 1.2 / 5.4 | 16.0 / 6.7 / 15.7 | 9.4 / 1.2 / 5.2 |
| sunlight E colour error, mean [°] ↓ | 4.05 | 9.38 | 3.92 |
| E0 colour error, mean [°] ↓ | n/a | 9.07 (fixed white) | 5.57 |
| K_d median rel. error R/G/B [%] ↓ | n/a | 27.5 / 27.9 / 27.0 (0.8 β̂_D) | 15.7 / 20.4 / 25.0 |
| depth d_s MAE / RMSE [m] ↓ | n/a | 1.93 / 2.52 | 1.78 / 2.34 |
| latency, network + restore [ms] ↓ | 1.22 | 1.25 | 1.46 |
| UIEB CIEDE2000, own E ↓ | 14.02 | 15.70 | 14.91 |
| UIEB CIEDE2000, Day 1b ↓ | 12.67 | 13.06 | 12.95 |
| Sea-thru chart error, own E [°] ↓ | 18.4 | 18.6 | 15.5 |
