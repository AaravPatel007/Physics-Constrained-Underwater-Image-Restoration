"""SunDepthBetaNet v2: 16 outputs; the sunlight E is computed from predicted E0, K_d and d_s.

Same backbone, input and beta_D / beta_B / B_inf heads as BetaNet (src/model.py); only the last layer is wider
(256 -> 16) and the three sunlight numbers are replaced by seven physical ones:

  o[0:3]   beta_D = exp(clamp(o) - 1.5)                 as BetaNet
  o[3:6]   beta_B = exp(clamp(o) - 1.5)                 as BetaNet
  o[6:9]   B_inf  = sigmoid(o - 1)                      as BetaNet
  o[9:12]  E0     = exp(clamp(o, -3, 3)), max -> 1      surface daylight colour (only its chromaticity matters)
  o[12:15] K_d    = exp(ln 1e-3 + ln 6000 * sigmoid(o)) in (0.001, 6) /m, log-uniform sensitivity
  o[15]    d_s    = 15 m * sigmoid(o)                    in (0, 15) m (labels 0.5-12 m, as in v1)
  E = E0 exp(-K_d d_s), max-normalised, floored at 0.03  (water_v2.sunlight_v2, same convention as BetaNet's E)
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BetaNet, net_input  # noqa: E402
from water_v2 import sunlight_v2  # noqa: E402

D_MAX = 15.0
KD_MIN, KD_MAX = 1e-3, 6.0


class SunDepthBetaNetV2(BetaNet):
    def __init__(self, use_depth=True, pretrained=True):
        super().__init__(use_depth, pretrained)
        self.head[-1] = nn.Linear(self.head[-1].in_features, 16)

    def forward(self, I, z, override=None):
        """override: dict with any of E0 / K_d / d_s to replace the prediction (attribution experiments)."""
        x = net_input(I, z if self.use_depth else None)
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        o = self.head(self.backbone(x)).float()
        p = {"beta_D": torch.exp(o[:, 0:3].clamp(-7, 2.6) - 1.5),
             "beta_B": torch.exp(o[:, 3:6].clamp(-7, 2.6) - 1.5),
             "B_inf": torch.sigmoid(o[:, 6:9] - 1.0)}
        E0 = torch.exp(o[:, 9:12].clamp(-3, 3))
        p["E0"] = E0 / E0.amax(1, keepdim=True)
        p["K_d"] = torch.exp(math.log(KD_MIN) + math.log(KD_MAX / KD_MIN) * torch.sigmoid(o[:, 12:15]))
        p["d_s"] = D_MAX * torch.sigmoid(o[:, 15:16])
        u = {**p, **(override or {})}
        p["E"] = sunlight_v2(u["E0"], u["K_d"], u["d_s"])
        return p
