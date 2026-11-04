"""SunDepthBetaNet: BetaNet whose sunlight colour E comes from a predicted depth below the surface.

    baseline BetaNet : image + range -> ResNet-18 -> 12 numbers: beta_D(3), beta_B(3), B_inf(3), E(3)
    SunDepthBetaNet  : image + range -> ResNet-18 -> 10 numbers: beta_D(3), beta_B(3), B_inf(3), d_s(1)
                       E_c = exp(-K_d,c d_s) / max_c(...), floor 0.03,   K_d = KAPPA0 * beta_D   (sun_physics.py)

Everything else is BetaNet unchanged: same backbone, stem, input scaling and the same output transforms for beta_D,
beta_B and B_inf. The output dict has the same keys (plus d_s and K_d), so physics.restore, train.coef_loss,
train.img_loss and safe_restore work on it without changes.

d_s is bounded to (0, D_MAX) = (0, 15) m by a sigmoid: positive, and wide enough for every depth in the training data
(0.5-12 m) and for every depth that reproduces a training E under the fixed KAPPA0 (up to 12 * 1.0 / 0.8 = 15 m).

detach_kd (default True): K_d is computed from a *detached* beta_D, so beta_D receives exactly the gradients it gets
in the baseline (coefficient loss + water removal in the image loss) and the sunlight term cannot pull on it. The
sunlight path then trains only the depth unit. This keeps the comparison to "only the E component changed".
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import BetaNet, net_input  # noqa: E402

from sun_physics import D_MAX, KAPPA0, sunlight_from_depth  # noqa: E402


class SunDepthBetaNet(BetaNet):
    def __init__(self, use_depth=True, pretrained=True, kappa0=KAPPA0, d_max=D_MAX, detach_kd=True):
        super().__init__(use_depth, pretrained)
        self.head[-1] = nn.Linear(256, 10)  # 3 beta_D + 3 beta_B + 3 B_inf + 1 d_s (was 12: ... + 3 E)
        self.kappa0, self.d_max, self.detach_kd = kappa0, d_max, detach_kd

    def forward(self, I, z, d_override=None):
        """d_override (B,1) metres: use this depth instead of the predicted one (for oracle-depth tests)."""
        x = net_input(I, z if self.use_depth else None)
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        o = self.head(self.backbone(x)).float()
        beta_D = torch.exp(o[:, 0:3].clamp(-7, 2.6) - 1.5)   # identical to BetaNet
        beta_B = torch.exp(o[:, 3:6].clamp(-7, 2.6) - 1.5)   # identical to BetaNet
        B_inf = torch.sigmoid(o[:, 6:9] - 1.0)                 # identical to BetaNet
        d_s = self.d_max * torch.sigmoid(o[:, 9:10])           # new: depth below the surface [m]
        d_use = d_s if d_override is None else d_override
        K_d = self.kappa0 * (beta_D.detach() if self.detach_kd else beta_D)
        return {"beta_D": beta_D, "beta_B": beta_B, "B_inf": B_inf, "E": sunlight_from_depth(K_d, d_use),
                "d_s": d_s, "K_d": K_d}
