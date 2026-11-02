"""Networks.

BetaNet  : image (+ depth) -> 12 numbers (beta_D, beta_B, B_inf, illuminant E per RGB channel).
           The restored image is then produced by the physics inversion in physics.restore, so the
           network's output is interpretable and the image comes "for free".
UNetDirect: black-box baseline, image (+ depth) -> restored image directly (UWCNN-style).
"""
import torch
import torch.nn as nn
import torchvision


def net_input(I, z):
    """Network sees a gamma-compressed image (closer to what CNNs are used to) + scaled range."""
    x = I.clamp(0, 1).sqrt()
    return x if z is None else torch.cat([x, (z / 10.0)], 1)


class BetaNet(nn.Module):
    def __init__(self, use_depth=True, pretrained=True):
        super().__init__()
        self.use_depth = use_depth
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        if use_depth:  # 4-channel stem: depth filter initialised from the mean RGB filter
            c = r.conv1
            r.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                r.conv1.weight[:, :3] = c.weight
                r.conv1.weight[:, 3:] = c.weight.mean(1, keepdim=True)
        r.fc = nn.Identity()
        self.backbone = r
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 12))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, I, z):
        x = net_input(I, z if self.use_depth else None)
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        o = self.head(self.backbone(x)).float()
        E = torch.sigmoid(o[:, 9:12] + 1.0)
        return {"beta_D": torch.exp(o[:, 0:3].clamp(-7, 2.6) - 1.5),
                "beta_B": torch.exp(o[:, 3:6].clamp(-7, 2.6) - 1.5),
                "B_inf": torch.sigmoid(o[:, 6:9] - 1.0),
                "E": (E / E.amax(1, keepdim=True)).clamp_min(0.03)}


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNetDirect(nn.Module):
    def __init__(self, use_depth=True, ch=(32, 64, 128, 256)):
        super().__init__()
        self.use_depth = use_depth
        cin = 4 if use_depth else 3
        self.downs = nn.ModuleList()
        for c in ch:
            self.downs.append(_block(cin, c))
            cin = c
        self.ups = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.upconv.append(nn.ConvTranspose2d(cin, c, 2, 2))
            self.ups.append(_block(2 * c, c))
            cin = c
        self.out = nn.Conv2d(cin, 3, 1)

    def forward(self, I, z):
        x = net_input(I, z if self.use_depth else None)
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for up, blk in zip(self.upconv, self.ups):
            x = up(x)
            x = blk(torch.cat([x, skips.pop()], 1))
        # predict a residual on the gamma image, return linear
        return (self.out(x).float() + net_input(I, None)).clamp(0, 1.2) ** 2


class RefineNet(nn.Module):
    """Small residual corrector on top of a frozen BetaNet. Sees (BetaNet's restored image, the water photo, range) and
    predicts a correction in gamma space, so it can only nudge the physics result, not replace it (~0.1M params vs
    UNetDirect's 7.8M). The last conv is zero-initialised: at step 0 the output equals BetaNet's restoration exactly."""
    def __init__(self, ch=(16, 32, 64)):
        super().__init__()
        self.downs, cin = nn.ModuleList(), 7
        for c in ch:
            self.downs.append(_block(cin, c))
            cin = c
        self.upconv, self.ups = nn.ModuleList(), nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.upconv.append(nn.ConvTranspose2d(cin, c, 2, 2))
            self.ups.append(_block(2 * c, c))
            cin = c
        self.out = nn.Conv2d(cin, 3, 1)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, Jhat, I, z):
        g = Jhat.clamp(0, 1).sqrt()
        x, skips = torch.cat([g, I.clamp(0, 1).sqrt(), z / 10.0], 1), []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for up, blk in zip(self.upconv, self.ups):
            x = blk(torch.cat([up(x), skips.pop()], 1))
        return (self.out(x).float() + g).clamp(0, 1.2) ** 2
