"""Day 1 (review 2): safer use of BetaNet's coefficients on real photos. No retraining.

Three fixes for the fins failure (UIEB val #2: red is 2-3 levels of 255, the model's red haze is 16x the red in
the photo, gray-world then blows the few surviving noise pixels up into red specks):

  1. fog cap        : backscatter can never be brighter than the darkest pixels at that range (Sea-thru's
                      dark-pixel idea used as an upper bound), so the red channel is not erased.
  2. weak-channel   : where a channel's direct signal is within the noise floor, rebuild it from the cleanest
     smoothing        channel with a guided filter (He, Sun & Tang, ECCV 2010): red ~ a*green + b per patch.
                      Same idea as Ancuti et al. (TIP 2018), who compensate underwater red from green.
  3. robust white   : white balance from pixels that are signal in every channel, with a capped per-channel gain,
     balance          so a near-empty channel cannot be boosted without limit.

All images linear RGB (1, 3, H, W) torch tensors, range z (1, 1, H, W) in metres, as in physics.py.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from physics import _bc, linear_to_srgb

# The recipe in use from Day 1b on: safe_restore(I, z, wat, **CURRENT_RECIPE). White balance as tuned in Day 1;
# 3 px smoothing patch (r=1) instead of 15 px, which removes the pink halo at object edges (eval_day1b.py).
DAY1_RECIPE = dict(wb_cap=16.0, wb_p=4, eps=1e-4, r=7, margin=3.0)
CURRENT_RECIPE = dict(DAY1_RECIPE, r=1)


def box(x, r):
    """Mean over a (2r+1)^2 window, reflect padding."""
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def noise_floor(I):
    """Noise std of the photo in sRGB units (where JPEG and 8-bit rounding live), from the median absolute 3x3
    high-pass residual (robust to edges). A channel crushed to a few levels hides its own noise (its residual is
    mostly 0), so every channel gets the largest channel's value. Never below 1/255."""
    s = linear_to_srgb(I)
    res = (s - box(s, 1)).flatten(2)
    sigma = 1.4826 * (res - res.median(-1, keepdim=True)[0]).abs().median(-1)[0] * math.sqrt(9 / 8)
    return sigma.amax(1, keepdim=True).clamp_min(1 / 255).expand(-1, 3)  # (1, 3)


def noise_linear(I, sigma):
    """Per-pixel noise std in linear units: the sRGB noise floor times the slope of the sRGB->linear curve."""
    s = linear_to_srgb(I)
    slope = torch.where(s <= 0.04045, torch.full_like(s, 1 / 12.92), 2.4 / 1.055 * ((s + 0.055) / 1.055) ** 1.4)
    return _bc(sigma) * slope


def fog_cap(I, z, B, sigma, nbins=10, q=0.01, margin=3.0):
    """Upper bound on the backscatter at each range: the q-th percentile of the photo in that range bin, plus
    `margin` noise stds (noise pulls the darkest pixels below the true fog level). Fog grows with range, so the
    bound at range z is the smallest bound at any range >= z."""
    zf, If = z.flatten().cpu().numpy(), I[0].flatten(1).cpu().numpy()
    edges = np.quantile(zf, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, zf, side="right") - 1, 0, nbins - 1)
    centres, dark = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 50:
            continue
        centres.append(np.median(zf[m]))
        dark.append(np.quantile(If[:, m], q, axis=1))
    dark = np.minimum.accumulate(np.array(dark)[::-1], axis=0)[::-1]  # reverse cumulative min over range
    cap = torch.from_numpy(np.stack([np.interp(zf, centres, dark[:, c]) for c in range(3)]).reshape(I.shape)).to(I)
    return torch.minimum(B, cap + margin * noise_linear(cap, sigma))


def guided_filter(p, g, r=7, eps=1e-4):
    """He et al. guided filter: p ~ a*g + b in every (2r+1)^2 patch; eps keeps a ~ 0 (plain average) in flat
    patches, where the guide's variance is only noise, so specks average out while edges follow the guide."""
    mg, mp = box(g, r), box(p, r)
    a = (box(g * p, r) - mg * mp) / (box(g * g, r) - mg * mg + eps)
    b = mp - a * mg
    return box(a, r) * g + box(b, r)


def trust(D, I, sigma, k=3.0):
    """0..1 per pixel and channel: 0 when the direct signal is within the noise (SNR <= 1), 1 when SNR >= k."""
    snr = D / noise_linear(I, sigma)
    return ((snr - 1) / (k - 1)).clamp(0, 1)


def smooth_weak(D, w, r=7, eps=1e-4):
    """Rebuild untrusted pixels of each channel from the most trusted channel (the guide) with a guided filter."""
    gi = int(w.mean((0, 2, 3)).argmax())
    g = D[:, gi:gi + 1].clamp_min(0)
    out = D.clone()
    for c in range(3):
        if c != gi:
            sm = guided_filter(D[:, c:c + 1], g, r, eps).clamp_min(0)
            out[:, c:c + 1] = w[:, c:c + 1] * D[:, c:c + 1] + (1 - w[:, c:c + 1]) * sm
    return out


def robust_white_balance(x, w, cap=8.0, p=1, min_frac=0.05):
    """Gray-world (p=1) / shades-of-gray (p>1) from pixels trusted in all channels and not clipped, falling back
    to all pixels when fewer than min_frac qualify; each channel's gain is limited to [1/cap, cap]."""
    m = (w.amin(1) > 0.5) & (x.amax(1) < 0.98)
    if m.float().mean() < min_frac:
        m = torch.ones_like(m)
    v = x.clamp_min(0).permute(1, 0, 2, 3)[:, m]  # (3, N)
    stat = v.pow(p).mean(1).pow(1 / p) + 1e-6
    gain = (stat.mean() / stat).clamp(1 / cap, cap)
    return x * gain[None, :, None, None]


def safe_restore(I, z, wat, max_gain=50.0, use_cap=True, use_smooth=True, wb="robust", wb_cap=8.0, wb_p=1,
                 r=7, eps=1e-4, margin=3.0):
    """Remove water like physics.remove_water, with the Day 1 fixes switchable for ablation. wb: "robust",
    "gray" (plain gray-world) or "E" (divide by the predicted sunlight colour, as physics.restore does).
    With use_cap=use_smooth=False and wb="gray" this equals gray_world(remove_water(...).clamp(0, 1.5))."""
    B = _bc(wat["B_inf"]) * (1 - torch.exp(-_bc(wat["beta_B"]) * z))
    diag = {"fog_over_signal": (I < B).float().mean((0, 2, 3)).tolist()}
    sigma = noise_floor(I)
    if use_cap:
        B = fog_cap(I, z, B, sigma, margin=margin)
    D = I - B
    diag["fog_over_signal_after"] = (D < 0).float().mean((0, 2, 3)).tolist()
    w = trust(D, I, sigma)
    diag["noise_floor_255"] = (sigma[0] * 255).tolist()
    diag["trusted_frac"] = (w > 0.5).float().mean((0, 2, 3)).tolist()
    if use_smooth:
        D = smooth_weak(D, w, r, eps)
    gain = torch.exp((_bc(wat["beta_D"]) * z).clamp(max=math.log(max_gain)))
    if wb == "E":
        return D * gain / _bc(wat["E"]), diag
    J = (D * gain).clamp(0, 1.5)
    if wb == "robust":
        J = robust_white_balance(J, w, wb_cap, wb_p)
    else:
        m = J.mean((2, 3), keepdim=True)
        J = J * (m.mean(1, keepdim=True) / (m + 1e-6))
    return J, diag


def speck_score(srgb):
    """Red specks: mean |red - 5x5 median of red| in 8-bit levels, on an (H, W, 3) sRGB numpy image."""
    import cv2
    r = (np.clip(srgb[..., 0], 0, 1) * 255).astype(np.uint8)
    return float(np.abs(r.astype(np.float32) - cv2.medianBlur(r, 5).astype(np.float32)).mean())
