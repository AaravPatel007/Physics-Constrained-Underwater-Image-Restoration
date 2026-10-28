"""Underwater image formation (Akkaynak & Treibitz, CVPR 2018/2019) in batched torch.

    I_c = J_c * exp(-beta_D_c * z)  +  B_inf_c * (1 - exp(-beta_B_c * z))
          `---- direct signal ----'     `--------- backscatter ---------'

  beta_D : attenuation (extinction) coefficient of the direct signal, per colour channel [1/m]
  beta_B : backscatter coefficient, per colour channel [1/m]
  B_inf  : veiling light (colour of the water far away), per colour channel
  z      : camera-to-scene range [m]
  E      : colour of the sunlight reaching the scene (filtered by the water column above it);
           J above is the scene under that light, so restoration also divides by E (white balance)

All images here are *linear* RGB in [0, 1], tensors shaped (B, 3, H, W); depth (B, 1, H, W).
"""
import math

import torch

# Residual energy ratio N_lambda per metre for the ten Jerlov water types, as tabulated in
# UWCNN (Li, Anwar & Porikli, Pattern Recognition 2020, Table I).  beta = -ln(N).  Order: R, G, B.
JERLOV_N = {
    "I":   (0.805, 0.961, 0.982),
    "IA":  (0.804, 0.955, 0.975),
    "IB":  (0.830, 0.950, 0.968),
    "II":  (0.800, 0.925, 0.940),
    "III": (0.750, 0.885, 0.890),
    "1C":  (0.750, 0.885, 0.875),
    "3C":  (0.710, 0.820, 0.800),
    "5C":  (0.670, 0.730, 0.670),
    "7C":  (0.620, 0.610, 0.500),
    "9C":  (0.550, 0.460, 0.290),
}
OCEANIC = ["I", "IA", "IB", "II", "III"]
COASTAL = ["1C", "3C", "5C", "7C", "9C"]
JERLOV_BETA = {k: [-math.log(n) for n in v] for k, v in JERLOV_N.items()}


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


MURKY_BETA = 0.8  # mean beta_D [1/m] above which even the true coefficients cannot restore the image (~Jerlov 9C)


def sample_water(n, device="cpu", gen=None, water_type=None, murky_keep=1.0):
    """Sample n random, physically plausible waters.

    beta_D: pick a point on the oceanic or coastal Jerlov branch (continuous interpolation between
            neighbouring types), then a global turbidity factor and per-channel jitter.
    beta_B: differs from beta_D (the key point of the revised model): per-channel ratio in [0.5, 2].
    B_inf : scattering albedo x colour of sunlight after d_v metres of descent through the same
            water, mixed with a little grey, scaled to a random brightness.
    If water_type is given (e.g. "3C"), use exactly that Jerlov type with no jitter.
    murky_keep < 1 keeps only that fraction of waters with mean beta_D > MURKY_BETA (rejection
    sampling), so near-hopeless water is seen less often but not never.
    Returns dict of (n, 3) tensors.
    """
    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen)

    def draw(m):
        branch = torch.rand(m, generator=gen) < 0.5
        pos = U(0, 4, m)
        lo, frac = pos.floor().long().clamp(max=3), pos - pos.floor().clamp(max=3)
        tabs = [torch.tensor([JERLOV_BETA[k] for k in names]) for names in (OCEANIC, COASTAL)]
        bd = torch.where(branch[:, None],
                         tabs[0][lo] * (1 - frac[:, None]) + tabs[0][lo + 1] * frac[:, None],
                         tabs[1][lo] * (1 - frac[:, None]) + tabs[1][lo + 1] * frac[:, None])
        bd = bd * torch.exp(U(math.log(0.6), math.log(1.6), m, 1))      # overall turbidity
        bd = bd * torch.exp(0.15 * torch.randn(m, 3, generator=gen))     # per-channel jitter
        bb = bd * torch.exp(U(math.log(0.5), math.log(2.0), m, 3))
        return bd, bb

    if water_type is not None:
        bd = torch.tensor(JERLOV_BETA[water_type]).expand(n, 3).clone()
        bb = bd.clone()
    elif murky_keep >= 1.0:
        bd, bb = draw(n)
    else:
        bds, bbs, got = [], [], 0
        while got < n:
            bd, bb = draw(2 * n)
            keep = (bd.mean(1) <= MURKY_BETA) | (torch.rand(2 * n, generator=gen) < murky_keep)
            bds.append(bd[keep]), bbs.append(bb[keep])
            got += int(keep.sum())
        bd, bb = torch.cat(bds)[:n], torch.cat(bbs)[:n]

    # E: colour of the sunlight that reaches the scene after d_v metres of descent (normalised, max 1).
    # This is why a reef 1 m from the camera at 10 m depth is already blue-green.
    d_v = U(0.5, 12.0, n, 1)
    E = torch.exp(-bd * d_v * U(0.6, 1.0, n, 1))
    E = (E / E.amax(1, keepdim=True)).clamp_min(0.03)
    # B_inf ~ (b/c) * E: scattering albedo b/c is low where absorption dominates (red in ocean
    # water), approximated as (min beta / beta)^0.5.
    albedo = (bd.amin(1, keepdim=True) / bd) ** 0.5
    col = albedo * E * U(0.85, 1.15, n, 3)
    col = 0.9 * col / col.amax(1, keepdim=True) + 0.1
    binf = col / col.amax(1, keepdim=True) * U(0.15, 0.65, n, 1)
    return {"beta_D": bd.to(device), "beta_B": bb.to(device), "B_inf": binf.to(device), "E": E.to(device)}


def _bc(v):  # (B,3) -> (B,3,1,1)
    return v[:, :, None, None]


def add_water(J, z, w):
    """Forward model. J linear (B,3,H,W), z (B,1,H,W) metres. Returns I, direct, backscatter.
    J is lit by the depth-filtered sunlight E when w has one."""
    if "E" in w:
        J = J * _bc(w["E"])
    direct = J * torch.exp(-_bc(w["beta_D"]) * z)
    back = _bc(w["B_inf"]) * (1 - torch.exp(-_bc(w["beta_B"]) * z))
    return direct + back, direct, back


def remove_water(I, z, w, max_gain=50.0):
    """Analytic inversion: J = (I - B) * exp(beta_D z). Gain capped so noise in fully-attenuated
    regions is not amplified without bound."""
    back = _bc(w["B_inf"]) * (1 - torch.exp(-_bc(w["beta_B"]) * z))
    gain = torch.exp((_bc(w["beta_D"]) * z).clamp(max=math.log(max_gain)))
    return (I - back) * gain


def restore(I, z, w, max_gain=50.0):
    """Full restoration: remove the water along the line of sight, then the illuminant colour."""
    J = remove_water(I, z, w, max_gain)
    return J / _bc(w["E"]) if "E" in w else J


def gray_world(x, eps=1e-6):
    """Per-image gray-world white balance on linear RGB (B,3,H,W)."""
    m = x.mean((2, 3), keepdim=True)
    return x * (m.mean(1, keepdim=True) / (m + eps))
