"""Sunlight from depth: E_c(d_s) = E0_c * exp(-K_d,c * d_s), and the ground-truth depth labels for it.

Two parts:

1. sunlight_from_depth(): the physics layer of the new model. Turns one depth below the surface d_s [m] and the
   diffuse attenuation K_d [1/m] per channel into the sunlight colour E, in exactly the convention the rest of the
   pipeline uses for E (brightest channel = 1, floor 0.03; see physics.sample_water and model.BetaNet).

2. sample_water_sun(): a copy of physics.sample_water that makes the *same random draws in the same order* and
   additionally returns the two numbers the original throws away: the sunlight depth d_v and the factor kappa that
   the generator uses as K_d = kappa * beta_D. So the training, validation and test waters (all drawn from seeded
   generators) are identical to the baseline's, and we get their depth labels for free. verify_rng.py checks this.

Units: depths in metres, K_d and beta in 1/m, E dimensionless (relative colour).

Option A (chosen 3 Nov 2026, see README.md): K_d,c = KAPPA0 * beta_D,c with the fixed KAPPA0 = 0.8, the midpoint of
the generator's kappa ~ U(0.6, 1.0). This is the relation the synthetic generator itself uses (with a random kappa
per water); it is a modelling assumption of that generator, not a measured property of sea water.
"""
import math

import torch

from physics import COASTAL, JERLOV_BETA, MURKY_BETA, OCEANIC

KAPPA0 = 0.8          # K_d / beta_D used by the new model (generator: kappa ~ U(0.6, 1.0) per water)
KAPPA_RANGE = (0.6, 1.0)
DV_RANGE = (0.5, 12.0)  # generator's sunlight depth d_v [m] (low_red = 0, as in baseline_v1)
D_MAX = 15.0          # upper bound of the predicted depth: 12 m * (1.0 / 0.8) = largest depth that reproduces E
E_FLOOR = 0.03        # same floor as the generator and BetaNet's E output


def sunlight_from_depth(K_d, d_s, E0=None, floor=E_FLOOR):
    """E_c = E0_c exp(-K_d,c d_s), normalised so the brightest channel is 1, floored at `floor`.

    K_d (B,3) [1/m], d_s (B,1) [m], E0 (3,) or None (= white, as in the generator). Computed in log space and
    shifted by the per-image maximum before exp(), so deep / murky water cannot underflow. Returns (B,3)."""
    logE = -K_d * d_s
    if E0 is not None:
        logE = logE + torch.as_tensor(E0, dtype=logE.dtype, device=logE.device).log()
    E = torch.exp(logE - logE.amax(1, keepdim=True))
    return E.clamp_min(floor)


def sample_water_sun(n, device="cpu", gen=None, water_type=None, murky_keep=1.0, low_red=0.0):
    """physics.sample_water, line for line (same draws, same order, same results), plus the hidden labels.

    Returns the same dict as physics.sample_water and, in addition:
      d_s   : (n,1) sunlight depth below the surface d_v [m]
      kappa : (n,1) K_d / beta_D of this water
      K_d   : (n,3) the generator's diffuse attenuation, kappa * beta_D [1/m]
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
        bd = bd * torch.exp(U(math.log(0.6), math.log(1.6), m, 1))
        bd = bd * torch.exp(0.15 * torch.randn(m, 3, generator=gen))
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

    d_v = U(0.5, 12.0, n, 1)
    deep = torch.zeros(n, 1, dtype=torch.bool)
    if low_red > 0:
        deep = torch.rand(n, 1, generator=gen) < low_red
        d_v = torch.where(deep, U(12.0, 30.0, n, 1), d_v)
    kappa = U(0.6, 1.0, n, 1)                      # the original draws this inline: exp(-bd * d_v * U(0.6, 1.0))
    E = torch.exp(-bd * d_v * kappa)
    E = E / E.amax(1, keepdim=True)
    E = torch.where(deep, E.clamp_min(0.002), E.clamp_min(0.03))
    albedo = (bd.amin(1, keepdim=True) / bd) ** 0.5
    col = albedo * E * U(0.85, 1.15, n, 3)
    grey = torch.full((n, 1), 0.1)
    if low_red > 0:
        grey = torch.where(deep, torch.exp(U(math.log(0.002), math.log(0.1), n, 1)), grey)
    col = (1 - grey) * col / col.amax(1, keepdim=True) + grey
    binf = col / col.amax(1, keepdim=True) * U(0.15, 0.65, n, 1)
    return {"beta_D": bd.to(device), "beta_B": bb.to(device), "B_inf": binf.to(device), "E": E.to(device),
            "d_s": d_v.to(device), "kappa": kappa.to(device), "K_d": (kappa * bd).to(device)}


WATER_KEYS = ("beta_D", "beta_B", "B_inf", "E")


def coefs_only(w):
    """The four keys the original pipeline knows about (what add_water / restore / coef_loss read)."""
    return {k: w[k] for k in WATER_KEYS}
