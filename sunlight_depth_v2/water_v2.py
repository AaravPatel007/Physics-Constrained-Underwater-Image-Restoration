"""Synthetic water v2: the sunlight E is built from three separate physical quantities.

    E_c = E0_c * exp(-K_d,c * d_s)       normalised to max 1, floored at 0.03 (same convention as before)

  E0   surface daylight colour: a point on the CIE daylight locus (Judd et al. 1964; CIE 15:2004) at a correlated
       colour temperature (CCT) drawn uniformly in mired between 4000 K and 10000 K, converted to linear sRGB and
       white-balanced to D65 (6504 K -> [1, 1, 1], the old generator's implicit white). One effective degree of
       freedom (the CCT); the network still outputs 3 numbers, compared up to scale.
  K_d  diffuse attenuation of downwelling irradiance, from the Jerlov K_d table (Jerlov 1976, as tabulated in UWCNN
       Table I at 600 / 525 / 475 nm; this is physics.JERLOV_N, whose -ln is K_d). Same water body as the line of
       sight: same Jerlov position and same turbidity factor as beta_D. Two physical differences from beta_D:
         - its own per-channel wideband spread (same size as beta_D's, sigma 0.15): the RGB-effective coefficient
           depends on the light spectrum and the path, which differ between the downwelling path and the line of
           sight (Akkaynak et al. 2017, Eq. 9);
         - the sun angle: K_d ~ (a + b_b) / mu_w (Gordon 1989), with mu_w the cosine of the refracted solar zenith
           angle (sin theta_w = sin theta_s / 1.34). The table is taken as the overhead-sun value.
         theta_s ~ U(0, 60 deg) (daytime; the 1/mu_w form is an approximation for moderate zenith angles).
  d_s  depth below the surface, U(0.5, 12) m: exactly the old generator's d_v (same draw).

beta_D, beta_B, d_s and the random parts of B_inf are drawn exactly as in physics.sample_water (low_red = 0): the old
generator's kappa draw is still made (and ignored) so the random stream stays aligned, and the new quantities are
drawn after all the old ones. So with the same seed, every image gets the same water and depth as before; only
the sunlight (E0, K_d, E) and therefore B_inf (veiling light = albedo x the light at depth) change.

Returned keys: beta_D, beta_B, B_inf, E (as before) + E0, K_d, d_s, cct, theta_s.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from physics import COASTAL, JERLOV_BETA, MURKY_BETA, OCEANIC  # noqa: E402

JERLOV_KD = JERLOV_BETA      # -ln(Jerlov residual energy ratio per metre) = K_d of the Jerlov tables [1/m]
CCT_RANGE = (4000.0, 10000.0)  # K; sampled uniformly in 1/CCT (mired)
SUN_ZENITH_DEG = (0.0, 60.0)
N_WATER = 1.34               # refractive index of sea water (Snell's law for the refracted sun)
KD_SPREAD = 0.15             # per-channel log spread, the same size as beta_D's in physics.sample_water
DS_RANGE = (0.5, 12.0)       # m, the old generator's d_v
E_FLOOR = 0.03

# XYZ -> linear sRGB (IEC 61966-2-1)
_M = torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def daylight_xy(cct):
    """CIE daylight locus chromaticity (x_D, y_D) for 4000 K <= CCT <= 25000 K (CIE 15:2004)."""
    t = torch.as_tensor(cct, dtype=torch.float64)
    x_lo = -4.6070e9 / t ** 3 + 2.9678e6 / t ** 2 + 0.09911e3 / t + 0.244063
    x_hi = -2.0064e9 / t ** 3 + 1.9018e6 / t ** 2 + 0.24748e3 / t + 0.237040
    x = torch.where(t <= 7000, x_lo, x_hi)
    return x, -3.0 * x ** 2 + 2.870 * x - 0.275


def _xy_to_rgb(x, y):
    XYZ = torch.stack([x / y, torch.ones_like(x), (1 - x - y) / y], -1)
    return XYZ @ _M.double().T


_WHITE = _xy_to_rgb(*daylight_xy(6504.0))


def daylight_rgb(cct):
    """Surface daylight colour E0 in linear sRGB, white-balanced to 6504 K, normalised to max 1. (..., 3) float32."""
    rgb = (_xy_to_rgb(*daylight_xy(cct)) / _WHITE).clamp_min(1e-4)
    return (rgb / rgb.amax(-1, keepdim=True)).float()


def mu_w(theta_s_deg):
    """Cosine of the refracted solar zenith angle just below a flat surface."""
    s = torch.sin(torch.deg2rad(theta_s_deg)) / N_WATER
    return torch.sqrt(1 - s ** 2)


def sunlight_v2(E0, K_d, d_s, floor=E_FLOOR):
    """E = E0 exp(-K_d d_s), max-normalised, floored. Computed in log space (no underflow in murky water)."""
    logE = E0.clamp_min(1e-6).log() - K_d * d_s
    return torch.exp(logE - logE.amax(1, keepdim=True)).clamp_min(floor)


def sample_water_v2(n, device="cpu", gen=None, water_type=None, murky_keep=1.0, low_red=0.0):
    if low_red:
        raise ValueError("low_red is not part of the v2 generator (the v1 recipe uses low_red = 0)")

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen)

    tabs = [torch.tensor([JERLOV_BETA[k] for k in names]) for names in (OCEANIC, COASTAL)]

    def draw(m):  # physics.sample_water.draw, also returning the shared water (table x turbidity) before jitter
        branch = torch.rand(m, generator=gen) < 0.5
        pos = U(0, 4, m)
        lo, frac = pos.floor().long().clamp(max=3), pos - pos.floor().clamp(max=3)
        base = torch.where(branch[:, None],
                           tabs[0][lo] * (1 - frac[:, None]) + tabs[0][lo + 1] * frac[:, None],
                           tabs[1][lo] * (1 - frac[:, None]) + tabs[1][lo + 1] * frac[:, None])
        base = base * torch.exp(U(math.log(0.6), math.log(1.6), m, 1))   # overall turbidity (shared)
        bd = base * torch.exp(0.15 * torch.randn(m, 3, generator=gen))   # line-of-sight wideband spread
        bb = bd * torch.exp(U(math.log(0.5), math.log(2.0), m, 3))
        return bd, bb, base

    if water_type is not None:
        bd = torch.tensor(JERLOV_BETA[water_type]).expand(n, 3).clone()
        bb, base = bd.clone(), torch.tensor(JERLOV_KD[water_type]).expand(n, 3).clone()
    elif murky_keep >= 1.0:
        bd, bb, base = draw(n)
    else:
        bds, bbs, bas, got = [], [], [], 0
        while got < n:
            bd, bb, base = draw(2 * n)
            keep = (bd.mean(1) <= MURKY_BETA) | (torch.rand(2 * n, generator=gen) < murky_keep)
            bds.append(bd[keep]), bbs.append(bb[keep]), bas.append(base[keep])
            got += int(keep.sum())
        bd, bb, base = torch.cat(bds)[:n], torch.cat(bbs)[:n], torch.cat(bas)[:n]

    d_s = U(*DS_RANGE, n, 1)                  # = old d_v
    U(0.6, 1.0, n, 1)                         # old kappa draw: made and discarded, keeps the stream aligned
    col_jit = U(0.85, 1.15, n, 3)             # old B_inf colour jitter
    bright = U(0.15, 0.65, n, 1)              # old B_inf brightness
    # new quantities, drawn after all the old ones
    mired = U(1e6 / CCT_RANGE[1], 1e6 / CCT_RANGE[0], n, 1)
    cct = 1e6 / mired
    theta = U(*SUN_ZENITH_DEG, n, 1)
    kd_spread = torch.randn(n, 3, generator=gen)
    if water_type is not None:
        kd_spread = torch.zeros(n, 3)          # exact Jerlov type: no spread, as for beta_D
    K_d = base * torch.exp(KD_SPREAD * kd_spread) / mu_w(theta)
    E0 = daylight_rgb(cct[:, 0])
    E = sunlight_v2(E0, K_d, d_s)

    # B_inf exactly as physics.sample_water, but from the new light at depth E
    albedo = (bd.amin(1, keepdim=True) / bd) ** 0.5
    col = albedo * E * col_jit
    grey = torch.full((n, 1), 0.1)
    col = (1 - grey) * col / col.amax(1, keepdim=True) + grey
    binf = col / col.amax(1, keepdim=True) * bright
    out = {"beta_D": bd, "beta_B": bb, "B_inf": binf, "E": E, "E0": E0, "K_d": K_d, "d_s": d_s, "cct": cct,
           "theta_s": theta}
    return {k: v.to(device) for k, v in out.items()}
