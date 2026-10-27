"""Image-quality and depth metrics. Images are float numpy (H, W, 3) sRGB in [0, 1]."""
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(x, y):
    return peak_signal_noise_ratio(y, np.clip(x, 0, 1), data_range=1.0)


def ssim(x, y):
    return structural_similarity(y, np.clip(x, 0, 1), channel_axis=-1, data_range=1.0)


def ciede2000(x, y):
    return float(deltaE_ciede2000(rgb2lab(np.clip(x, 0, 1)), rgb2lab(y)).mean())


def full_reference(x, y):
    return {"psnr": psnr(x, y), "ssim": ssim(x, y), "de2000": ciede2000(x, y)}


def angular_error_deg(rgb):
    """Angle between an RGB triplet (linear) and the neutral axis (1,1,1); 0 = perfectly grey.
    The colour-chart metric used by Sea-thru (Akkaynak & Treibitz 2019) and SQUID (Berman 2020)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    cos = rgb.sum(-1) / (np.sqrt(3) * np.linalg.norm(rgb, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def uciqe(img):
    """UCIQE (Yang & Sowmya, TIP 2015): 0.4680*sigma_chroma + 0.2745*con_L + 0.2576*mu_sat."""
    lab = rgb2lab(np.clip(img, 0, 1))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    con_l = (np.percentile(L, 99) - np.percentile(L, 1)) / 100
    sat = chroma / np.sqrt(chroma ** 2 + L ** 2 + 1e-12)
    return 0.4680 * chroma.std() / 100 + 0.2745 * con_l + 0.2576 * sat.mean()


def depth_errors(pred, gt, align=None):
    """AbsRel, RMSE, delta<1.25 on valid pixels. align='scale' -> median scaling (relative models);
    align='affine-inv' -> least-squares scale+shift in inverse depth (for disparity outputs)."""
    m = np.isfinite(gt) & (gt > 0.1)
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    if align == "scale":
        p = p * np.median(g) / np.median(p)
    elif align == "affine-inv":
        A = np.stack([p, np.ones_like(p)], 1)
        s, t = np.linalg.lstsq(A, 1 / g, rcond=None)[0]
        p = 1 / np.clip(s * p + t, 1e-3, None)
    p = np.clip(p, 1e-3, None)
    return {"absrel": float(np.mean(np.abs(p - g) / g)), "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25))}
