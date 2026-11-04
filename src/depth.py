"""Monocular depth with Depth Anything V2 (Hugging Face transformers checkpoints).

variants: "metric-indoor"  (fine-tuned on Hypersim, metres, max ~20 m)
          "metric-outdoor" (fine-tuned on Virtual KITTI, metres, max ~80 m)
          "relative"       (affine-invariant inverse depth / disparity; needs scale + shift)
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from device import DEV

CKPT = {"metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "metric-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "relative": "depth-anything/Depth-Anything-V2-Small-hf"}
_cache = {}


def _load(variant, device):
    if variant not in _cache:
        proc = AutoImageProcessor.from_pretrained(CKPT[variant])
        model = AutoModelForDepthEstimation.from_pretrained(CKPT[variant]).to(device).eval()
        _cache[variant] = (proc, model)
    return _cache[variant]


@torch.no_grad()
def predict(images_uint8, variant="metric-indoor", device=DEV, bs=2):
    """images_uint8: list/array of (H, W, 3) sRGB uint8. Returns list of (H, W) float32 maps."""
    proc, model = _load(variant, device)
    out = []
    for b in range(0, len(images_uint8), bs):
        ims = [Image.fromarray(np.asarray(x)) for x in images_uint8[b:b + bs]]
        inp = proc(images=ims, return_tensors="pt").to(device)
        pred = model(**inp).predicted_depth
        for p, im in zip(pred, ims):
            p = torch.nn.functional.interpolate(p[None, None], size=im.size[::-1], mode="bilinear", align_corners=False)
            out.append(p[0, 0].float().cpu().numpy())
    return out
