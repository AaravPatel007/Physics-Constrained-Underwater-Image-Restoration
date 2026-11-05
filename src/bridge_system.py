"""BridgeNet (the older DAv2 + Sea-thru + NAFNet pipeline) and the BridgeNet/day3a hybrid as stress-test systems.

Reads the old project in place and writes nothing there. A photo where Sea-thru's fitting fails falls back to the raw
photo as stage 1, and is counted in FAILED, since dropping it would break the pairing with the other systems.
"""
import sys

sys.dont_write_bytecode = True
OLD = "/home/smeer/teamwork_projects/underwater_bridge_nn"
sys.path.append(OLD)

import cv2
import numpy as np
import torch
from src.models.bridge_net import BridgeNet  # noqa: E402
from src.pipeline.seathru_pipeline import process_seathru_dav2  # noqa: E402

from eval_real import t
from physics import linear_to_srgb, srgb_to_linear

_ck = torch.load(OLD + "/checkpoints/best_model.pth", map_location="cuda", weights_only=True)
NET = BridgeNet(in_channels=7, out_channels=3, width=32)
NET.load_state_dict(_ck["model_state_dict"])
NET = NET.cuda().eval()
FAILED = [0]
DAY3A = "TouchUp [day3a_touchup_global]"


def srgb(x):
    return np.clip(linear_to_srgb(x.float().clamp(0, 1.5))[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)


def bridge(raw):
    try:
        st, nd = process_seathru_dav2(raw, depth_hint=None, device="cuda")
    except Exception as e:
        FAILED[0] += 1
        print("sea-thru failed:", repr(e)[:80], "total", FAILED[0], flush=True)
        g = (raw * [.299, .587, .114]).sum(-1, keepdims=True).astype(np.float32)
        st, nd = raw, g
    st = np.clip(st, 0, 1).astype(np.float32)
    nd = nd if nd.ndim == 3 else nd[..., None]
    x = np.concatenate([raw, st, nd], -1).astype(np.float32)
    with torch.no_grad():
        o = NET(torch.from_numpy(x).permute(2, 0, 1)[None].cuda(), torch.from_numpy(st).permute(2, 0, 1)[None].cuda())
    return np.clip(o[0].permute(1, 2, 0).cpu().numpy(), 0, 1)


def trust(raw, b):
    """Weight for BridgeNet: local structure matches the photo, on real texture, without blown highlights."""
    gray = lambda s: (s * [.299, .587, .114]).sum(-1).astype(np.float32)
    box = lambda v: cv2.blur(v, (15, 15))
    x, y = gray(raw), gray(b)
    mx, my = box(x), box(y)
    vx, vy = np.maximum(box(x * x) - mx * mx, 0), np.maximum(box(y * y) - my * my, 0)
    corr = (box(x * y) - mx * my) / np.sqrt(vx * vy + 1e-6)
    w = np.clip((corr - .3) / .4, 0, 1) * np.clip(np.sqrt(vx) / .02, 0, 1) * (y < .93)
    return cv2.GaussianBlur(w.astype(np.float32), (0, 0), 6)


def system(I, z, outs):
    raw = srgb(I)
    b = bridge(raw)
    w = trust(raw, b)[..., None]
    h = w * b + (1 - w) * srgb(outs[DAY3A])
    return {"BridgeNet": srgb_to_linear(t(b)), "Hybrid": srgb_to_linear(t(h))}
