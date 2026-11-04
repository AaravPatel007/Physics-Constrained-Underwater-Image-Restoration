"""Compute device: CUDA (the Linux laptop all earlier results came from), else Apple MPS, else CPU."""
import torch

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def autocast():
    # bf16 autocast as before on CUDA; MPS/CPU run in fp32 (bf16 autocast there is slow or unsupported)
    return torch.autocast("cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16, enabled=DEV == "cuda")
