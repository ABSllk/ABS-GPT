import os
import numpy as np

DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

if DEVICE == "cuda":
    import cupy as cp
    xp = cp
    FLOAT_DTYPE = cp.float32
else:
    xp = np
    FLOAT_DTYPE = np.float32

def to_cpu(array):
    if DEVICE == "cuda" and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array