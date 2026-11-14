from core.backend import xp
from core.tensor import Tensor
from functools import lru_cache
from pathlib import Path
import numpy as np
import cupy as cp

FORWARD_SOURCE = Path(__file__).with_name("attention_forward.cu").read_text()
BACKWARD_SOURCE = Path(__file__).with_name("attention_backward.cu").read_text()

@lru_cache(None)
def forward_kernel(d, device_id):
    shared_limit = cp.cuda.runtime.getDeviceProperties(device_id)["sharedMemPerBlock"]
    key_tile = min(32, shared_limit // (2 * d * 4))
    assert key_tile > 0
    code = f"#define HEAD_DIM {d}\n#define KEY_TILE {key_tile}\n" + FORWARD_SOURCE
    return cp.RawKernel(code, "attention_forward", options=("--std=c++11",))

@lru_cache(None)
def backward_gradient_kernel(d, device_id):
    shared_limit = cp.cuda.runtime.getDeviceProperties(device_id)["sharedMemPerBlock"]

    key_tile = min(32, shared_limit // (2 * d * 4))
    assert key_tile > 0

    code = f"#define HEAD_DIM {d}\n#define KEY_TILE {key_tile}\n" + BACKWARD_SOURCE
    module = cp.RawModule(code=code, options=("--std=c++11",))

    return (
        module.get_function("attention_delta"),
        module.get_function("attention_dq"),
        module.get_function("attention_dkdv"),
    )

def fused_forward(q, k, v, past_len=0):
    B, H, Tq, d = q.shape
    Tk = k.shape[-2]
    assert d > 0 and k.shape == v.shape and k.shape[:2] == (B, H) and k.shape[-1] == d
    assert q.dtype == k.dtype == v.dtype == cp.float32
    output = cp.empty(q.shape, dtype=q.dtype)
    lse = cp.empty(q.shape[:-1] + (1,), dtype=q.dtype)
    strides = tuple(np.int64(s // a.itemsize) for a in (q, k, v) for s in a.strides)
    args = (q, k, v, output, lse, np.int32(H), np.int32(Tq), np.int32(Tk), np.int32(past_len)) + strides
    kernel = forward_kernel(d, cp.cuda.Device().id)
    kernel(((Tq + 7) // 8, B * H), (256,), args)
    return output, lse

def fused_backward(q, k, v, output, lse, grad, past_len=0):
    B, H, Tq, d = q.shape
    Tk = k.shape[-2]
    rows = B * H * Tq

    delta = cp.empty((B, H, Tq, 1), dtype=cp.float32)
    delta_kernel, dq_kernel, dkdv_kernel = backward_gradient_kernel(
        d, cp.cuda.Device().id
    )

    grad, output, lse = map(cp.ascontiguousarray, (grad, output, lse))

    delta_kernel(
        ((rows + 7) // 8,),
        (256,),
        (output, grad, delta, np.int32(rows), np.int32(d))
    )

    dq = cp.empty(q.shape, dtype=q.dtype)
    dk = cp.empty(k.shape, dtype=k.dtype)
    dv = cp.empty(v.shape, dtype=v.dtype)

    strides = tuple(
        np.int64(s // a.itemsize)
        for a in (q, k, v)
        for s in a.strides
    )

    dims = (
        np.int32(H), np.int32(Tq),
        np.int32(Tk), np.int32(past_len),
    )
    common = (q, k, v, grad, lse, delta)

    dq_kernel(
        ((Tq + 7) // 8, B * H),
        (256,),
        common + (dq,) + dims + strides,
    )

    dkdv_kernel(
        ((Tk + 7) // 8, B * H),
        (256,),
        common + (dk, dv) + dims + strides,
    )

    return dq, dk, dv

def tiled_attention(Q, K, V, past_len=0, block_size=32):
    """Q: [B,H,Tq,d], K/V: [B,H,Tk,d]; causal, no padding/dropout."""
    q, k, v = Q.data, K.data, V.data
    scale = q.shape[-1] ** -0.5

    def scores(qi, kj, i, j):
        query_pos = past_len + xp.arange(i, i + qi.shape[-2])[:, None]
        key_pos = xp.arange(j, j + kj.shape[-2])[None, :]
        s = (qi @ kj.swapaxes(-1, -2)) * scale
        return xp.where(key_pos > query_pos, -xp.inf, s)

    output, lse = fused_forward(q, k, v, past_len)

    out = Tensor(
        output, 
        require_grad=Q.require_grad or K.require_grad or V.require_grad, 
        pre=(Q, K, V), 
        opr="tiled_attention",
    )

    def _backward():
        dq, dk, dv = fused_backward(q, k, v, output, lse, out.grad, past_len)
        for tensor, grad in ((Q, dq), (K, dk), (V, dv)):
            if tensor.require_grad:
                tensor.grad += grad

    out._set_backward(_backward)
    return out
