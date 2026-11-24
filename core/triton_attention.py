import cupy as cp
import triton
import triton.language as tl

from core.tensor import Tensor
from core.triton_ops import Pointer


@triton.jit
def _forward(
    Q, K, V, O, LSE, SEG,
    HQ: tl.constexpr, HKV: tl.constexpr,
    TQ, TK,
    D: tl.constexpr, PAST,
    QB, QH,
    QT, QD,
    KB, KH,
    KT, KD,
    VB, VH,
    VT, VD,
    BD: tl.constexpr, BLOCK: tl.constexpr,
    PRECISION: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr,
):
    batch = tl.program_id(1) // HQ
    q_head = tl.program_id(1) % HQ
    kv_head = q_head % HKV

    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    features = tl.arange(0, BD)
    
    q = tl.load(Q + batch * QB + q_head * QH + rows[:, None] * QT + features[None, :] * QD,
                (rows[:, None] < TQ) & (features[None, :] < D), other=0)

    m = tl.full((BLOCK,), -float("inf"), tl.float32)
    ell = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK, BD), tl.float32)

    limit = tl.minimum(TK, PAST + (tl.program_id(0) + 1) * BLOCK)

    for start in range(0, limit, BLOCK):
        keys = start + cols

        k = tl.load(K + batch * KB + kv_head * KH + keys[None, :] * KT + features[:, None] * KD, (keys[None, :] < TK) & (features[:, None] < D), other=0)
        v = tl.load(V + batch * VB + kv_head * VH + keys[:, None] * VT + features[None, :] * VD, (keys[:, None] < TK) & (features[None, :] < D), other=0)

        score = tl.dot(q, k, input_precision=PRECISION) * (D ** -0.5)
        valid = _attention_valid(SEG, batch, rows, keys, TQ, TK, PAST, HAS_SEGMENTS)
        score = tl.where(valid, score, -float("inf"))

        next_m = tl.maximum(m, tl.max(score, axis=1))
        safe_m = tl.where(next_m == -float("inf"), 0.0, next_m)

        p = tl.exp(score - safe_m[:, None])
        alpha = tl.exp(m - safe_m)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        ell = ell * alpha + tl.sum(p, axis=1)
        m = next_m

    demoninator = tl.where(ell > 0, ell, 1.0)
    row_base = (batch * HQ + q_head) * TQ + rows
    tl.store(O + row_base[:, None] * D + features[None, :], acc / demoninator[:, None],
             (rows[:, None] < TQ) & (features[None, :] < D))
    tl.store(LSE + row_base, tl.where(ell > 0, m + tl.log(demoninator), 0.0), rows < TQ)

@triton.jit
def _delta(O, G, DELTA, ROWS, D: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    features = tl.arange(0, BD)
    mask = (rows[:, None] < ROWS) & (features[None, :] < D)
    o = tl.load(O + rows[:, None] * D + features[None, :], mask, other=0).to(tl.float32)
    g = tl.load(G + rows[:, None] * D + features[None, :], mask, other=0).to(tl.float32)
    tl.store(DELTA + rows, tl.sum(o * g, axis=1), rows < ROWS)


@triton.jit
def _backward(
    Q, K, V, G, LSE, DELTA, DQ, DK, DV, SEG,
    HQ: tl.constexpr, HKV: tl.constexpr,
    TQ, TK,
    D: tl.constexpr, PAST,
    QB, QH,
    QT, QD,
    KB, KH,
    KT, KD,
    VB, VH,
    VT, VD,
    BD: tl.constexpr, BLOCK: tl.constexpr,
    COMPUTE_DQ: tl.constexpr, PRECISION: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr,
):
    tile = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = tl.arange(0, BLOCK)
    features = tl.arange(0, BD)

    scale = D ** -0.5
    acc = tl.zeros((BLOCK, BD), tl.float32)
    acc_v = tl.zeros((BLOCK, BD), tl.float32)

    if COMPUTE_DQ:
        batch, q_head = tl.program_id(1) // HQ, tl.program_id(1) % HQ
        kv_head = q_head % HKV
        qbase, kbase, vbase = batch * QB + q_head * QH, batch * KB + kv_head * KH, batch * VB + kv_head * VH
        row_base = (batch * HQ + q_head) * TQ

        q = tl.load(Q + qbase + tile[:, None] * QT + features[None, :] * QD,
                    (tile[:, None] < TQ) & (features[None, :] < D), other=0)
        g = tl.load(G + (row_base + tile[:, None]) * D + features[None, :],
                    (tile[:, None] < TQ) & (features[None, :] < D), other=0)
        lse = tl.load(LSE + row_base + tile, tile < TQ, other=0)
        delta = tl.load(DELTA + row_base + tile, tile < TQ, other=0)

        limit = tl.minimum(TK, PAST + (tl.program_id(0) + 1) * BLOCK)

        for start in range(0, limit, BLOCK):
            keys = start + inner
            k = tl.load(K + kbase + keys[:, None] * KT + features[None, :] * KD,
                        (keys[:, None] < TK) & (features[None, :] < D), other=0)
            v = tl.load(V + vbase + keys[:, None] * VT + features[None, :] * VD,
                        (keys[:, None] < TK) & (features[None, :] < D), other=0)

            score = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
            valid = _attention_valid(SEG, batch, tile, keys, TQ, TK, PAST, HAS_SEGMENTS)
            p = tl.exp(tl.where(valid, score - lse[:, None], -float("inf")))
            dp = tl.dot(g, tl.trans(v), input_precision=PRECISION)
            ds = p * (dp - delta[:, None]) * scale
            acc += tl.dot(ds.to(k.dtype), k, input_precision=PRECISION)
        tl.store(DQ + (row_base + tile[:, None]) * D + features[None, :], acc,
                 (tile[:, None] < TQ) & (features[None, :] < D))
    else:
        batch, kv_head = tl.program_id(1) // HKV, tl.program_id(1) % HKV
        kbase, vbase = batch * KB + kv_head * KH, batch * VB + kv_head * VH
        key_base = (batch * HKV + kv_head) * TK

        k = tl.load(K + kbase + tile[:, None] * KT + features[None, :] * KD,
                    (tile[:, None] < TK) & (features[None, :] < D), other=0)
        v = tl.load(V + vbase + tile[:, None] * VT + features[None, :] * VD,
                    (tile[:, None] < TK) & (features[None, :] < D), other=0)
        first = tl.maximum(0, tl.program_id(0) * BLOCK - PAST)
        for group in range(HQ // HKV):
            q_head = group * HKV + kv_head
            qbase = batch * QB + q_head * QH
            row_base = (batch * HQ + q_head) * TQ
            for start in range(first, TQ, BLOCK):
                rows = start + inner
                q = tl.load(Q + qbase + rows[:, None] * QT + features[None, :] * QD,
                            (rows[:, None] < TQ) & (features[None, :] < D), other=0)
                g = tl.load(G + (row_base + rows[:, None]) * D + features[None, :],
                            (rows[:, None] < TQ) & (features[None, :] < D), other=0)
                lse = tl.load(LSE + row_base + rows, rows < TQ, other=0)
                delta = tl.load(DELTA + row_base + rows, rows < TQ, other=0)
                score = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
                valid = _attention_valid(SEG, batch, rows, tile, TQ, TK, PAST, HAS_SEGMENTS)
                p = tl.exp(tl.where(valid, score - lse[:, None], -float("inf")))
                dp = tl.dot(g, tl.trans(v), input_precision=PRECISION)
                ds = p * (dp - delta[:, None]) * scale
                acc += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=PRECISION)
                acc_v += tl.dot(tl.trans(p.to(g.dtype)), g, input_precision=PRECISION)

        tl.store(DK + (key_base + tile[:, None]) * D + features[None, :], acc,
                 (tile[:, None] < TK) & (features[None, :] < D))
        tl.store(DV + (key_base + tile[:, None]) * D + features[None, :], acc_v,
                 (tile[:, None] < TK) & (features[None, :] < D))


def _settings(q, k, v, past_len, precision):
    bq, hq, tq, dq = q.shape
    bk, hkv, tk, dk = k.shape
    assert k.shape == v.shape and bk == bq and dk == dq
    assert q.dtype == k.dtype == v.dtype and q.dtype in (cp.float16, cp.float32)
    assert min(bq, hq, tq, dq, k.shape[-2]) > 0 and dq <= 256 and past_len >= 0
    assert precision in ("ieee", "tf32", "tf32x3")
    strides = tuple(s // a.itemsize for a in (q, k, v) for s in a.strides)
    return (hq, hkv, tq, tk, dq, past_len, *strides, max(32, triton.next_power_of_2(dq)))


def fused_forward(q, k, v, past_len=0, precision="ieee", segment_ids=None):
    segments = _prepare_segments(segment_ids, q, k, past_len)
    settings = _settings(q, k, v, past_len, precision)
    output = cp.empty(q.shape, dtype=q.dtype)
    lse = cp.empty(q.shape[:-1] + (1,), dtype=cp.float32)
    block = 16 if q.shape[-1] > 128 else 32
    _forward[(triton.cdiv(q.shape[-2], block), q.shape[0] * q.shape[1])](
        *(Pointer(a) for a in (q, k, v, output, lse, segments if segments is not None else q)), *settings, block, precision, segments is not None, num_warps=4, num_stages=1 if q.shape[-1] > 128 else 2)
    return output, lse


def fused_backward(q, k, v, output, lse, grad, past_len=0, precision="ieee", segment_ids=None):
    segments = _prepare_segments(segment_ids, q, k, past_len)
    settings = _settings(q, k, v, past_len, precision)
    grad = cp.ascontiguousarray(grad)
    assert grad.shape == output.shape == q.shape and grad.dtype == q.dtype
    rows, d = q.size // q.shape[-1], q.shape[-1]
    delta = cp.empty(q.shape[:-1] + (1,), dtype=cp.float32)
    _delta[(triton.cdiv(rows, 32),)](Pointer(output), Pointer(grad), Pointer(delta),
        rows, d, settings[-1], num_warps=4)
    dq, dk, dv = (cp.empty(a.shape, dtype=a.dtype) for a in (q, k, v))
    args = tuple(Pointer(a) for a in (q, k, v, grad, lse, delta, dq, dk, dv, segments if segments is not None else q))
    block = 16 if d > 128 else 32
    for compute_dq, length, heads in ((True, q.shape[-2], q.shape[1]), (False, k.shape[-2], k.shape[1])):
        _backward[(triton.cdiv(length, block), q.shape[0] * heads)](
            *args, *settings, block, compute_dq, precision, segments is not None,
            num_warps=4, num_stages=1 if d > 128 else 2)
    return dq, dk, dv


def tiled_attention(Q, K, V, past_len=0, precision="ieee", segment_ids=None):
    segments = _prepare_segments(segment_ids, Q, K, past_len)
    output, lse = fused_forward(Q.data, K.data, V.data, past_len, precision, segments)
    out = Tensor(output, require_grad=Q.require_grad or K.require_grad or V.require_grad,
                 pre=(Q, K, V), opr="triton_attention")

    def backward():
        grads = fused_backward(Q.data, K.data, V.data, output, lse, out.grad, past_len, precision, segments)
        for tensor, grad in zip((Q, K, V), grads):
            tensor.accumulate_grad(grad)

    out._set_backward(backward)
    return out

def _prepare_segments(segment_ids, q, k, past_len):
    if segment_ids is None:
        return None
    segments = cp.asarray(segment_ids)
    return cp.ascontiguousarray(segments, dtype=cp.int64)

@triton.jit
def _attention_valid(
    SEG, batch, rows, keys, 
    TQ, TK, PAST, HAS_SEGMENTS: tl.constexpr
):
    valid = (rows[:, None] < TQ) & (keys[None, :] < TK) & (keys[None, :] <= PAST + rows[:, None])
    if HAS_SEGMENTS:
        query_segments = tl.load(SEG + batch * TQ + rows, rows < TQ, other=-1)
        key_segments = tl.load(SEG + batch * TK + keys, keys < TK, other=-1)
        valid = valid & (query_segments[:, None] >= 0) & (query_segments[:, None] == key_segments[None, :])
    return valid