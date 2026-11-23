import cupy as cp
import triton
import triton.language as tl

from core.tensor import Tensor

triton.runtime.driver.active.get_current_stream = lambda device: cp.cuda.get_current_stream().ptr

class Pointer:
    def __init__(self, array):
        self.array, self.dtype = array, array.dtype

    def data_ptr(self):
        return self.array.data.ptr

@triton.jit
def _softmax(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr,
             XR: tl.constexpr, XC: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        x = tl.load(X + row * XR + cols * XC, cols < COLS, other=-float("inf")).to(tl.float32)
        exp = tl.exp(x - tl.max(x, axis=0))
        tl.store(Y + row * COLS + cols, exp / tl.sum(exp, axis=0), cols < COLS)

@triton.jit
def _softmax_backward(Y, G, DX, COLS: tl.constexpr, GR: tl.constexpr, GC: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    y = tl.load(Y + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
    g = tl.load(G + row * GR + cols * GC, cols < COLS, other=0).to(tl.float32)
    dx = y * (g - tl.sum(y * g, axis=0))
    tl.store(DX + row * COLS + cols, dx, cols < COLS)

@triton.jit
def _matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
            BM: tl.constexpr, BNC: tl.constexpr, BKC: tl.constexpr,
            PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BNC)
    first_m = (pid // (8 * num_n)) * 8
    group_size = tl.minimum(num_m - first_m, 8)
    pid_m = first_m + (pid % (8 * num_n)) % group_size
    pid_n = (pid % (8 * num_n)) // group_size
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BNC + tl.arange(0, BNC)
    ks = tl.arange(0, BKC)
    acc = tl.zeros((BM, BNC), tl.float32)
    for block in range(tl.cdiv(K, BKC)):
        kval = block * BKC + ks
        a = tl.load(A + rows[:, None] * AM + kval[None, :] * AK,
                    (rows[:, None] < M) & (kval[None, :] < K), other=0)
        b = tl.load(B + kval[:, None] * BK + cols[None, :] * BN,
                    (kval[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(a, b, input_precision=PRECISION)
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))

def _check_array(x):
    assert x.ndim == 2 and min(x.shape) > 0
    assert x.dtype in (cp.float16, cp.float32)

def softmax(x):
    _check_array(x)
    rows, cols = x.shape
    y = cp.empty_like(x, order="C")
    block = triton.next_power_of_2(cols)
    warps = 4 if block <= 2048 else 8
    sm_count = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["multiProcessorCount"]
    _softmax[(min(rows, 4 * sm_count),)](Pointer(x), Pointer(y), rows, cols,
        *(s // x.itemsize for s in x.strides), block, num_warps=warps)
    return y

def matmul(a, b, precision="ieee"):
    _check_array(a)
    _check_array(b)
    assert a.shape[1] == b.shape[0] and a.dtype == b.dtype
    assert precision in ("ieee", "tf32", "tf32x3")
    m, k = a.shape
    n = b.shape[1]
    c = cp.empty((m, n), dtype=a.dtype)
    _matmul[(triton.cdiv(m, 32) * triton.cdiv(n, 64),)](
        Pointer(a), Pointer(b), Pointer(c), m, n, k,
        *(s // a.itemsize for s in a.strides), *(s // b.itemsize for s in b.strides),
        32, 64, 32, precision, num_warps=4, num_stages=3)
    return c

@triton.jit
def _grouped_linear(X, PARAMS, ENDS, Y, K: tl.constexpr, N: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    expert = tl.program_id(0)
    start = tl.load(ENDS + expert - 1, expert > 0, 0)
    end = tl.load(ENDS + expert)
    first_row = start + tl.program_id(1) * BM
    if first_row < end:
        params = PARAMS + expert * 5
        w = tl.load(params).to(tl.pointer_type(X.dtype.element_ty))
        bias = tl.load(params + 1).to(tl.pointer_type(X.dtype.element_ty))
        wk = tl.load(params + 2)
        wn = tl.load(params + 3)
        bs = tl.load(params + 4)
        rows = first_row + tl.arange(0, BM)
        cols = tl.program_id(2) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            ks = block * BK + tl.arange(0, BK)
            a = tl.load(X + rows[:, None] * K + ks[None, :],
                        (rows[:, None] < end) & (ks[None, :] < K), 0)
            b = tl.load(w + ks[:, None] * wk + cols[None, :] * wn,
                        (ks[:, None] < K) & (cols[None, :] < N), 0)
            acc += tl.dot(a, b, input_precision="tf32x3")
        acc += tl.load(bias + cols * bs, cols < N, 0)[None, :]
        tl.store(Y + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < end) & (cols[None, :] < N))


def grouped_linear(x, params, ends, max_rows, out_dim):
    x = cp.ascontiguousarray(x)
    out = cp.empty((x.shape[0], out_dim), dtype=x.dtype)
    grid = (ends.size, triton.cdiv(max_rows, 16), triton.cdiv(out_dim, 64))
    _grouped_linear[grid](Pointer(x), Pointer(params), Pointer(ends), Pointer(out), x.shape[1], out_dim, 16, 64, 32)
    return out

def softmax_tensor(x):
    y = softmax(x.data)
    out = Tensor(y, require_grad=x.require_grad, pre=(x,), opr="triton_softmax")

    def backward():
        g = out.grad
        dx = cp.empty_like(y)
        rows, cols = y.shape
        _softmax_backward[(rows,)](Pointer(y), Pointer(g), Pointer(dx), cols,
            *(s // g.itemsize for s in g.strides), triton.next_power_of_2(cols),
            num_warps=4 if cols <= 2048 else 8)
        x.grad += dx

    out._set_backward(backward)
    return out

def matmul_tensor(a, b, precision="ieee"):
    out = Tensor(matmul(a.data, b.data, precision), require_grad=a.require_grad or b.require_grad,
                 pre=(a, b), opr="triton_matmul")

    def backward():
        if a.require_grad:
            a.grad += matmul(out.grad, cp.ascontiguousarray(b.data.T), precision)
        if b.require_grad:
            b.grad += matmul(a.data.T, out.grad, precision)

    out._set_backward(backward)
    return out
