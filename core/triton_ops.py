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
