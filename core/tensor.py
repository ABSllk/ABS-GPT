from core.backend import xp, FLOAT_DTYPE
from contextlib import contextmanager

def unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
    pad_shape = (1,) * (grad.ndim - len(shape)) + tuple(shape)
    axes = tuple(i for i in range(len(pad_shape)) if pad_shape[i] == 1)
    return xp.sum(grad, axis=axes, keepdims=True).reshape(shape)


class Tensor:
    _grad_enabled = True

    @classmethod
    @contextmanager
    def no_grad(cls):
        prev = cls._grad_enabled
        cls._grad_enabled = False
        try:
            yield
        finally:
            cls._grad_enabled = prev

    def __init__(self, data, require_grad=False, pre=(), opr=""):
        self.data = xp.asarray(data, dtype=FLOAT_DTYPE)
        self.require_grad = require_grad and (Tensor._grad_enabled or len(pre) == 0)
        self.grad = xp.zeros_like(self.data) if self.require_grad and not pre else None
        self.pre = pre if self.require_grad else ()
        self.opr = opr

        self._backward = lambda: None

    def _set_backward(self, backward_fn):
        if self.require_grad:
            self._backward = backward_fn

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, opr={self.opr!r})"

    @staticmethod
    def _as_tensor(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    def __add__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, require_grad=self.require_grad or other.require_grad, pre=(self, other), opr="+")

        def _backward():
            if self.require_grad:
                self.grad += unbroadcast(out.grad, self.grad.shape)
            if other.require_grad:
                other.grad += unbroadcast(out.grad, other.grad.shape)

        out._set_backward(_backward)
        return out

    def __mul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, require_grad=self.require_grad or other.require_grad, pre=(self, other), opr="*")

        def _backward():
            if self.require_grad:
                self.grad += unbroadcast(out.grad * other.data, self.grad.shape)
            if other.require_grad:
                other.grad += unbroadcast(out.grad * self.data, other.grad.shape)

        out._set_backward(_backward)
        return out

    def __matmul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data @ other.data, require_grad=self.require_grad or other.require_grad, pre=(self, other), opr="@")

        def _backward():
            if self.require_grad:
                self.grad += unbroadcast(out.grad @ xp.swapaxes(other.data, -1, -2), self.grad.shape)

            if other.require_grad:
                other.grad += unbroadcast(xp.swapaxes(self.data, -1, -2) @ out.grad, other.grad.shape)

        out._set_backward(_backward)
        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, require_grad=self.require_grad, pre=(self,), opr="**")

        def _backward():
            if self.require_grad:
                self.grad += out.grad * power * (self.data ** (power - 1))

        out._set_backward(_backward)
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * Tensor(-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def relu(self):
        out = Tensor(xp.maximum(0, self.data), require_grad=self.require_grad, pre=(self,), opr="relu")

        def _backward():
            if self.require_grad:
                self.grad += out.grad * (self.data > 0).astype(FLOAT_DTYPE)

        out._set_backward(_backward)
        return out

    def silu(self):
        z = xp.exp(-xp.abs(self.data))
        sigmoid = xp.where(self.data >= 0, 1 / (1 + z), z / (1 + z)) #防溢出
        out = Tensor(self.data * sigmoid, require_grad=self.require_grad, pre=(self,), opr="silu")
        def _backward():
            if self.require_grad:
                self.grad += out.grad * (sigmoid * (1 + self.data * (1 - sigmoid)))

        out._set_backward(_backward)
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), require_grad=self.require_grad, pre=(self,), opr="sum")

        def _backward():
            if self.require_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = xp.expand_dims(grad, axis=axis)
                # Broadcasting adds the reduced gradient without allocating
                # an input-sized ones array and a second product array.
                self.grad += grad

        out._set_backward(_backward)
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        return s / (self.data.size // s.data.size)

    def exp(self):
        out = Tensor(xp.exp(self.data), require_grad=self.require_grad, pre=(self,), opr="exp")

        def _backward():
            if self.require_grad:
                self.grad += out.grad * out.data

        out._set_backward(_backward)
        return out

    def scatter_add_rows(self, row_ids, num_rows):
        out = Tensor(xp.zeros((num_rows, self.data.shape[1]), dtype=self.data.dtype), require_grad=self.require_grad, pre=(self,), opr="scatter_add_rows")

        def _backward():
            if self.require_grad:
                self.grad += out.grad[row_ids]

        out._set_backward(_backward)
        xp.add.at(out.data, row_ids, self.data)
        return out

    def log(self):
        out = Tensor(xp.log(self.data), require_grad=self.require_grad, pre=(self,), opr="log")

        def _backward():
            if self.require_grad:
                self.grad += out.grad / self.data

        out._set_backward(_backward)
        return out

    def minimum(self, axis=None, keepdims=False):
        out = Tensor(
            xp.minimum(self.data, other.data),
            require_grad=self.require_grad or other.require_grad,
            pre=(self, other),
            opr="minimum",
        )

        def _backward():
            equal = (self.data == other.data).astype(FLOAT_DTYPE) * 0.5
            if self.require_grad:
                weight = (self.data < other.data).astype(FLOAT_DTYPE) + equal
                self.grad += unbroadcast(out.grad * weight, self.grad.shape)
            if other.require_grad:
                weight = (self.data > other.data).astype(FLOAT_DTYPE) + equal
                other.grad += unbroadcast(out.grad * weight, other.grad.shape)

        out._set_backward(_backward)
        return out

    def clip(self, min_value, max_value):
        out = Tensor(xp.clip(self.data, min_value, max_value), require_grad=self.require_grad, pre=(self,), opr="clip")

        def _backward():
            if self.require_grad:
                mask = (self.data >= min_value) & (self.data <= max_value)
                self.grad += out.grad * mask

        out._set_backward(_backward)
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], require_grad=self.require_grad, pre=(self,), opr="getitem")
        
        def _backward():
            if self.require_grad:
                xp.add.at(self.grad, index, out.grad)

        out._set_backward(_backward)
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), require_grad=self.require_grad, pre=(self,), opr="reshape")

        def _backward():
            if self.require_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._set_backward(_backward)
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), require_grad=self.require_grad, pre=(self,), opr="transpose")

        def _backward():
            if self.require_grad:
                self.grad += out.grad.transpose(*(axes.index(i) for i in range(len(axes))))

        out._set_backward(_backward)
        return out

    def zero_grad(self):
        if self.require_grad:
            if self.grad is None:
                self.grad = xp.zeros_like(self.data)
            else:
                self.grad.fill(0)

    def backward(self, retain_graph=False):
        topo = []
        visited = set()

        # 拓扑排序，避免递归函数的自引用闭包保留整个图
        pending = [(self, False)]
        while pending:
            node, expanded = pending.pop()
            if expanded:
                topo.append(node)
                continue
            if node in visited:
                continue

            visited.add(node)
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.pre))

        # 中间梯度必须在显式保留图时重置，防止混入上一轮的梯度
        for node in topo:
            if node.pre and node.grad is not None:
                node.grad.fill(0)
        if self.grad is None:
            self.grad = xp.ones_like(self.data)
        else:
            self.grad.fill(1)

        # 释放遍历引用
        visited.clear()
        while topo:
            node = topo.pop()
            for child in node.pre:
                if child.require_grad and child.grad is None:
                    child.grad = xp.zeros_like(child.data)
            if node._backward:
                node._backward()
            if not retain_graph and node.pre:
                node._backward = None # 解除反向闭包保存的引用
                node.pre = () # 解除当前节点对输入节点的引用
                if node is not self:
                    node.grad = None # 释放用完的中间梯度引用


def checkpoint(module, x, **kwargs):
    if not Tensor._grad_enabled:
        return module(x, **kwargs)
    parameters = module.parameters()
    with Tensor.no_grad():
        result = module(x, **kwargs)
    multiple = isinstance(result, tuple)
    if multiple:
        shapes = [value.data.shape for value in result]
        sizes = [value.data.size for value in result]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        data = xp.concatenate([value.data.reshape(-1) for value in result])
    else:
        data = result.data
    out = Tensor(data, require_grad=x.require_grad or any(p.require_grad for p in parameters), pre=(x, *parameters), opr="checkpoint")

    def backward():
        enabled = Tensor._grad_enabled
        Tensor._grad_enabled = True
        try:
            fresh = Tensor(x.data, require_grad=x.require_grad)
            result = module(fresh, **kwargs)
            if multiple:
                loss = sum((value * Tensor(out.grad[start:end].reshape(shape))).sum()
                           for value, shape, start, end in zip(result, shapes, offsets[:-1], offsets[1:]))
            else:
                loss = (result * Tensor(out.grad)).sum()
            loss.backward()
            if x.require_grad:
                x.grad += fresh.grad
        finally:
            Tensor._grad_enabled = enabled

    out._set_backward(backward)
    if multiple:
        return tuple(out[start:end].reshape(shape) for shape, start, end in zip(shapes, offsets[:-1], offsets[1:]))
    return out
