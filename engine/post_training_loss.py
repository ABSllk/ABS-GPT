from math import inf, log

from core.backend import xp
from core.tensor import Tensor


def _expm1_minus_x(z):
    """Evaluate expm1(z) - z accurately near zero."""
    small = xp.abs(z.data) < 0.05
    x = Tensor.where(small, z, 0)
    series = x ** 2 * (0.5 + x * (1 / 6 + x * (1 / 24 + x / 120)))
    linear = z.clip(-xp.inf, xp.finfo(z.data.dtype).max)
    return Tensor.where(small, series, z.expm1() - linear)


class UnifiedLoss:
    """Evaluate Phi_n(z) = n * log(1 + exp(z) / n) - z.

    The closed forms at n = 0, 1, and infinity are -z, softplus(-z),
    and exp(z) - z. An offset of 1 centers the last case for sampled KL.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, z, *, offset=0):
        n = self.n
        if n == 0:
            loss = -z
        elif n == 1:
            loss = (-z).softplus()
        elif n == inf:
            loss = _expm1_minus_x(z)
            offset -= 1
        else:
            loss = n * (z - log(n)).softplus() - z
        return loss - offset if offset else loss
