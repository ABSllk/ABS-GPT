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
    """Compute L = lambda * D_p(t || u) - A * rho_bar before batch reduction.

    lambda is weight, p is power, t is target, and u is prediction (or
    exp(prediction) when log_prediction=True). A is advantage.

    D_p(t || u) = integral from t to u of (v - t) * v**(p - 2) dv.
    The implemented closed forms are
        D_0 = t/u - log(t/u) - 1
        D_1 = t * log(t/u) + u - t
        D_2 = (u - t)**2 / 2
    D_2 extends to real inputs. With class_axis set, D_1 sums over a
    normalized categorical distribution, where the linear terms cancel.

    rho_bar is policy_ratio rho when clip_epsilon is None. With epsilon
    set, rho_bar = min(rho, 1 + epsilon_high) for A > 0 and
    rho_bar = max(rho, 1 - epsilon_low) for A < 0. Pass a scalar for
    symmetric clipping or (epsilon_low, epsilon_high) for asymmetric clipping.
    For A = 0 the policy term is zero.
    """

    def __init__(self, power, weight=1.0, *, clip_epsilon=None, class_axis=None):
        self.power = power
        self.weight = weight
        self.clip_epsilon = clip_epsilon
        self.class_axis = class_axis

    def divergence(self, prediction, target, *, log_prediction=False):
        """Compute elementwise D_p(t || u) for a fixed target t."""
        t = Tensor._as_tensor(target).data
        p = self.power
        if p == 2:
            u = prediction.exp() if log_prediction else prediction
            return 0.5 * (u - t) ** 2

        zero_t = t == 0
        log_t = xp.log(xp.where(zero_t, 1, t))
        if p == 1:
            x = Tensor.where(zero_t, 0 if log_prediction else 1, prediction)
            log_u = x if log_prediction else x.log()
            loss = (Tensor(log_t) - log_u) * t
            if self.class_axis is not None:
                # The linear terms cancel for normalized categorical distributions.
                return loss.sum(axis=self.class_axis)
            u = prediction.exp() if log_prediction else prediction
            return u - t + loss

        if p == 0:
            zero_u = prediction.data == (-xp.inf if log_prediction else 0)
            x = Tensor.where(zero_t & zero_u, 0 if log_prediction else 1, prediction)
            log_u = x if log_prediction else x.log()
            return _expm1_minus_x(Tensor(xp.where(zero_t, -xp.inf, log_t)) - log_u)

    def __call__(self, prediction=None, target=None, *, advantage=0, policy_ratio=1, log_prediction=False, return_metrics=False):
        if self.weight:
            cost = self.divergence(prediction, target, log_prediction=log_prediction)
        else:
            cost = Tensor(0.) if prediction is None else prediction * 0
        if isinstance(advantage, Tensor):
            advantage = advantage.data
        epsilon = self.clip_epsilon
        if epsilon is not None:
            epsilon_low, epsilon_high = epsilon if isinstance(epsilon, (list, tuple)) else (epsilon, epsilon)
            policy_ratio = Tensor._as_tensor(policy_ratio)
            direction = xp.asarray(advantage)
            dtype = policy_ratio.data.dtype
            lower = xp.where(direction < 0, 1 - epsilon_low, 0).astype(dtype)
            upper = xp.where(direction > 0, 1 + epsilon_high, xp.inf).astype(dtype)
            policy_ratio = policy_ratio.clip(lower, upper)
        loss = self.weight * cost - policy_ratio * advantage
        if return_metrics:
            return loss, {"divergence_sum": float(cost.data.sum().item()) if self.weight else 0.0}
        return loss
