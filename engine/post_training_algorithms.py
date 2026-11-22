from core.backend import xp
from core.tensor import Tensor
from engine.post_training import PreferenceTrainer, RolloutTrainer, pair_log_probs, reduce_completion_log_probs
from engine.post_training_loss import UnifiedLoss


def _grpo_terms(current, old, advantages, mask, reference=None):
    mask = xp.asarray(mask, dtype=bool)
    logp = current[mask]
    advantage = Tensor._as_tensor(advantages).data
    if advantage.ndim == 1:
        advantage = advantage[:, None]
    advantage = xp.broadcast_to(advantage, current.data.shape)[mask]
    ratio = (logp - Tensor._as_tensor(old).data[mask]).exp()
    log_ratio_to_ref = None if reference is None else logp - Tensor._as_tensor(reference).data[mask]
    return dict(prediction=log_ratio_to_ref, target=1, advantage=advantage, policy_ratio=ratio, log_prediction=True, return_metrics=True)


class GRPOTrainer(RolloutTrainer):
    policy_terms = staticmethod(_grpo_terms)

    def __init__(self, *args, standardize_advantages=True, **kwargs):
        self.standardize_advantages = standardize_advantages
        super().__init__(*args, **kwargs)

    def default_advantage(self, batch):
        rewards = xp.asarray(batch["rewards"], dtype=xp.float32).reshape(-1, self.group_size)
        centered = rewards - rewards.mean(axis=1, keepdims=True)
        if self.standardize_advantages:
            centered = centered / (centered.std(axis=1, keepdims=True) + 1e-8)
        return centered.reshape(-1)

    @property
    def uses_reference(self):
        return bool(self.loss_fn.weight)

    def _policy_inputs(self, model, batch):
        return self.policy_terms(**super()._policy_inputs(model, batch))

    def loss_normalizer(self, metrics):
        return metrics["responses"] * self.max_new_tokens

    def train_step(self, rows, step):
        metrics = super().train_step(rows, step)
        metrics["kl"] = metrics.pop("divergence_sum") / metrics["tokens"]
        return metrics


class GSPOTrainer(GRPOTrainer):
    @staticmethod
    def policy_terms(current, old, advantages, mask, reference=None):
        log_ratio = reduce_completion_log_probs(current - old, mask, normalize=True)
        return dict(advantage=advantages, policy_ratio=log_ratio.exp(), return_metrics=True)

    def loss_normalizer(self, metrics):
        return metrics["responses"]


class DAPOLossTrainer(GRPOTrainer):
    def loss_normalizer(self, metrics):
        return metrics["tokens"]


def _logistic_inputs(margin):
    log_probs = Tensor.cat((-(-margin).softplus(), -margin.softplus()), axis=-1)
    return dict(prediction=log_probs, target=(1, 0), log_prediction=True)


class DPOTrainer(PreferenceTrainer):
    uses_reference = True

    def __init__(self, model, optimizer, pad_id, beta=0.1, **kwargs):
        super().__init__(
            model, optimizer, pad_id, loss_fn=UnifiedLoss(power=1, class_axis=-1),
            preference_inputs=lambda margin: _logistic_inputs(beta * margin), **kwargs,
        )


class IPOTrainer(PreferenceTrainer):
    uses_reference = True

    def __init__(self, model, optimizer, pad_id, tau=0.1, **kwargs):
        super().__init__(
            model, optimizer, pad_id, loss_fn=UnifiedLoss(power=2, weight=2),
            preference_inputs=lambda margin: dict(prediction=margin, target=1 / (2 * tau)), **kwargs,
        )


class SimPOTrainer(PreferenceTrainer):
    def __init__(self, model, optimizer, pad_id, beta=2.0, gamma=0.5, **kwargs):
        super().__init__(
            model, optimizer, pad_id, loss_fn=UnifiedLoss(power=1, class_axis=-1),
            preference_inputs=lambda margin: _logistic_inputs(beta * margin - gamma),
            length_normalize=True, **kwargs,
        )


class ORPOTrainer(PreferenceTrainer):
    def __init__(self, model, optimizer, pad_id, weight=0.1, **kwargs):
        self.cross_entropy = UnifiedLoss(power=1, class_axis=-1)
        self.preference_weight = weight
        super().__init__(model, optimizer, pad_id, loss_fn=self._orpo_loss, length_normalize=True, **kwargs)

    def _loss_inputs(self, model, batch):
        return dict(logp=pair_log_probs(model, batch, normalize=self.length_normalize))

    def _orpo_loss(self, logp):
        log_not_p = (-logp.expm1()).clip(xp.finfo(logp.data.dtype).tiny, 1).log()
        odds = logp - log_not_p
        preference = self.cross_entropy(**_logistic_inputs(odds[:, :1] - odds[:, 1:]))
        chosen = Tensor.cat((logp[:, :1], log_not_p[:, :1]), axis=-1)
        nll = self.cross_entropy(prediction=chosen, target=(1, 0), log_prediction=True)
        return nll + self.preference_weight * preference
