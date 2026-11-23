from core.backend import xp
from engine.post_training import PreferenceTrainer, RolloutTrainer, pair_log_probs, reduce_completion_log_probs
from engine.post_training_loss import UnifiedLoss


class GRPOTrainer(RolloutTrainer):
    def __init__(self, model, optimizer, tokenizer, reward_fn, *, standardize_advantages=True, **kwargs):
        self.standardize_advantages = standardize_advantages
        super().__init__(model, optimizer, tokenizer, None, reward_fn, **kwargs)

    def default_advantage(self, batch):
        rewards = xp.asarray(batch["rewards"], dtype=xp.float32).reshape(-1, self.group_size)
        centered = rewards - rewards.mean(axis=1, keepdims=True)
        if self.standardize_advantages:
            centered = centered / (centered.std(axis=1, keepdims=True) + 1e-8)
        return centered.reshape(-1)

    def loss_normalizer(self, metrics):
        return metrics["responses"] * self.max_new_tokens

class GSPOTrainer(GRPOTrainer):
    @staticmethod
    def policy_terms(current, old, advantages, mask, reference=None):
        log_ratio = reduce_completion_log_probs(current - old, mask, normalize=True)
        return log_ratio.exp(), advantages, None

    def loss_normalizer(self, metrics):
        return metrics["responses"]

class DAPOLossTrainer(GRPOTrainer):
    def loss_normalizer(self, metrics):
        return metrics["tokens"]

class DPOTrainer(PreferenceTrainer):
    uses_reference = True

    def __init__(self, model, optimizer, pad_id, beta=0.1, **kwargs):
        super().__init__(
            model, 
            optimizer, 
            pad_id, 
            loss_fn=UnifiedLoss(1),
            preference_inputs=lambda margin: dict(z=beta * margin), 
            **kwargs
        )

class IPOTrainer(PreferenceTrainer):
    uses_reference = True

    def __init__(self, model, optimizer, pad_id, tau=0.1, **kwargs):
        target = 1 / (2 * tau)
        super().__init__(
            model, 
            optimizer, 
            pad_id, 
            loss_fn=lambda error: error ** 2,
            preference_inputs=lambda margin: dict(error=margin - target), 
            **kwargs
        )

class SimPOTrainer(PreferenceTrainer):
    def __init__(self, model, optimizer, pad_id, beta=2.0, gamma=0.5, **kwargs):
        super().__init__(
            model, 
            optimizer, 
            pad_id, 
            loss_fn=UnifiedLoss(1),
            preference_inputs=lambda margin: dict(z=beta * margin - gamma),
            length_normalize=True, 
            **kwargs
        )

class ORPOTrainer(PreferenceTrainer):
    def __init__(self, model, optimizer, pad_id, weight=0.1, **kwargs):
        self.nll = UnifiedLoss(0)
        self.preference_loss = UnifiedLoss(1)
        self.preference_weight = weight
        super().__init__(
            model, 
            optimizer, 
            pad_id, 
            loss_fn=self._orpo_loss, 
            length_normalize=True, 
            **kwargs
        )

    def _loss_inputs(self, model, batch):
        return dict(logp=pair_log_probs(model, batch, normalize=self.length_normalize))

    def _orpo_loss(self, logp):
        log_not_p = (-logp.expm1()).clip(xp.finfo(logp.data.dtype).tiny, 1).log()
        odds = logp - log_not_p
        preference = self.preference_loss(odds[:, :1] - odds[:, 1:])
        return self.nll(logp[:, :1]) + self.preference_weight * preference
