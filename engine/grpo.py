from core.backend import xp
from core.nn import LogSoftmax
from core.tensor import Tensor

from engine.generator import Generator
from engine.trainer import BaseTrainer
from dataset.language_model_loader import render_prompt

def token_log_probs(logits, targets):
    batch, length, vocab = logits.data.shape
    targets = xp.asarray(targets, dtype=xp.int64).reshape(-1)

    values = LogSoftmax()(logits.reshape(-1, vocab))
    selected = values[xp.arange(targets.size), targets]

    return selected.reshape(batch, length)

def group_advantages(rewards, group_size):
    rewards = xp.asarray(rewards, dtype=xp.float32).reshape(-1, group_size)
    advantages = rewards - rewards.mean(axis=1, keepdims=True)
    advantages = advantages / (advantages.std(axis=1, keepdims=True) + 1e-8)
    return advantages.reshape(-1)

class GRPOLoss:
    def __init__(self, epsilon=0.2, beta=0.01):
        self.epsilon = epsilon
        self.beta = beta
    
    def __call__(self, current, old, advantages, mask, normalizer, reference=None):
        mask = xp.asarray(mask, dtype=bool)

        v = current[mask]
        old_v = Tensor(xp.asarray(old)[mask])

        advantages = xp.broadcast_to(xp.asarray(advantages)[:, None], current.data.shape)[mask]

        ratio = (v - old_v).exp()
        unclipped = ratio * advantages
        clipped = ratio.clip(1 - self.epsilon,1 + self.epsilon) * advantages

        loss = -unclipped.minimum(clipped)
        metrics = {"kl_sum": 0.0}

        if self.beta:
            difference = Tensor(xp.asarray(reference)[mask]) - v
            kl = difference.exp() - 1 - difference
            loss += self.beta * kl
            metrics["kl_sum"] = float(kl.data.sum().item())

        return loss.sum() / normalizer, metrics

class GRPOTrainer(BaseTrainer):
    def __init__(
        self, model, optimizer, tokenizer,
        loss_fn, reward_fn, 
        context_size=512, group_size=4, max_new_tokens=128, micro_batch_size=1, 
        scheduler=None, reference_fn=None, seed=42, max_grad_norm=1.0
    ):
        super().__init__(model, optimizer, max_grad_norm, micro_batch_size, scheduler)
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.context_size = context_size
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.reference_fn = reference_fn
        self.seed = seed

    def collect_rollouts(self, rows, seed):
        generator = Generator(self.model, self.context_size, seed=seed)
        rollouts = []
        for row in rows:
            prefix = [self.tokenizer.bos_id, *self.tokenizer.encode(render_prompt(row["prompt"]))]
            for _ in range(self.group_size):
                result = generator.generate(
                    prefix, 
                    self.tokenizer.eos_id, 
                    max_new_tokens=self.max_new_tokens, 
                    temperature=1.0, 
                    top_p=1.0
                )
                rollouts.append({
                    **row,
                    "prompt_ids": prefix,
                    "completion_ids": result["token_ids"],
                    "completion": self.tokenizer.decode(result["token_ids"], errors="replace"),
                    "finish_reason": result["finish_reason"],
                })
        return rollouts

    def log_probs(self, batch, model=None):
        model = self.model if model is None else model
        result = xp.empty(batch["targets"].shape, dtype=xp.float32)
        micro = self.micro_batch_size or len(result)
        with Tensor.no_grad():
            for start in range(0, len(result), micro):
                chunk = slice(start, start + micro)
                logits = model(batch["inputs"][chunk], token_mask=batch["token_mask"][chunk])
                result[chunk] = token_log_probs(logits, batch["targets"][chunk]).data
        return result

    def train_step(self, rows, step):
        rollouts = self.collect_rollouts(rows, self.seed + step)
        batch = self.collate_rollouts(rollouts, self.tokenizer.pad_id)
        rewards = xp.asarray(self.reward_fn(rollouts)["rewards"], dtype=xp.float32)
        advantages = group_advantages(rewards, self.group_size)
        old = self.log_probs(batch)
        reference = None
        if self.loss_fn.beta:
            with Tensor.no_grad():
                reference = self.reference_fn(batch)

        self.zero_grad()
        normalizer = len(rollouts) * self.max_new_tokens
        tokens = int(batch["loss_mask"].sum())
        micro = self.micro_batch_size or len(rollouts)
        total = 0.0
        kl_sum = 0.0
        for start in range(0, len(rollouts), micro):
            chunk = slice(start, start + micro)
            logits = self.model(batch["inputs"][chunk], token_mask=batch["token_mask"][chunk])
            current = token_log_probs(logits, batch["targets"][chunk])
            loss, metrics = self.loss_fn(
                current, 
                old[chunk], 
                advantages[chunk], 
                batch["loss_mask"][chunk], 
                normalizer, 
                reference=None if reference is None else reference[chunk]
            )
            total += float(loss.data.item())
            kl_sum += metrics["kl_sum"]
            loss.backward()
        self.step_optimizers(step)
        return {
            "train_loss": total,
            "tokens": tokens,
            "grad_norm": self.last_grad_norm,
            "reward": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()),
            "kl": kl_sum / tokens,
            "responses": len(rollouts),
            "eos_rate": sum(r["finish_reason"] == "eos" for r in rollouts) / len(rollouts),
        }

    def evaluate(self, rows):
        rollouts = self.collect_rollouts(rows, self.seed)
        rewards = xp.asarray(self.reward_fn(rollouts)["rewards"], dtype=xp.float32)
        return {
            "eval_reward": float(rewards.mean().item()),
            "eval_eos_rate": sum(r["finish_reason"] == "eos" for r in rollouts) / len(rollouts)
        }

    @staticmethod
    def collate_rollouts(rollouts, pad_id):
        size = len(rollouts)
        length = max(len(r['prompt_ids']) + len(r['completion_ids']) - 1 for r in rollouts)

        shape = (size, length)
        batch = {
            "inputs": xp.full(shape, pad_id, dtype=xp.int64),
            "targets": xp.full(shape, pad_id, dtype=xp.int64),
            "token_mask": xp.zeros(shape, dtype=bool),
            "loss_mask": xp.zeros(shape, dtype=bool),
        }

        for i, rollout in enumerate(rollouts):
            prefix = rollout['prompt_ids']
            token = xp.asarray(prefix + rollout["completion_ids"], dtype=xp.int64)
            n = len(token) - 1

            batch["inputs"][i, :n] = token[:-1]
            batch["targets"][i, :n] = token[1:]
            batch["token_mask"][i, :n] = True
            batch["loss_mask"][i, len(prefix) - 1:n] = True

        return batch
