import hashlib
import json
from math import inf
from pathlib import Path

from core.backend import DEVICE, to_cpu, xp
from core.tensor import Tensor
from dataset.language_model_loader import render_prompt
from engine.checkpoint import Checkpoint
from engine.generator import Generator
from engine.metrics import sha256_bytes, write_json
from engine.post_training_loss import UnifiedLoss
from engine.run import load_run
from engine.trainer import BaseTrainer


def token_log_probs(logits, targets):
    batch, length, vocab = logits.data.shape
    flat_logits = logits.data.reshape(-1, vocab)
    flat_targets = xp.asarray(targets, dtype=xp.int64).reshape(-1)
    rows = xp.arange(flat_targets.size)
    shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
    exps = xp.exp(shifted)
    normalizer = exps.sum(axis=-1, keepdims=True)
    values = (shifted[rows, flat_targets] - xp.log(normalizer[:, 0])).reshape(batch, length)
    out = Tensor(values, require_grad=logits.require_grad, pre=(logits,), opr="token_log_probs")

    if out.require_grad:
        probs = exps / normalizer

        def _backward():
            g = out.grad.reshape(-1)
            grad = -probs * g[:, None]
            grad[rows, flat_targets] += g
            logits.accumulate_grad(grad.reshape(batch, length, vocab))

        out._set_backward(_backward)
    return out


def model_log_probs(model, batch, rows=slice(None)):
    logits = model(batch["inputs"][rows], token_mask=batch["token_mask"][rows])
    return token_log_probs(logits, batch["targets"][rows])


def reduce_completion_log_probs(logp, mask, *, normalize=False):
    total = Tensor.where(mask, logp, 0).sum(axis=-1)
    return total / mask.sum(axis=-1, dtype=logp.data.dtype) if normalize else total


def collate_completions(completions, pad_id):
    length = max(len(r["prompt_ids"]) + len(r["completion_ids"]) - 1 for r in completions)
    shape = (len(completions), length)
    inputs = xp.full(shape, pad_id, dtype=xp.int64)
    targets = xp.full_like(inputs, pad_id)
    token_mask = xp.zeros(shape, dtype=bool)
    loss_mask = xp.zeros_like(token_mask)
    for i, row in enumerate(completions):
        prefix = row["prompt_ids"]
        tokens = xp.asarray(prefix + row["completion_ids"], dtype=xp.int64)
        n = len(tokens) - 1
        inputs[i, :n], targets[i, :n] = tokens[:-1], tokens[1:]
        token_mask[i, :n] = True
        loss_mask[i, len(prefix) - 1:n] = True
    return dict(inputs=inputs, targets=targets, token_mask=token_mask, loss_mask=loss_mask)


def collate_pairs(pairs, pad_id):
    completions = [answer for pair in pairs for answer in pair]
    flat = collate_completions(completions, pad_id)
    return {name: array.reshape(len(pairs), 2, -1) for name, array in flat.items()}


def pair_log_probs(model, batch, *, normalize=False):
    length = batch["inputs"].shape[-1]
    flat = {name: batch[name].reshape(-1, length) for name in ("inputs", "targets", "token_mask")}
    logp = model_log_probs(model, flat)
    mask = batch["loss_mask"].reshape(-1, length)
    return reduce_completion_log_probs(logp, mask, normalize=normalize).reshape(-1, 2)


def _score_reference(model, splits, pad_id, batch_size):
    scores = {}
    with Tensor.no_grad():
        for split, pairs in splits.items():
            scores[split] = values = []
            for start in range(0, len(pairs), batch_size):
                batch = collate_pairs(pairs[start:start + batch_size], pad_id)
                values.extend(to_cpu(pair_log_probs(model, batch).data).tolist())
                if start % (batch_size * 10) == 0 or len(values) == len(pairs):
                    print(f"Reference {split}: {len(values)}/{len(pairs)} pairs", flush=True)
    return scores


def prepare_preference_reference(run, model, tokenizer, splits):
    reference_path = run.path / "reference.npz"
    if not run.resume:
        Checkpoint.save_model(reference_path, model)
    with reference_path.open("rb") as file:
        reference_digest = hashlib.file_digest(file, "sha256").hexdigest()
    context_size = run.config["context_size"]
    key = {
        "reference": reference_digest,
        "tokenizer": sha256_bytes((run.path / "tokenizer.json").read_bytes()),
        "context_size": context_size,
        "data": {name: sha256_bytes(Path(run.config["data"][name]["path"]).read_bytes())
                 for name in splits},
    }
    path = run.path / "reference_log_probs.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if cache.get("key") != key:
        reference = load_run(run.path, "reference", frozen=True)[0] if run.resume else model
        batch_size = max(1, min(8, 4096 // context_size))
        scores = _score_reference(reference, splits, tokenizer.pad_id, batch_size)
        del reference
        if DEVICE == "cuda":
            xp.get_default_memory_pool().free_all_blocks()
        cache = {"key": key, "scores": scores}
        write_json(path, cache)
    for split, pairs in splits.items():
        for (chosen, rejected), (chosen_logp, rejected_logp) in zip(pairs, cache["scores"][split], strict=True):
            chosen["reference_logp"] = chosen_logp
            rejected["reference_logp"] = rejected_logp


class PostTrainingTrainer(BaseTrainer):
    def __init__(self, model, optimizer, *, prepare_batch, loss_inputs, loss_fn, evaluate_fn=None, advantage_fn=None,
                 micro_batch_size=None, max_grad_norm=1.0, scheduler=None):
        super().__init__(model, optimizer, max_grad_norm, micro_batch_size, scheduler)
        self.prepare_batch = prepare_batch
        self.loss_inputs = loss_inputs
        self.loss_fn = loss_fn
        self.evaluate_fn = evaluate_fn
        self.advantage_fn = advantage_fn

    def _microbatches(self, batch):
        size = len(next(iter(batch.values())))
        micro = self.micro_batch_size or size
        for start in range(0, size, micro):
            yield {name: value[start:start + micro] for name, value in batch.items()}

    def _prepare(self, rows, step):
        batch, normalizer, metrics = self.prepare_batch(rows, step)
        if self.advantage_fn is not None:
            batch = {**batch, "advantages": self.advantage_fn(batch)}
        return batch, normalizer, metrics

    def _batch_loss(self, batch, normalizer, metrics, *, backward=False):
        total = 0.0
        for chunk in self._microbatches(batch):
            result = self.loss_fn(**self.loss_inputs(self.model, chunk))
            if isinstance(result, tuple):
                result, stats = result
                for name, value in stats.items():
                    metrics[name] = metrics.get(name, 0.0) + value
            loss = result.sum() / normalizer
            if backward:
                loss.backward()
            total += float(loss.data.item())
        return total

    def train_step(self, rows, step):
        with Tensor.no_grad():
            batch, normalizer, batch_metrics = self._prepare(rows, step)
        self.zero_grad()
        metrics = dict(batch_metrics)
        total = self._batch_loss(batch, normalizer, metrics, backward=True)
        self.step_optimizers(step)
        return {**metrics, "train_loss": total, "grad_norm": self.last_grad_norm}

    def evaluate(self, rows):
        with Tensor.no_grad():
            if self.evaluate_fn is not None:
                return self.evaluate_fn(rows)
            micro = self.micro_batch_size or len(rows)
            metrics = {}
            total, count = 0.0, 0
            for start in range(0, len(rows), micro):
                batch, normalizer, stats = self._prepare(rows[start:start + micro], step=None)
                for name, value in stats.items():
                    metrics[name] = metrics.get(name, 0) + value
                total += self._batch_loss(batch, 1, metrics)
                count += normalizer
            return {"val_loss": total / count, **{f"eval_{name}": value for name, value in metrics.items()}}


class RolloutTrainer(PostTrainingTrainer):
    default_advantage = None

    def __init__(
        self, model, optimizer, tokenizer, loss_fn, reward_fn, context_size=512,
        group_size=4, max_new_tokens=128, micro_batch_size=1, scheduler=None,
        reference_fn=None, seed=42, max_grad_norm=1.0, *, advantage_fn=None, loss_inputs=None,
        kl_weight=0.0, clip_epsilon=None,
    ):
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.context_size = context_size
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.reference_fn = reference_fn
        self.seed = seed
        self.kl_weight = kl_weight
        self.clip_epsilon = clip_epsilon
        self._uses_default_loss = loss_fn is None
        if self._uses_default_loss:
            self.policy_curve = UnifiedLoss(0)
            self.reference_curve = UnifiedLoss(inf)
            loss_fn = self._rollout_loss
            if loss_inputs is None:
                loss_inputs = self._clipped_policy_inputs
        super().__init__(
            model, optimizer, prepare_batch=self._prepare_rollouts, evaluate_fn=self._evaluate_rollouts,
            loss_inputs=self._policy_inputs if loss_inputs is None else loss_inputs, loss_fn=loss_fn,
            advantage_fn=self.default_advantage if advantage_fn is None else advantage_fn,
            micro_batch_size=micro_batch_size, max_grad_norm=max_grad_norm, scheduler=scheduler,
        )

    def collect_rollouts(self, rows, seed):
        generator = Generator(self.model, self.context_size, seed=seed)
        expanded = []
        for row in rows:
            prefix = [self.tokenizer.bos_id, *self.tokenizer.encode(render_prompt(row["prompt"]))]
            expanded.extend((row, prefix) for _ in range(self.group_size))
        results = generator.generate_batch(
            [prefix for _, prefix in expanded],
            self.tokenizer.eos_id,
            max_new_tokens=self.max_new_tokens,
        )
        return [
            {
                **row, "prompt_ids": prefix, "completion_ids": result["token_ids"],
                "completion": self.tokenizer.decode(result["token_ids"], errors="replace"),
                "finish_reason": result["finish_reason"],
            }
            for (row, prefix), result in zip(expanded, results, strict=True)
        ]

    def log_probs(self, batch, model=None):
        model = self.model if model is None else model
        result = xp.empty(batch["targets"].shape, dtype=xp.float32)
        micro = self.micro_batch_size or len(result)
        with Tensor.no_grad():
            for start in range(0, len(result), micro):
                rows = slice(start, start + micro)
                result[rows] = model_log_probs(model, batch, rows).data
        return result

    def _score_rollouts(self, rows, seed):
        rollouts = self.collect_rollouts(rows, seed)
        rewards = xp.asarray(self.reward_fn(rollouts)["rewards"], dtype=xp.float32)
        metrics = {
            "reward": float(rewards.mean().item()),
            "eos_rate": sum(r["finish_reason"] == "eos" for r in rollouts) / len(rollouts),
        }
        return rollouts, rewards, metrics

    def loss_normalizer(self, metrics):
        return metrics["tokens"]

    def _prepare_rollouts(self, rows, step):
        rollouts, rewards, metrics = self._score_rollouts(rows, self.seed + step)
        batch = collate_completions(rollouts, self.tokenizer.pad_id)
        old = self.log_probs(batch)
        needs_reference = bool(self.kl_weight) if self._uses_default_loss else self.reference_fn is not None
        if needs_reference:
            batch["reference"] = self.reference_fn(batch)
        batch.update(rewards=rewards, old=old)
        metrics.update(tokens=int(batch["loss_mask"].sum().item()), responses=len(rollouts),
                       reward_std=float(rewards.std().item()))
        return batch, self.loss_normalizer(metrics), metrics

    @staticmethod
    def _policy_inputs(model, batch):
        return dict(current=model_log_probs(model, batch), old=batch["old"], mask=batch["loss_mask"],
                    advantages=batch.get("advantages"), reference=batch.get("reference"))

    @staticmethod
    def policy_terms(current, old, advantages, mask, reference=None):
        mask = xp.asarray(mask, dtype=bool)
        logp = current[mask]
        advantage = Tensor._as_tensor(advantages).data
        if advantage.ndim == 1:
            advantage = advantage[:, None]
        advantage = xp.broadcast_to(advantage, current.data.shape)[mask]
        ratio = (logp - Tensor._as_tensor(old).data[mask]).exp()
        reference_log_ratio = None if reference is None else -(logp - Tensor._as_tensor(reference).data[mask])
        return ratio, advantage, reference_log_ratio

    def _clipped_policy_inputs(self, model, batch):
        ratio, advantage, reference_log_ratio = self.policy_terms(**self._policy_inputs(model, batch))
        if isinstance(advantage, Tensor):
            advantage = advantage.data
        epsilon = self.clip_epsilon
        if epsilon is not None:
            low, high = epsilon if isinstance(epsilon, (list, tuple)) else (epsilon, epsilon)
            direction = xp.asarray(advantage)
            dtype = ratio.data.dtype
            lower = xp.where(direction < 0, 1 - low, 0).astype(dtype)
            upper = xp.where(direction > 0, 1 + high, xp.inf).astype(dtype)
            ratio = ratio.clip(lower, upper)
        return dict(z=ratio, advantage=advantage, reference_log_ratio=reference_log_ratio)

    def _rollout_loss(self, z, advantage, reference_log_ratio):
        loss = self.policy_curve(z) * advantage
        if self.kl_weight:
            kl = self.reference_curve(reference_log_ratio, offset=1)
            return loss + self.kl_weight * kl, {"divergence_sum": float(kl.data.sum().item())}
        return loss, {"divergence_sum": 0.0}

    def train_step(self, rows, step):
        metrics = super().train_step(rows, step)
        if self._uses_default_loss:
            metrics["kl"] = metrics.pop("divergence_sum") / metrics["tokens"]
        return metrics

    def _evaluate_rollouts(self, rows):
        _, _, metrics = self._score_rollouts(rows, self.seed)
        return {f"eval_{key}": value for key, value in metrics.items()}


class PreferenceTrainer(PostTrainingTrainer):
    uses_reference = False

    def __init__(self, model, optimizer, pad_id, *, loss_fn, preference_inputs=None, length_normalize=False, **kwargs):
        self.pad_id = pad_id
        self.preference_inputs = preference_inputs
        self.length_normalize = length_normalize
        super().__init__(
            model,
            optimizer,
            prepare_batch=self._prepare_pairs,
            loss_inputs=self._loss_inputs,
            loss_fn=loss_fn, **kwargs
        )

    def _prepare_pairs(self, pairs, step):
        batch = collate_pairs(pairs, self.pad_id)
        if self.uses_reference:
            batch["reference"] = xp.asarray(
                [[answer["reference_logp"] for answer in pair] for pair in pairs], dtype=xp.float32,
            )
        return batch, len(pairs), {"tokens": int(batch["loss_mask"].sum().item()), "pairs": len(pairs)}

    def _loss_inputs(self, model, batch):
        scores = pair_log_probs(model, batch, normalize=self.length_normalize)
        if "reference" in batch:
            scores = scores - batch["reference"]
        return self.preference_inputs(scores[:, :1] - scores[:, 1:])
