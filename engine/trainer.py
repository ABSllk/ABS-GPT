import math
import time

import numpy as np
from cupyx.profiler import time_range

from core.backend import to_cpu, xp
from core.nn import FusedCrossEntropyLoss
from core.optimizer import clip_grad_norm
from core.tensor import Tensor
from engine.metrics import synchronize, perplexity


class LRScheduler:
    def __init__(self, optimizers, base_lrs, warmup_steps=0, cooldown=None):
        self.optimizers = optimizers
        self.base_lrs = list(base_lrs)
        self.warmup_steps = warmup_steps
        self.cooldown = cooldown

    def get_factor(self, step):
        factor = (
            min(1.0, step / self.warmup_steps)
            if self.warmup_steps else 1.0
        )

        if self.cooldown is not None:
            config = self.cooldown
            progress = (step - config["start_step"]) / config["steps"]
            progress = min(1.0, max(0.0, progress))

            if config.get("shape", "linear") == "cosine":
                progress = (1.0 - math.cos(math.pi * progress)) / 2.0

            start = config.get("start_lr_ratio", 1.0)
            end = config["min_lr_ratio"]
            factor *= start + (end - start) * progress

        return factor

    def step(self, step):
        factor = self.get_factor(step)
        for optimizer, base_lr in zip(self.optimizers, self.base_lrs):
            optimizer.param_groups[0]["lr"] = base_lr * factor


class FixedValidation:
    def __init__(self, token_ids, starts, context_size, batch_size):
        self.token_ids = token_ids
        self.starts = np.asarray(starts, dtype=np.int64)
        self.context_size = context_size
        self.batch_size = batch_size

    def evaluate(self, model):
        loss_fn = FusedCrossEntropyLoss()
        total_loss, total_tokens = 0.0, 0
        synchronize()
        start = time.perf_counter()
        with Tensor.no_grad():
            for offset in range(0, len(self.starts), self.batch_size):
                positions = self.starts[offset:offset + self.batch_size, None] + np.arange(self.context_size)
                targets = self.token_ids[positions + 1]
                loss = loss_fn(model(self.token_ids[positions]), targets)
                total_loss += float(loss.data.item()) * targets.size
                total_tokens += targets.size
        synchronize()
        loss = total_loss / total_tokens
        return {"val_loss": loss, "perplexity": perplexity(loss),
                "eval_tokens": total_tokens, "eval_seconds": time.perf_counter() - start}


def mtp_loss_masks(shape, depth_count, loss_mask=None, token_mask=None, document_ids=None):
    if all(mask is None for mask in (loss_mask, token_mask, document_ids)):
        return [None] * depth_count
    valid = xp.asarray(token_mask, dtype=bool) if token_mask is not None else xp.ones(shape, dtype=bool)
    groups = xp.asarray(document_ids) if document_ids is not None else None
    if groups is not None:
        valid = valid & (groups >= 0)
    masks = []
    for depth in range(1, depth_count + 1):
        valid = valid[:, :-1]
        if token_mask is not None:
            valid = valid & xp.asarray(token_mask)[:, depth:]
        if groups is not None:
            valid = valid & (groups[:, :-depth] == groups[:, depth:])
        masks.append(valid & xp.asarray(loss_mask)[:, depth:] if loss_mask is not None else valid)
    return masks


class BaseTrainer:
    def __init__(self, model, optimizer, max_grad_norm=1.0, micro_batch_size=None, scheduler=None):
        self.model = model
        self.optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        self.parameters = [p for item in self.optimizers for p in item.parameters]
        self.max_grad_norm = max_grad_norm
        self.micro_batch_size = micro_batch_size
        self.scheduler = scheduler

    @time_range("zero_grad")
    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step_optimizers(self, step):
        with time_range("grad_clip"):
            self.last_grad_norm = clip_grad_norm(self.parameters, self.max_grad_norm)
        if self.scheduler is not None:
            self.scheduler.step(step)
        for optimizer in self.optimizers:
            with time_range(type(optimizer).__name__):
                optimizer.step()

    def fit(self, train_loader, eval_loader, steps, eval_interval, run, validation_callback=None):
        def validate():
            run.release_cache()
            metrics = self.evaluate(eval_loader)
            run.save(self.model, self.optimizers, train_loader, metrics)
            return bool(validation_callback(run.state["completed_steps"], metrics, self.model)) if validation_callback else False

        run.begin(self.model)
        stopped = validate() if not run.resume else False
        stop_reason = "planned_steps"
        for step in range(run.state["completed_steps"] + 1, steps + 1):
            if stopped:
                stop_reason = "validation_guard"
                break
            if run.time_exceeded():
                stop_reason = "time_budget"
                break
            batch = run.next_batch(train_loader, step)
            metrics = self.train_step(batch, step=step)
            metrics["learning_rate"] = self.optimizers[0].param_groups[0]["lr"]
            for optimizer in self.optimizers[1:]:
                metrics[f"{type(optimizer).__name__.lower()}_learning_rate"] = optimizer.param_groups[0]["lr"]
            run.log_train(step, metrics, train_loader)
            if step % eval_interval == 0 or step == steps or step in run.eval_at:
                if validate():
                    stop_reason = "validation_guard"
                    break
        if run.last_validation_step != run.state["completed_steps"]:
            validate()
        run.finish(stop_reason)


class LanguageModelTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, max_grad_norm=1.0, micro_batch_size=None, scheduler=None,
                 return_mtp=False, micro_batch_size_by_context=None):
        super().__init__(model, optimizer, max_grad_norm, micro_batch_size, scheduler)
        self.loss_fn = loss_fn
        self.return_mtp = return_mtp
        self.micro_batch_size_by_context = {
            int(context): size for context, size in (micro_batch_size_by_context or {}).items()
        }

    @time_range("train_step")
    def train_step(self, batch, step=None):
        inputs, targets = batch["inputs"], batch["targets"]
        loss_mask, token_mask = batch.get("loss_mask"), batch.get("token_mask")
        segment_ids, document_ids = batch.get("segment_ids"), batch.get("document_ids")
        return_mtp = self.return_mtp
        self.zero_grad()
        batch_size = len(inputs)
        micro = self.micro_batch_size_by_context.get(inputs.shape[1], self.micro_batch_size or batch_size)
        total = ntp_total = mtp_total = aux_total = 0.0
        counts_total = None
        use_moe = getattr(self.model, "use_moe", False)            
        total_targets = int(loss_mask.sum()) if loss_mask is not None else batch_size
        total_inputs = int(token_mask.sum()) if token_mask is not None else batch_size
        groups = document_ids if document_ids is not None else segment_ids
        mtp_masks = mtp_loss_masks(inputs.shape, self.model.mtp_block_num, loss_mask, token_mask, groups) if return_mtp else []
        mtp_counts = [int(mask.sum()) if mask is not None else batch_size * (targets.shape[1] - depth)
                      for depth, mask in enumerate(mtp_masks, start=1)]
        if total_targets <= 0 or total_inputs <= 0:
            raise ValueError("Training batch must contain supervised and valid tokens")
        for offset in range(0, batch_size, micro):
            end = offset + micro
            chunk_inputs = inputs[offset:end]
            chunk_targets = targets[offset:end]

            chunk_loss_mask = loss_mask[offset:end] if loss_mask is not None else None
            chunk_token_mask = token_mask[offset:end] if token_mask is not None else None
            chunk_segment_ids = segment_ids[offset:end] if segment_ids is not None else None
            chunk_document_ids = document_ids[offset:end] if document_ids is not None else None
            chunk_mtp_masks = [mask[offset:end] if mask is not None else None for mask in mtp_masks]
            mtp_weights = [(int(mask.sum()) if mask is not None else (end - offset) * (targets.shape[1] - depth)) / max(count, 1)
                           for depth, (mask, count) in enumerate(zip(chunk_mtp_masks, mtp_counts), start=1)]
            target_count = int(chunk_loss_mask.sum()) if chunk_loss_mask is not None else len(chunk_targets)
            input_count = int(chunk_token_mask.sum()) if chunk_token_mask is not None else len(chunk_inputs)
            
            ntp_weight = target_count / total_targets 
            aux_weight = input_count / total_inputs

            loss = self._forward_loss(
                inputs=chunk_inputs,
                targets=chunk_targets, 
                return_mtp=return_mtp, 
                loss_mask=chunk_loss_mask,
                token_mask=chunk_token_mask,
                segment_ids=chunk_segment_ids,
                document_ids=chunk_document_ids,
                mtp_masks=chunk_mtp_masks,
                mtp_weights=mtp_weights,
                ntp_weight=ntp_weight, 
                aux_weight=aux_weight
            )
            total += float(loss.data.item())
            if use_moe:
                ntp_total += self.last_ntp_loss * ntp_weight
                mtp_total += self.last_mtp_loss
                aux_total += self.last_moe_aux_loss * aux_weight

            with time_range("backward"):
                loss.backward()

            if use_moe:
                counts = np.stack([to_cpu(block.feed_forward.last_expert_counts) for block in self.model.transformer_blocks])
                counts_total = counts if counts_total is None else counts_total + counts
        if use_moe:
            self.last_ntp_loss, self.last_mtp_loss, self.last_moe_aux_loss = ntp_total, mtp_total, aux_total
            self.last_expert_counts = counts_total
        self.step_optimizers(step)
        metrics = {
            "train_loss": total,
            "tokens": int(loss_mask.sum()) if loss_mask is not None else targets.size,
            "grad_norm": self.last_grad_norm,
            "context_size": inputs.shape[1],
            "batch_size": batch_size,
        }
        if use_moe:
            metrics.update(ntp_loss=ntp_total, mtp_loss=mtp_total, moe_aux_loss=aux_total)
            fractions = counts_total / counts_total.sum(axis=1, keepdims=True)
            metrics["routing"] = {
                "min_expert_fraction": float(fractions.min()),
                "max_expert_fraction": float(fractions.max()),
                "empty_experts": int((counts_total == 0).sum()),
                "expert_counts": counts_total.tolist(),
            }
        return metrics

    @time_range("forward")
    def _forward_loss(self, inputs, targets, return_mtp, loss_mask, token_mask, segment_ids, document_ids, mtp_masks, mtp_weights, ntp_weight, aux_weight):
        use_moe = getattr(self.model, "use_moe", False)
        mtp_args = {"document_ids": document_ids} if return_mtp and document_ids is not None else {}
        if use_moe:
            outputs, aux_loss = self.model(inputs, return_mtp=return_mtp, return_aux=True, token_mask=token_mask, segment_ids=segment_ids, **mtp_args)
            if return_mtp:
                logits, mtp_logits = outputs
            else:
                logits = outputs
        elif return_mtp:
            logits, mtp_logits = self.model(inputs, return_mtp=True, token_mask=token_mask, segment_ids=segment_ids, **mtp_args)
        else:
            logits = self.model(inputs, token_mask=token_mask, segment_ids=segment_ids)
        ntp_loss = self.loss_fn(logits, targets, loss_mask)
        loss = ntp_weight * ntp_loss

        if return_mtp:
            terms = [weight * self.loss_fn(pred, targets[:, depth:], mask)
                     for depth, (pred, mask, weight) in enumerate(zip(mtp_logits, mtp_masks, mtp_weights), start=1)
                     if weight > 0]
            mtp_loss = sum(terms) / len(mtp_logits) if terms else Tensor(0.0)
            loss = loss + self.model.mtp_loss_weight * mtp_loss
        if use_moe:
            loss = loss + aux_weight * self.model.moe_aux_weight * aux_loss
            self.last_ntp_loss = float(ntp_loss.data.item())
            self.last_moe_aux_loss = float(aux_loss.data.item())
            self.last_mtp_loss = float(mtp_loss.data.item()) if return_mtp else 0.0
        return loss

    def evaluate(self, loader, steps=None):
        if isinstance(loader, FixedValidation):
            return loader.evaluate(self.model)
        total_loss = 0.0
        total_tokens = 0
        batches = (loader.iter_eval_batches() if steps is None
                   else (loader.next_batch() for _ in range(steps)))
        with Tensor.no_grad():
            for batch in batches:
                mask = batch.get("loss_mask")
                count = int(mask.sum()) if mask is not None else batch["targets"].size
                logits = self.model(
                    batch["inputs"],
                    token_mask=batch.get("token_mask"),
                    segment_ids=batch.get("segment_ids"),
                )
                loss = self.loss_fn(logits, batch["targets"], mask)
                total_loss += float(loss.data.item()) * count
                total_tokens += count
        return {"val_loss": total_loss / total_tokens}

