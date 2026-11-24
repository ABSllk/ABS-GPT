from collections import defaultdict
import numpy as np

from core.tensor import Tensor
from core.backend import to_cpu, xp
from core.nn import KVCache, Softmax

class Generator:
    def __init__(self, model, context_size, seed=None):
        self.model = model
        self.context_size = context_size
        self.softmax = Softmax(axis=-1)
        self.rng = xp.random.RandomState(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate(self, start_tokens, end_token, max_new_tokens=20, temperature=1.0, do_sample=True, top_p=1.0):
        tokens = list(start_tokens)
        generated = []
        finish_reason = "length"
        cache = [KVCache(self.context_size) for _ in self.model.transformer_blocks]

        with Tensor.no_grad():
            for _ in range(max_new_tokens):
                past_len = cache[0].length

                # 若窗口满了，则清空缓存
                if 0 < past_len < self.context_size:
                    current_tokens = tokens[-1:]
                else:
                    for layer_cache in cache:
                        layer_cache.length = 0
                    current_tokens = tokens[-self.context_size:]

                context = xp.array([current_tokens], dtype=xp.int64)
                logits = self.model(context, cache)
                if logits.data.ndim == 3:
                    logits = logits[:, -1, :]
                if do_sample:
                    logits = logits / temperature
                    probabilities = self.softmax(logits).data[0]
                    if top_p < 1.0:
                        order = xp.argsort(probabilities)[::-1]
                        selected = probabilities[order]
                        selected = xp.where(xp.cumsum(selected) - selected < top_p, selected, 0)
                        selected = selected / selected.sum()
                        choice = self.rng.choice(len(selected), p=selected, size=1)[0]
                        next_token = int(order[choice])
                    else:
                        next_token = int(self.rng.choice(len(probabilities), p=probabilities, size=1)[0])
                else:
                    next_token = int(xp.argmax(logits.data[0]).item())

                generated.append(next_token)
                tokens.append(next_token)
                if next_token == end_token:
                    finish_reason = "eos"
                    break

        return {
            "token_ids": generated,
            "finish_reason": finish_reason,
        }

    def generate_batch(self, prefixes, end_token, max_new_tokens=20, temperature=1.0, do_sample=True, top_p=1.0, seeds=None):
        if not prefixes or max_new_tokens <= 0:
            return [{"token_ids": [], "finish_reason": "length"} for _ in prefixes]

        if seeds is None:
            seeds = self._np_rng.integers(0, 2**32, size=len(prefixes)).tolist()
        groups = defaultdict(list)
        for index, (prefix, seed) in enumerate(zip(prefixes, seeds, strict=True)):
            groups[len(prefix)].append((index, list(prefix), int(seed)))

        results = [None] * len(prefixes)
        with Tensor.no_grad():
            for group in groups.values():
                tokens = [prefix for _, prefix, _ in group]
                generated = [[] for _ in group]
                finished = {}
                rngs = [np.random.default_rng(seed) for _, _, seed in group]
                active = list(range(len(group)))
                cache = [KVCache(self.context_size) for _ in self.model.transformer_blocks]
                selected = None

                for _ in range(max_new_tokens):
                    past_len = cache[0].length
                    if 0 < past_len < self.context_size:
                        inputs = selected[:, None]
                    else:
                        for layer_cache in cache:
                            layer_cache.length = 0
                        inputs = xp.asarray([tokens[row][-self.context_size:] for row in active], dtype=xp.int64)

                    logits = self.model(inputs, cache)
                    logits = logits.data[:, -1, :] if logits.data.ndim == 3 else logits.data

                    if do_sample:
                        scaled = logits / temperature
                        shifted = scaled - scaled.max(axis=-1, keepdims=True)
                        probs = xp.exp(shifted)
                        probs /= probs.sum(axis=-1, keepdims=True)
                        if top_p < 1.0:
                            order = xp.argsort(probs, axis=-1)[:, ::-1]
                            mass = xp.take_along_axis(probs, order, axis=-1)
                            cdf = xp.cumsum(mass, axis=-1)
                            kept = xp.where(cdf - mass < top_p, mass, 0).sum(axis=-1)
                            draws = xp.asarray([rngs[row].random() for row in active], dtype=probs.dtype) * kept
                            at = (cdf < draws[:, None]).sum(axis=-1).clip(0, order.shape[1] - 1)
                            selected = order[xp.arange(len(active)), at]
                        else:
                            cdf = xp.cumsum(probs, axis=-1)
                            draws = xp.asarray([rngs[row].random() for row in active], dtype=probs.dtype) * cdf[:, -1]
                            selected = (cdf < draws[:, None]).sum(axis=-1).clip(0, probs.shape[1] - 1)
                    else:
                        selected = xp.argmax(logits, axis=-1)

                    survivors = []
                    for idx, (row, token) in enumerate(zip(active, to_cpu(selected).tolist(), strict=True)):
                        generated[row].append(token)
                        tokens[row].append(token)
                        if token == end_token:
                            finished[row] = "eos"
                        else:
                            survivors.append(idx)
                    if not survivors:
                        break
                    if len(survivors) != len(active):
                        active = [active[idx] for idx in survivors]
                        keep = xp.asarray(survivors, dtype=xp.int64)
                        for layer_cache in cache:
                            layer_cache.retain(keep)
                        selected = selected[keep]

                for row, (index, _, _) in enumerate(group):
                    results[index] = {
                        "token_ids": generated[row],
                        "finish_reason": finished.get(row, "length"),
                    }
        return results
