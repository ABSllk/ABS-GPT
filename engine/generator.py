from core.tensor import Tensor
from core.backend import xp
from core.nn import Softmax

class Generator:
    def __init__(self, model, context_size, seed=None):
        self.model = model
        self.context_size = context_size
        self.softmax = Softmax(axis=-1)
        self.rng = xp.random.RandomState(seed)

    def generate(self, start_tokens, end_token, max_new_tokens=20, temperature=1.0, do_sample=True, top_p=1.0):
        tokens = list(start_tokens)
        generated = []
        finish_reason = "length"
        cache = [{} for _ in self.model.transformer_blocks]

        with Tensor.no_grad():
            for _ in range(max_new_tokens):
                past_len = cache[0]["K"].shape[2] if cache[0] else 0

                # 若窗口满了，则清空缓存
                if 0 < past_len < self.context_size:
                    current_tokens = tokens[-1:]
                else:
                    for layer_cache in cache:
                        layer_cache.clear()
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
