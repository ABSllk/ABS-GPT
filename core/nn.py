from core.backend import xp, to_cpu
# from .flash_attention import tiled_attention
from .triton_attention import tiled_attention
from .triton_ops import matmul_tensor
from .tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def named_parameters(self, prefix="", _visited=None, trainable_only=True):
        if _visited is None:
            _visited = set()

        def walk(value, name):
            if isinstance(value, Tensor):
                if (value.require_grad or not trainable_only) and id(value) not in _visited:
                    _visited.add(id(value))
                    yield name, value

            elif isinstance(value, Module):
                yield from value.named_parameters(name + ".", _visited, trainable_only=trainable_only)

            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    yield from walk(item, f"{name}.{i}")

            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from walk(item, f"{name}.{key}")

        for name, attr in self.__dict__.items():
            yield from walk(attr, prefix + name)
    
    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def state_dict(self):
        return {name: param.data.copy() for name, param in self.named_parameters(trainable_only=False)}

    def load_state_dict(self, state_dict):
        parameters = dict(self.named_parameters(trainable_only=False))
        for name, value in state_dict.items():
            param = parameters[name]
            param.data = xp.asarray(value, dtype=param.data.dtype)
            param.zero_grad()
        

class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.use_triton = True
        self.weight = Tensor(xp.random.randn(input_size, output_size) * xp.sqrt(2.0 / input_size), require_grad=True)
        self.bias = Tensor(xp.zeros((output_size)), require_grad=True)

    def __call__(self, inputs):
        shape = inputs.data.shape
        x = inputs.reshape(-1, shape[-1])
        output = (matmul_tensor(x, self.weight) if self.use_triton else x @ self.weight) + self.bias
        return output.reshape(*shape[:-1], self.weight.data.shape[1])

    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        return inputs.relu()

class SiLu(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        return inputs.silu()

class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = Tensor(xp.random.randn(vocab_size, embedding_dim) * xp.sqrt(2.0 / vocab_size), require_grad=True)

    def __call__(self, inputs):
        return self.weight[inputs]

    def parameters(self):
        return [self.weight]

class LogSoftmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, inputs):
        max_vals = xp.max(inputs.data, axis=self.axis, keepdims=True)
        shifted = inputs - Tensor(max_vals)
        log_sum_exp = (shifted.exp().sum(axis=self.axis, keepdims=True).log())
        return shifted - log_sum_exp

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        self.log_softmax = LogSoftmax(axis=axis)

    def __call__(self, inputs):
        return self.log_softmax(inputs).exp()

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def __call__(self, inputs, targets, loss_mask=None):
        targets = xp.array(targets, dtype=xp.int64)
        flat_inputs = inputs.reshape(-1, inputs.data.shape[-1])
        flat_targets = targets.reshape(-1)
        if loss_mask is not None:
            loss_mask = xp.asarray(loss_mask, dtype=xp.bool).reshape(-1)
            flat_inputs = flat_inputs[loss_mask]
            flat_targets = flat_targets[loss_mask]

        log_probs = self.log_softmax(flat_inputs)
        return -log_probs[range(len(flat_targets)), flat_targets].mean()


class FusedCrossEntropyLoss(Module):
    def __call__(self, inputs, targets, loss_mask=None):
        logits = inputs.data.reshape(-1, inputs.data.shape[-1])
        targets = xp.asarray(targets, dtype=xp.int64).reshape(-1)
        if loss_mask is not None:
            loss_mask = xp.asarray(loss_mask, dtype=xp.bool).reshape(-1)
            logits = logits[loss_mask]
            targets = targets[loss_mask]

        rows = xp.arange(len(targets))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = xp.exp(shifted)
        normalizer = probabilities.sum(axis=-1, keepdims=True)
        value = (xp.log(normalizer[:, 0]) - shifted[rows, targets]).mean()
        probabilities /= normalizer
        out = Tensor(value, require_grad=inputs.require_grad, pre=(inputs,), opr="fused_cross_entropy")

        def backward():
            gradient = probabilities.copy()
            gradient[rows, targets] -= 1
            gradient *= out.grad / len(targets)

            full_gradient = xp.zeros_like(inputs.data.reshape(-1, inputs.data.shape[-1]))
            if loss_mask is not None:
                full_gradient[loss_mask] = gradient
            else:
                full_gradient = gradient

            inputs.grad += full_gradient.reshape(inputs.data.shape)

        out._set_backward(backward)
        return out

class SelfAttention(Module):
    def __init__(self, head_size, head_num, kv_head_num=None):
        super().__init__()
        self.embedding_dim = head_size * head_num
        self.head_size = head_size
        self.head_num = head_num
        self.kv_head_num = head_num if kv_head_num is None else kv_head_num
        assert self.kv_head_num > 0 and head_num % self.kv_head_num == 0, "head-num must be divisible by kv-head-num"
        
        #形状：[x,d]
        self.W_q = Linear(self.embedding_dim, self.embedding_dim) 
        self.W_k = Linear(self.embedding_dim, self.kv_head_num * self.head_size) 
        self.W_v = Linear(self.embedding_dim, self.kv_head_num * self.head_size)

        self.output = Linear(self.embedding_dim, self.embedding_dim)

    def rope(self, x, offset=0):
        t, d = x.data.shape[-2:]
        
        freq = 10000.0 ** (-xp.arange(0, d, 2, dtype=xp.float32) / d)
        angles = (xp.arange(t, dtype=xp.float32) + offset)[:, None] * freq
        cos = xp.repeat(xp.cos(angles), 2, axis=-1)
        sin = (xp.sin(angles)[..., None] * xp.asarray([-1, 1], dtype=xp.float32)).reshape(t, d)
        swapped = x.reshape(*x.data.shape[:-1], d // 2, 2)[..., ::-1].reshape(*x.data.shape)
        return x * Tensor(cos) + swapped * Tensor(sin)

    def __call__(self, inputs, cache=None, segment_ids=None):
        batch_size, seq_len, _ = (inputs.data.shape)

        Q = self.W_q(inputs).reshape(batch_size, seq_len, self.head_num, self.head_size).transpose(0, 2, 1, 3)
        K = self.W_k(inputs).reshape(batch_size, seq_len, self.kv_head_num, self.head_size).transpose(0, 2, 1, 3)
        V = self.W_v(inputs).reshape(batch_size, seq_len, self.kv_head_num, self.head_size).transpose(0, 2, 1, 3)
        
        past_length = cache["K"].shape[2] if cache is not None and "K" in cache else 0

        Q = self.rope(Q, past_length)
        K = self.rope(K, past_length)

        if cache is not None:
            if "K" in cache:
                K = Tensor(xp.concatenate([cache["K"], K.data], axis=2))
                V = Tensor(xp.concatenate([cache["V"], V.data], axis=2))
            cache["K"] = K.data
            cache["V"] = V.data

        # 普通的注意力计算
        # scores = Q @ K.transpose(0, 1, 3, 2) / xp.sqrt(self.head_size)
        # query_pos = xp.arange(seq_len) + past_length
        # key_pos = xp.arange(K.data.shape[2])
        # mask = key_pos[None, :] > query_pos[:, None]
        # scores = scores + Tensor(xp.where(mask, -1e9, 0))

        # attention_weights = self.softmax(scores)
        # output = (attention_weights @ V).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim)

        # Flash Attention
        output = tiled_attention(Q, K, V ,past_len=past_length, segment_ids=segment_ids)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim)

        return self.output(output)
    
    def parameters(self):
        return (self.W_q.parameters() + self.W_k.parameters() + self.W_v.parameters() + self.output.parameters())

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(xp.ones(normalized_shape), require_grad=True)
        self.beta = Tensor(xp.zeros(normalized_shape), require_grad=True)

    def __call__(self, inputs):
        mean = inputs.mean(axis=-1, keepdims=True)
        variance = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (inputs - mean) / ((variance + self.eps) ** 0.5)
        return self.gamma * normalized + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(xp.ones(normalized_shape), require_grad=True)

    def __call__(self, inputs):
        rms = ((inputs ** 2).mean(axis=-1, keepdims=True) + self.eps) ** 0.5
        return self.gamma * (inputs / rms)

class FeedForward(Module):
    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.linear1 = Linear(embedding_dim, hidden_dim)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, embedding_dim)

    def __call__(self, inputs):
        return self.linear2(self.relu(self.linear1(inputs)))

class SwiGLUFeedForward(Module):
    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 8 * embedding_dim // 3
        self.gate_linear = Linear(embedding_dim, hidden_dim)
        self.activation = SiLu()
        self.up_linear = Linear(embedding_dim, hidden_dim)
        self.down_linear = Linear(hidden_dim, embedding_dim)

    def __call__(self, inputs):
        gate = self.activation(self.gate_linear(inputs))
        up = self.up_linear(inputs)
        down = self.down_linear(gate * up)
        return down

class MoEFeedForward(Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_experts=8, top_k=2, use_shared_experts=False):
        if hidden_dim == None:
            self.hidden_dim = (8 * embedding_dim // 3) // top_k
        else:
            self.hidden_dim = hidden_dim

        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_experts = use_shared_experts
        self.experts = [SwiGLUFeedForward(embedding_dim, self.hidden_dim) for _ in range(num_experts)]
        self.shared_expert = SwiGLUFeedForward(embedding_dim, self.hidden_dim) if use_shared_experts else None
        for expert in self.experts:
            for linear in (expert.gate_linear, expert.up_linear, expert.down_linear):
                linear.use_triton = False
        self.router = Linear(embedding_dim, num_experts)
        self.softmax = Softmax(axis=-1)
        self.router.weight.data[:] = xp.random.randn(*self.router.weight.data.shape) * xp.sqrt(2.0 / embedding_dim)
        self.router.bias.data.fill(0)

    def __call__(self, inputs, return_aux=False, token_mask=None):
        batch_size, seq_len, embedding_dim = inputs.data.shape
        x = inputs.reshape(-1, embedding_dim)
        num_tokens = x.data.shape[0]

        router_logits = self.router(x)
        router_probs = self.softmax(router_logits)

        top_k_indices = xp.argsort(router_probs.data, axis=-1)[:, -self.top_k:]
        rows = xp.arange(num_tokens)[:, None]
        top_k_probs = router_probs[rows, top_k_indices]
        top_k_probs = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

        flat_experts = top_k_indices.reshape(-1)
        grouped = xp.argsort(flat_experts)

        counts = (flat_experts[:, None] == xp.arange(self.num_experts)[None, :]).sum(axis=0)
        host_counts = to_cpu(counts)
        output = self.shared_expert(x) if self.use_shared_experts else Tensor(xp.zeros_like(x.data))
        offset = 0
        for expert_id, expert in enumerate(self.experts):
            count = int(host_counts[expert_id])
            entries = grouped[offset:offset + count]
            offset += count
            if count == 0:
                continue
            token_ids, slots = entries // self.top_k, entries % self.top_k
            
            expert_inputs = x[token_ids]
            expert_outputs = expert(expert_inputs)
            expert_weights = top_k_probs[token_ids, slots].reshape(-1, 1)
            expert_outputs = expert_outputs * expert_weights

            output = output + expert_outputs.scatter_add_rows(token_ids, num_tokens)
        output = output.reshape(batch_size, seq_len, embedding_dim)

        if return_aux:
            if token_mask is not None:
                valid = xp.asarray(token_mask, dtype=bool).reshape(-1)
                flat_experts = top_k_indices[valid].reshape(-1)
                counts = (flat_experts[:, None] == xp.arange(self.num_experts)[None, :]).sum(axis=0)
                num_tokens = int(valid.sum())
                router_probs = router_probs[valid]
                host_counts = to_cpu(counts)
            self.last_expert_counts = host_counts
            frequencies = Tensor(counts / float(num_tokens * self.top_k))
            aux_loss = (frequencies * router_probs.mean(axis=0)).sum() * self.num_experts
            return output, aux_loss

        return output

class TransformerBlock(Module):
    def __init__(self, head_size, head_num, ffn_type="relu",use_moe=False, moe_num_experts=8, moe_top_k=2, kv_head_num=None, moe_shared_experts=False):
        super().__init__()
        self.norm_position = "post"
        self.ffn_type = ffn_type
        embedding_dim = head_size * head_num
        self.self_attention = SelfAttention(head_size, head_num, kv_head_num)
        self.layer_norm1 = RMSNorm(embedding_dim)
        self.use_moe = use_moe
        if use_moe:
            self.feed_forward = MoEFeedForward(embedding_dim,num_experts=moe_num_experts,top_k=moe_top_k, use_shared_experts=moe_shared_experts)
        else:
            self.feed_forward = {"relu": FeedForward, "swiglu": SwiGLUFeedForward}[ffn_type](embedding_dim)

        self.layer_norm2 = RMSNorm(embedding_dim)

    def __call__(self, x, cache=None, return_aux=False, token_mask=None, segment_ids=None):
        x = self.layer_norm1(x + self.self_attention(x, cache=cache, segment_ids=segment_ids))

        if self.use_moe:
            if return_aux:
                ffn_output, aux_loss = self.feed_forward(x, return_aux=True, token_mask=token_mask)
                return self.layer_norm2(x + ffn_output), aux_loss

        ffn_output = self.feed_forward(x, token_mask=token_mask) if self.use_moe else self.feed_forward(x)
        return self.layer_norm2(x + ffn_output)
