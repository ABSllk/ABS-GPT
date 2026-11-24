from collections import defaultdict
from core.backend import DEVICE, xp
import json

if DEVICE == "cuda":
    from core.triton_ops import (
        adamw_step,
        multi_tensor_norm_sq,
        multi_tensor_scale,
        muon_update,
        newton_schulz_batched,
    )


def clip_grad_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if DEVICE == "cuda" and all(g.dtype == xp.float32 and g.flags.c_contiguous for g in grads):
        ptrs = xp.asarray([g.data.ptr for g in grads], dtype=xp.int64)
        sizes = xp.asarray([g.size for g in grads], dtype=xp.int32)
        total_norm = float(xp.sqrt(multi_tensor_norm_sq(ptrs, sizes).sum()).item())
        if max_norm is not None and total_norm > max_norm:
            multi_tensor_scale(ptrs, sizes, max_norm / (total_norm + 1e-6))
        return total_norm
    total_norm = float(xp.sqrt(sum((g * g).sum() for g in grads)).item())
    if max_norm is not None and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale
    return total_norm

class Optimizer:
    @staticmethod
    def build(model, adamw_config, muon_config=None, name=None):
        if muon_config is None or name == "adamw":
            return [AdamW(model.parameters(), **adamw_config)]

        muon_params, adam_params = [], []
        for name, parameter in model.named_parameters():
            use_muon = (
                parameter.data.ndim == 2 and name.endswith(".weight")
                and not name.startswith(("embedding.", "lm_head."))
                and "router" not in name.split(".")
            )
            (muon_params if use_muon else adam_params).append(parameter)
        return [AdamW(adam_params, **adamw_config), Muon(muon_params, **muon_config)]

    def __init__(self, parameters, lr=0.01, **defaults):
        self.parameters = list(parameters)
        self.state = {}
        self.param_groups = [
            {
                "params": self.parameters,
                "lr": lr,
                **defaults,
            }
        ]

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def clip_grad_norm(self, max_norm):
        return clip_grad_norm(self.parameters, max_norm)

    def state_dict(self, copy=True):
        ids = {p: i for i, p in enumerate(self.parameters)}
        groups = []
        for group in self.param_groups:
            packed = {key: value for key, value in group.items() if key != "params"}
            packed["params"] = [ids[p] for p in group["params"]]
            groups.append(packed)
        
        result = {}
        layout = []

        for i, p in enumerate(self.parameters):
            field = {}
            for name, value in self.state.get(p, {}).items():
                is_array = isinstance(value, xp.ndarray)
                field[name] = is_array
                result[f"state.{i}.{name}"] = value.copy() if copy and is_array else value
            layout.append(field)

        result["meta"] = json.dumps({
            "version": 1,
            "optimizer": type(self).__name__,
            "shapes": [list(p.data.shape) for p in self.parameters],
            "param_groups": groups,
            "layout": layout
        })

        return result

    def load_state_dict(self, state_dict):
        meta = json.loads(str(state_dict["meta"]))
        save_groups = meta["param_groups"]
        new_state = {}
        for i, p in enumerate(self.parameters):
            fields = meta["layout"][i]
            if not fields:
                continue
            
            state = {}
            for name, is_array in fields.items():
                value = state_dict[f"state.{i}.{name}"]
                if is_array:
                    value = xp.asarray(value, dtype=p.data.dtype).copy()
                elif hasattr(value, "item"):
                    value = value.item()
                state[name] = value
            new_state[p] = state
        new_groups = [
            {
                **group,
                "params": [self.parameters[i] for i in group["params"]]
            } for group in save_groups
        ]
        self.state = new_state
        self.param_groups = new_groups


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

class AdamW(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {
                        "step": 0,
                        "m": xp.zeros_like(p.data),
                        "v": xp.zeros_like(p.data),
                    }

                state = self.state[p]
                state["step"] += 1
                step = state["step"]
                m, v = state["m"], state["v"]
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step

                if DEVICE == "cuda":
                    adamw_step(p.grad, lr, beta1, beta2, bc1, bc2, eps, weight_decay, p.data, m, v)
                else:
                    m *= beta1
                    m += (1 - beta1) * p.grad
                    v *= beta2
                    v += (1 - beta2) * (p.grad ** 2)
                    p.data -= lr * ((m / bc1) / (xp.sqrt(v / bc2) + eps) + weight_decay * p.data)

class Muon(Optimizer):
    def __init__(self, parameters, lr=0.001, momentum=0.95, weight_decay=0.01): # 此处momentum是系数，而非动量本身
        super().__init__(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def ortholaize(self, update, steps=5):
        X = update.astype(xp.float32, copy=True)
        norms = xp.sqrt((X * X).sum(axis=(-2, -1), keepdims=True))
        X /= norms + 1e-6
        transpose = X.shape[-2] > X.shape[-1]

        if transpose:
            X = xp.swapaxes(X, -1, -2)
        if DEVICE == "cuda":
            was_2d = X.ndim == 2
            X = newton_schulz_batched(X[None] if was_2d else X, steps)
            if was_2d:
                X = X[0]
        else:
            for _ in range(steps):
                A = X @ xp.swapaxes(X, -1, -2)
                B = -4.7750 * A + (2.0315 * A) @ A
                X = 3.4445 * X + B @ X
        if transpose:
            X = xp.swapaxes(X, -1, -2)
        return X

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            decay = 1 - lr * weight_decay

            buckets = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = {"buffer": xp.zeros_like(p.data)}
                buckets[p.data.shape].append(p)

            for (in_feature, out_feature), bucket in buckets.items():
                grads = xp.stack([p.grad for p in bucket])
                buffers = xp.stack([self.state[p]["buffer"] for p in bucket])
                grad_term = (1 - momentum) * grads
                buffers *= momentum
                buffers += grad_term
                updates = self.ortholaize(grad_term + momentum * buffers)
                step_scale = lr * (max(1.0, out_feature / in_feature) ** 0.5)

                for i, p in enumerate(bucket):
                    buf = self.state[p]["buffer"]
                    if DEVICE == "cuda":
                        muon_update(updates[i], buffers[i], decay, step_scale, p.data, buf)
                    else:
                        buf[...] = buffers[i]
                        p.data *= decay
                        p.data -= step_scale * updates[i]
