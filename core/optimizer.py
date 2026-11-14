from core.backend import xp
import json
from core.triton_ops import matmul

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
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def clip_grad_norm(self, max_norm):
        total_norm = 0.0
        for p in self.parameters:
            param_norm = xp.linalg.norm(p.grad)
            total_norm += param_norm ** 2
        total_norm = xp.sqrt(total_norm)

        if total_norm > max_norm:
            scale_factor = max_norm / (total_norm + 1e-6)
            for p in self.parameters:
                p.grad *= scale_factor
        return float(total_norm.item())

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

                m *= beta1
                m += (1 - beta1) * p.grad
                v *= beta2
                v += (1 - beta2) * (p.grad ** 2)

                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                p.data -= lr * (m_hat / (xp.sqrt(v_hat) + eps) + weight_decay * p.data)

class Muon(Optimizer):
    def __init__(self, parameters, lr=0.001, momentum=0.95, weight_decay=0.01): # 此处momentum是系数，而非动量本身
        super().__init__(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def ortholaize(self, update, steps=5):
        X = update.astype(xp.float32, copy=True)
        X /= xp.linalg.norm(X) + 1e-6
        transpose = X.shape[0] > X.shape[1]

        if transpose:
            X = X.T
        for _ in range(steps):
            A = matmul(X, X.T, precision="tf32")
            B = -4.7750 * A + matmul(2.0315 * A, A, precision="tf32")
            X = 3.4445 * X + matmul(B, X, precision="tf32")
        if transpose:
            X = X.T
        return X

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                if p not in self.state:
                    self.state[p] = {
                        "buffer": xp.zeros_like(p.data),
                    }
                buffer = self.state[p]["buffer"]

                buffer *= momentum
                buffer += (1 - momentum) * p.grad
                update = (1 - momentum) * p.grad + momentum * buffer
                update = self.ortholaize(update)

                in_feature, out_feature = p.data.shape
                scale = max(1.0, out_feature / in_feature) ** 0.5

                p.data *= 1- lr * weight_decay
                p.data -= lr * scale * update
