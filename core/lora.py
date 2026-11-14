from dataclasses import dataclass

from core.backend import xp
from core.nn import Linear, Module
from core.tensor import Tensor

@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0

    target_modules: tuple[str, ...] = (
        "W_q",
        "W_k",
        "W_v",
        "output",
    )

    module_prefix: str = "transformer_blocks"

class LoRALinear(Linear):
    def __init__(self, base, rank=8, alpha=16):
        in_dim, out_dim = base.weight.data.shape

        self.weight = base.weight
        self.bias = base.bias
        self.use_triton = base.use_triton

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = Tensor(
            xp.random.randn(in_dim, rank) / in_dim ** 0.5,
            require_grad=True,
        )
        self.lora_B = Tensor(
            xp.zeros((rank, out_dim)),
            require_grad=True,
        )

        for parameter in (self.weight, self.bias):
            parameter.require_grad = False
            parameter.grad = None
    
    def __call__(self, x):
        base_out = super().__call__(x)
        lora_out = ((x @ self.lora_A) @ self.lora_B) * self.scale
        return base_out + lora_out

    def parameters(self):
        return [self.lora_A, self.lora_B]

def inject_lora(model, config):
    targets = [
        (path, parent, key, module)
        for path, parent, key, module in _module_slots(model)
        if isinstance(module, Linear)
        and path.startswith(config.module_prefix + ".")
        and path.rsplit(".", 1)[-1] in config.target_modules
    ]

    for _, parameter in model.named_parameters(trainable_only=False):
        parameter.require_grad = False
        parameter.grad = None

    for path, parent, key, base in targets:
        replacement = LoRALinear(base, rank=config.rank, alpha=config.alpha)

        if isinstance(parent, (list, dict)):
            parent[key] = replacement
        else:
            setattr(parent, key, replacement)

    trainable_count = sum(parameter.data.size for _, parameter in model.named_parameters())

    total_count = sum(parameter.data.size for _, parameter in model.named_parameters(trainable_only=False))

    return {
        "target_paths": [path for path, _, _, _ in targets],
        "trainable_parameters": int(trainable_count),
        "total_parameters": int(total_count),
    }

def _module_slots(root):
    def walk(value, path):
        if not isinstance(value, (Module, list, tuple, dict)):
            return

        if isinstance(value, Module):
            items = vars(value).items()
        elif isinstance(value, dict):
            items = value.items()
        else:
            items = enumerate(value)

        for key, child in items:
            child_path = f"{path}.{key}" if path else str(key)

            if isinstance(child, Module):
                yield child_path, value, key, child

            yield from walk(child, child_path)

    yield from walk(root, "")
