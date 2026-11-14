import json
from pathlib import Path

import numpy as np

from core.backend import to_cpu


class Checkpoint:
    MODEL_PREFIX = "model."
    OPTIMIZER_PREFIX = "optimizer."

    @staticmethod
    def _as_numpy(value):
        return np.asarray(to_cpu(value))

    @staticmethod
    def _save_arrays(path, arrays):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".tmp")

        with temporary_path.open("wb") as file:
            np.savez_compressed(file, **arrays)

        temporary_path.replace(path)

    @classmethod
    def save_model(cls, path, model):
        arrays = {
            name: cls._as_numpy(parameter.data)
            for name, parameter in model.named_parameters(trainable_only=False)
        }
        cls._save_arrays(path, arrays)

    @classmethod
    def save_training_state(cls, path, model, optimizer, step, best_loss, train_loader=None, metadata=None):
        optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        arrays = {
            f"{cls.MODEL_PREFIX}{name}": cls._as_numpy(parameter.data)
            for name, parameter in model.named_parameters(trainable_only=False)
        }
        if len(optimizers) > 1:
            arrays[f"{cls.OPTIMIZER_PREFIX}count"] = np.asarray(len(optimizers), dtype=np.int64)
        for i, item in enumerate(optimizers):
            prefix = cls.OPTIMIZER_PREFIX if len(optimizers) == 1 else f"{cls.OPTIMIZER_PREFIX}{i}."
            arrays.update({
                f"{prefix}{name}": cls._as_numpy(value)
                for name, value in item.state_dict(copy=False).items()
            })
        arrays["trainer.step"] = np.asarray(step, dtype=np.int64)
        arrays["trainer.best_loss"] = np.asarray(best_loss, dtype=np.float64)
        arrays["checkpoint.version"] = np.asarray(2, dtype=np.int64)
        if metadata is not None:
            arrays["trainer.metadata"] = np.asarray(json.dumps(metadata, allow_nan=False))
        if train_loader is not None:
            arrays["train_loader.state"] = np.asarray(json.dumps(train_loader.state_dict()))

        cls._save_arrays(path, arrays)

    @classmethod
    def load_training_state(cls, path, model, optimizer, train_loader=None):
        optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        with np.load(path, allow_pickle=False) as checkpoint:
            model_state = {
                key.removeprefix(cls.MODEL_PREFIX): checkpoint[key]
                for key in checkpoint.files
                if key.startswith(cls.MODEL_PREFIX)
            }
            model.load_state_dict(model_state)
            for i, item in enumerate(optimizers):
                prefix = cls.OPTIMIZER_PREFIX if len(optimizers) == 1 else f"{cls.OPTIMIZER_PREFIX}{i}."
                optimizer_state = {
                    key.removeprefix(prefix): checkpoint[key]
                    for key in checkpoint.files if key.startswith(prefix)
                }
                item.load_state_dict(optimizer_state)

            if train_loader is not None:
                train_loader.load_state_dict(json.loads(checkpoint["train_loader.state"].item()))

            return {
                "step": int(checkpoint["trainer.step"].item()),
                "best_loss": float(checkpoint["trainer.best_loss"].item()),
                "metadata": json.loads(checkpoint["trainer.metadata"].item()) if "trainer.metadata" in checkpoint else {},
            }
