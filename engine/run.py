import json
import shutil
import time
from pathlib import Path

import numpy as np

from core.backend import DEVICE, xp
from core.lora import LoRAConfig, inject_lora
from dataset.language_model_loader import render_prompt
from engine.checkpoint import Checkpoint
from engine.trainer import LRScheduler
from engine.metrics import append_csv, pool_used_mib, read_csv, sha256_bytes, synchronize, truncate_csv, write_json
from models.transformer_language_model import TransformerLanguageModel
from tokenizers.bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_config(parser, filename, argv=None):
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / filename)
    args = parser.parse_args(argv)
    config = None if args.resume else json.loads(args.config.read_text(encoding="utf-8"))
    for name, value in vars(args).copy().items():
        if name.startswith("settings."):
            if not args.resume:
                keys = name.split(".")[1:]
                section = config
                for key in keys[:-1]:
                    section = section[key]
                section[keys[-1]] = str(value.resolve()) if isinstance(value, Path) else value
            delattr(args, name)
    args.settings = config
    return args


class TrainingRun:
    def __init__(self, path, config, resume=False, clock=time.perf_counter):
        self.path, self.config, self.resume = Path(path).resolve(), config, resume
        self.stage = config.get("stage", "pretrain")
        self.monitor = config.setdefault("monitor", "val_loss")
        self.monitor_mode = config.setdefault("monitor_mode", "min")
        self.state = dict(
            completed_steps=0, tokens_seen=0, best_step=0,
            best_metric=float("inf") if self.monitor_mode == "min" else -float("inf"),
            last_metrics={}, train_seconds=0.0, wall_seconds=0.0,
            peak_pool_used_mib=None, train_tokens_per_second=None, stop_reason=None,
        )
        self.previous_wall = 0.0
        self.clock, self.max_wall_seconds = clock, None
        self.eval_at = set(config.get("extra_eval_steps", []))
        self.last_validation_step = -1

    @classmethod
    def sft(cls, args, base):
        config = args.settings
        config["model"] = base["model"]
        config["activation_checkpointing"] = base.get("activation_checkpointing")
        config["base_run"] = str((ROOT / config["base_run"]).resolve())
        config["context_size"] = config["context_size"] or base["model"]["context_size"]
        if not config["lora"]["enabled"]:
            config["lora"] = None
        if config["optimizer"]["lr"] is None:
            config["optimizer"]["lr"] = 1e-4 if config["lora"] else 1e-5
        config["prompt_template"] = render_prompt("{prompt}")
        for data in config["data"].values():
            path = (ROOT / data["path"]).resolve()
            data.update(path=str(path), sha256=sha256_bytes(path.read_bytes()))
        return cls(args.run_dir, config)

    @classmethod
    def transformer(cls, args, clock=time.perf_counter):
        task_started = clock()
        run = args.run_dir.resolve()
        if args.resume:
            config = json.loads((run / "manifest.json").read_text())
            config.setdefault("packing", "documents" if config.get("sampling") == "shuffled_document_packs_without_replacement_v1" else "stream")
        else:
            config = args.settings
            settings, data = config["model"], config["data"]
            run.mkdir(parents=True)
            for split in ("train", "eval"):
                data[f"{split}_path"] = str((ROOT / data[f"{split}_path"]).resolve())
            tokenizer = BPETokenizer.load(ROOT / config["tokenizer_path"])
            tokenizer.save(run / "tokenizer.json")
            settings["vocab_size"] = tokenizer.vocab_size()
            for name, path in config["artifacts"].items():
                if path:
                    shutil.copyfile(ROOT / path, run / f"{name}.json")
        if args.steps is not None:
            config["planned_steps"] = args.steps
        if args.compact_optimizer_state:
            config["optimizer"]["compact_state"] = True
        if args.eval_at and not args.resume:
            config["extra_eval_steps"] = args.eval_at
        config["max_wall_seconds"] = args.max_wall_seconds
        result = cls(run, config, args.resume, clock)
        result.task_started, result.max_wall_seconds = task_started, args.max_wall_seconds
        result.eval_at.update(args.eval_at)
        return result

    def validation_starts(self, train_loader, eval_loader):
        run, manifest = self.path, self.config
        settings, data = manifest["model"], manifest["data"]
        if self.resume:
            starts = np.load(run / "validation_starts.npy")
        else:
            starts = np.random.default_rng(manifest["validation_seed"]).integers(
                0, len(eval_loader.token_ids) - settings["context_size"], size=manifest["validation_windows"])
            np.save(run / "validation_starts.npy", starts)
            manifest["validation_starts_sha256"] = sha256_bytes((run / "validation_starts.npy").read_bytes())
            manifest["token_stream_sha256"] = {"train": sha256_bytes(train_loader.token_ids.tobytes()),
                                              "eval": sha256_bytes(eval_loader.token_ids.tobytes())}
            data.update(train_token_count=len(train_loader.token_ids), eval_token_count=len(eval_loader.token_ids))

        return starts

    def start(self, model, optimizer, loader, tokenizer):
        if self.resume:
            state = Checkpoint.load_training_state(self.path / "last.npz", model, optimizer, loader)
            progress = state["metadata"]
            self.state.update((key, progress[key]) for key in self.state.keys() & progress.keys())
            self.state["completed_steps"], self.state["best_metric"] = state["step"] + 1, state["best_loss"]
            self.previous_wall = self.state["wall_seconds"]
            for name in ("train.csv", "validation.csv", "routing.csv"):
                truncate_csv(self.path / name, self.state["completed_steps"])
            self.last_validation_step = self.state["completed_steps"]
            self.state["last_metrics"] = {
                self.monitor: float(read_csv(self.path / "validation.csv")[-1][self.monitor])
            }
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            tokenizer.save(self.path / "tokenizer.json")
        if not self.resume:
            self.config["artifact_sha256"] = {p.name: sha256_bytes(p.read_bytes()) for p in self.path.glob("*.json")}
        self.config["parameters"] = sum(p.data.size for _, p in model.named_parameters(trainable_only=False))
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        self.config["trainable_parameters"] = sum(p.data.size for item in optimizers for p in item.parameters)
        write_json(self.path / "manifest.json", self.config)

    def scheduler(self, optimizers, args=None):
        base_lrs = [self.config["optimizer"]["lr"]]
        if len(optimizers) > 1:
            base_lrs.append(self.config["muon_optimizer"]["lr"])
        warmup_steps = self.config.get("warmup_steps", 0)
        cooldown = self.config.get("lr_cooldown")
        if cooldown is not None:
            cooldown.setdefault("start_step", warmup_steps)
            cooldown.setdefault("start_lr_ratio", 1.0)
        scheduler = LRScheduler(optimizers, base_lrs, warmup_steps, cooldown)
        if args is not None and args.cooldown_steps is not None:
            start_step = max(self.state["completed_steps"], warmup_steps)
            self.config["lr_cooldown"] = {
                "start_step": start_step, "start_lr_ratio": scheduler.get_factor(start_step),
                "steps": args.cooldown_steps, "min_lr_ratio": args.min_lr_ratio, "shape": args.cooldown_shape,
            }
            scheduler.cooldown = self.config["lr_cooldown"]
        write_json(self.path / "manifest.json", self.config)
        return scheduler

    def begin(self, model):
        state = self.state
        synchronize()
        self.session_start = self.clock()
        if self.stage == "pretrain":
            state["peak_pool_used_mib"] = max(state["peak_pool_used_mib"] or 0, pool_used_mib() or 0)
            if not self.resume:
                Checkpoint.save_model(self.path / "initial.npz", model)
            print(f"Run: {self.path}; parameters: {self.config['parameters']:,}; starting update: {state['completed_steps']}", flush=True)

    def time_exceeded(self):
        if self.max_wall_seconds is None:
            return False
        reserve = min(600.0, self.max_wall_seconds * 0.1)
        return self.clock() - self.task_started >= self.max_wall_seconds - reserve

    def release_cache(self):
        if self.config.get("release_cuda_cache") and DEVICE == "cuda":
            xp.get_default_memory_pool().free_all_blocks()

    def next_batch(self, loader, step):
        synchronize()
        self.step_started = self.clock()
        self.release_cache()
        batch = loader.next_batch()
        if self.stage == "pretrain":
            schedule = self.config.get("context_schedule") or [[1, self.config["model"]["context_size"]]]
            self.context = next(length for start, length in reversed(schedule) if start <= step)
            if self.config["packing"] == "stream":
                batch = {key: value.reshape(-1, self.context) for key, value in batch.items()}
        return batch

    def log_train(self, step, metrics, loader):
        state = self.state
        metrics = dict(metrics)
        tokens = metrics.pop("tokens")
        routing = metrics.pop("routing", None)
        state["completed_steps"] = step
        state["tokens_seen"] += tokens
        synchronize()
        duration = self.clock() - self.step_started
        state["train_seconds"] += duration
        used = pool_used_mib()
        state["peak_pool_used_mib"] = max(state["peak_pool_used_mib"] or 0, used) if used is not None else None
        row = {
            "step": step, "tokens_seen": state["tokens_seen"],
            "train_loss": metrics.pop("train_loss"), "epoch": loader.epoch,
            "learning_rate": metrics.pop("learning_rate"), "grad_norm": metrics.pop("grad_norm"),
        }
        for name in list(metrics):
            if name.endswith("_learning_rate"):
                row[name] = metrics.pop(name)
        row.update(step_seconds=duration, tokens_per_second=tokens / duration,
                   train_seconds=state["train_seconds"], wall_seconds=self.previous_wall + self.clock() - self.session_start,
                   pool_used_mib=used)
        row.update(metrics)
        append_csv(self.path / "train.csv", row)
        if step % self.config.get("log_interval", 1) == 0:
            print(f"Step {step}, train_loss={row['train_loss']:.4f}, tokens={state['tokens_seen']}", flush=True)
            if routing is not None:
                routing["expert_counts"] = json.dumps(routing["expert_counts"])
                append_csv(self.path / "routing.csv", {"step": step, **routing})

    def save(self, model, optimizer, loader, metrics):
        state = self.state
        self.last_validation_step = state["completed_steps"]
        state["last_metrics"] = dict(metrics)
        value = metrics[self.monitor]
        better = value < state["best_metric"] if self.monitor_mode == "min" else value > state["best_metric"]
        if better:
            state["best_metric"], state["best_step"] = value, state["completed_steps"]
            Checkpoint.save_model(self.path / "best.npz", model)
        Checkpoint.save_training_state(self.path / "last.npz", model, optimizer, state["completed_steps"] - 1,
                                       state["best_metric"], loader, metadata=self.snapshot())
        row = {"step": state["completed_steps"], "tokens_seen": state["tokens_seen"]}
        append_csv(self.path / "validation.csv", row | metrics)
        print(f"Step {state['completed_steps']}, {self.monitor}={value:.4f}, best={state['best_metric']:.4f}", flush=True)

    def snapshot(self):
        state = self.state
        state["wall_seconds"] = self.previous_wall + self.clock() - self.session_start
        state["train_tokens_per_second"] = (state["tokens_seen"] / state["train_seconds"]
                                           if state["train_seconds"] > 0 else None)
        return state.copy()

    def finish(self, stop_reason):
        self.state["stop_reason"] = stop_reason
        write_json(self.path / "summary.json", self.snapshot())


def load_run(run_dir, checkpoint="best"):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load(run / "tokenizer.json")
    settings = manifest["model"]
    model = TransformerLanguageModel(**settings)
    model.use_checkpoint = manifest.get("activation_checkpointing") == "transformer_blocks"
    lora = manifest.get("lora")
    if lora is not None:
        inject_lora(model, LoRAConfig(**lora["config"]))
    with np.load(run / f"{checkpoint}.npz", allow_pickle=False) as arrays:
        state = ({key.removeprefix("model."): arrays[key] for key in arrays.files if key.startswith("model.")}
                 if checkpoint == "last" else {key: arrays[key] for key in arrays.files})
        model.load_state_dict(state)
    return model, tokenizer, manifest
