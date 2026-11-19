"""Packed SFT: python3 -m scripts.train_sft --help."""

import argparse
from pathlib import Path

from core.nn import FusedCrossEntropyLoss
from core.optimizer import AdamW
from dataset.language_model_loader import load_sft_data
from engine.run import TrainingRun, parse_config
from engine.trainer import LanguageModelTrainer


def main(argv=None, *, validation_callback=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cooldown-steps", type=int, default=None)
    parser.add_argument("--cooldown-shape", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--base-run", dest="settings.base_run", type=Path)
    parser.add_argument("--base-checkpoint", dest="settings.base_checkpoint", type=str)
    parser.add_argument("--train-data", dest="settings.data.train.path", type=Path)
    parser.add_argument("--eval-data", dest="settings.data.eval.path", type=Path)
    parser.add_argument("--lr", dest="settings.optimizer.lr", type=float)
    parser.add_argument("--warmup-steps", dest="settings.warmup_steps", type=int)
    parser.add_argument("--lora", dest="settings.lora.enabled", action="store_true")
    parser.add_argument("--lora-rank", dest="settings.lora.config.rank", type=int)
    parser.add_argument("--lora-alpha", dest="settings.lora.config.alpha", type=float)
    parser.add_argument("--lora-targets", dest="settings.lora.config.target_modules", nargs="+")
    parser.add_argument("--batch-size", dest="settings.batch_size", type=int)
    parser.add_argument("--micro-batch-size", dest="settings.micro_batch_size", type=int)
    parser.add_argument("--context-size", dest="settings.context_size", type=int)
    parser.add_argument("--context-buckets", dest="settings.context_buckets", type=int, nargs="+")
    parser.add_argument("--eval-interval", dest="settings.eval_interval", type=int)
    parser.add_argument("--seed", dest="settings.seed", type=int)
    args = parse_config(parser, "sft.json", argv)
    run, model, tokenizer = TrainingRun.finetune(args)
    config = run.config

    loaders = {}
    for split, data in config["data"].items():
        loaders[split] = load_sft_data(
            data["path"],
            tokenizer, 
            config["context_size"], 
            config["batch_size"], 
            seed=config["seed"], 
            context_buckets=config.get("context_buckets"),
        )

    parameters = [p for name, p in model.named_parameters() if not name.startswith("mtp_")]
    optimizer = AdamW(parameters, **config["optimizer"])
    run.start(model, optimizer, loaders["train"], tokenizer)

    trainer = LanguageModelTrainer(
        model, 
        optimizer, 
        FusedCrossEntropyLoss(), 
        micro_batch_size=config["micro_batch_size"], 
        scheduler=run.scheduler([optimizer], args)
    )
    trainer.fit(
        loaders["train"], 
        loaders["eval"], 
        config["planned_steps"], 
        config["eval_interval"], 
        run,
        validation_callback=validation_callback
    )

if __name__ == "__main__":
    main()
