"""Train a policy from rollouts or offline preference data."""

import argparse
from functools import partial
from importlib import import_module
from pathlib import Path

from core.optimizer import AdamW
from dataset.language_model_loader import PromptLoader, load_preference_data, load_prompt_data
from engine.checkpoint import Checkpoint
from engine.post_training import prepare_preference_reference
from engine.post_training_algorithms import (
    DAPOLossTrainer,
    DPOTrainer,
    GRPOTrainer,
    GSPOTrainer,
    IPOTrainer,
    ORPOTrainer,
    SimPOTrainer,
)
from engine.run import TrainingRun, load_run, parse_config


ROLLOUT_TRAINERS = {"grpo": GRPOTrainer, "dr_grpo": GRPOTrainer, "gspo": GSPOTrainer, "dapo_loss": DAPOLossTrainer}
PREFERENCE_TRAINERS = {"dpo": DPOTrainer, "ipo": IPOTrainer, "simpo": SimPOTrainer, "orpo": ORPOTrainer}


def _default_config(args):
    stage = getattr(args, "settings.stage", "grpo")
    return "grpo.json" if stage in ROLLOUT_TRAINERS else "dpo.json"


def _load_data(config, tokenizer):
    stage = config["stage"]
    if stage in ROLLOUT_TRAINERS:
        load = partial(load_prompt_data, max_new_tokens=config["max_new_tokens"])
    else:
        load = load_preference_data

    rows = {}
    for split, data in config["data"].items():
        rows[split] = load(data["path"], tokenizer, config["context_size"])
        data["rows"] = len(rows[split])
    train_loader = PromptLoader(rows["train"], config["batch_size"], config["seed"])
    eval_rows = rows["eval"][:config["eval_samples"]] if stage in ROLLOUT_TRAINERS else rows["eval"]
    return train_loader, eval_rows


def _make_trainer(model, optimizer, tokenizer, config, reference, **common):
    stage = config["stage"]
    if stage in ROLLOUT_TRAINERS:
        module, name = config["reward_fn"].split(":")
        reward_fn = getattr(import_module(module), name)
        epsilon = config[stage]["epsilon"] if stage in {"gspo", "dapo_loss"} else config["epsilon"]
        trainer = ROLLOUT_TRAINERS[stage](
            model, optimizer, tokenizer, reward_fn, 
            kl_weight=config["beta"], 
            clip_epsilon=epsilon,
            context_size=config["context_size"], 
            group_size=config["group_size"], 
            max_new_tokens=config["max_new_tokens"],
            standardize_advantages=config.get("standardize_advantages", True),
            max_grad_norm=config["max_grad_norm"], 
            seed=config["seed"], 
            **common
        )
        if reference is not None:
            trainer.reference_fn = partial(trainer.log_probs, model=reference)
        return trainer

    options = {"beta": config["beta"]} if stage == "dpo" else dict(config[stage])
    return PREFERENCE_TRAINERS[stage](model, optimizer, pad_id=tokenizer.pad_id, **options, **common)


def main(argv=None, *, validation_callback=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--algorithm", dest="settings.stage", choices=(*ROLLOUT_TRAINERS, *PREFERENCE_TRAINERS))
    parser.add_argument("--base-run", dest="settings.base_run", type=Path)
    parser.add_argument("--base-checkpoint", dest="settings.base_checkpoint")
    parser.add_argument("--train-data", dest="settings.data.train.path", type=Path)
    parser.add_argument("--eval-data", dest="settings.data.eval.path", type=Path)
    parser.add_argument("--beta", dest="settings.beta", type=float, help="GRPO KL weight or DPO beta")
    parser.add_argument("--lr", dest="settings.optimizer.lr", type=float)
    parser.add_argument("--batch-size", dest="settings.batch_size", type=int)
    parser.add_argument("--micro-batch-size", dest="settings.micro_batch_size", type=int)
    parser.add_argument("--context-size", dest="settings.context_size", type=int)
    parser.add_argument("--eval-interval", dest="settings.eval_interval", type=int)
    parser.add_argument("--lora", dest="settings.lora.enabled", action="store_true")
    args = parse_config(parser, _default_config, argv)

    run, model, tokenizer = TrainingRun.finetune(args)
    config = run.config
    stage = config["stage"]
    if stage in ROLLOUT_TRAINERS and stage != "grpo":
        config.update(beta=0.0, standardize_advantages=stage != "dr_grpo")
    train_loader, eval_rows = _load_data(config, tokenizer)
    parameters = [p for name, p in model.named_parameters() if not name.startswith("mtp_")]
    optimizer = AdamW(parameters, **config["optimizer"])
    if not args.resume:
        run.start(model, optimizer, train_loader, tokenizer)
    if stage in PREFERENCE_TRAINERS and PREFERENCE_TRAINERS[stage].uses_reference:
        prepare_preference_reference(run, model, tokenizer, {"train": train_loader.rows, "eval": eval_rows})
    if args.resume:
        run.start(model, optimizer, train_loader, tokenizer)

    reference = None
    if stage == "grpo" and config["beta"]:
        if not args.resume:
            Checkpoint.save_model(run.path / "reference.npz", model)
        reference, _, _ = load_run(run.path, "reference", frozen=True)

    trainer = _make_trainer(
        model,
        optimizer,
        tokenizer,
        config,
        reference,
        micro_batch_size=config["micro_batch_size"],
        scheduler=run.scheduler([optimizer])
    )
    trainer.fit(
        train_loader,
        eval_rows,
        config["planned_steps"],
        config["eval_interval"],
        run,
        validation_callback=validation_callback
    )


if __name__ == "__main__":
    main()
