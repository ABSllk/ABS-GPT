import argparse
from functools import partial
from importlib import import_module
from pathlib import Path

from core.optimizer import AdamW
from dataset.language_model_loader import PromptLoader, load_prompt_data
from engine.checkpoint import Checkpoint
from engine.grpo import GRPOLoss, GRPOTrainer
from engine.run import TrainingRun, load_run, parse_config


def main(argv=None, *, validation_callback=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    args = parse_config(parser, "grpo.json", argv)

    run, model, tokenizer = TrainingRun.finetune(args)
    config = run.config

    rows = {}
    for split, data in config["data"].items():
        rows[split] = load_prompt_data(data["path"], tokenizer, config["context_size"], config["max_new_tokens"])
        data["rows"] = len(rows[split])
    train_loader = PromptLoader(rows["train"], config["batch_size"], config["seed"])
    eval_rows = rows["eval"][:config["eval_samples"]]

    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("mtp_")]
    optimizer = AdamW(parameters, **config["optimizer"])
    run.start(model, optimizer, train_loader, tokenizer)

    module_name, function_name = config["reward_fn"].split(":")
    reward_fn = getattr(import_module(module_name), function_name)

    trainer = GRPOTrainer(
        model,
        optimizer,
        tokenizer,
        GRPOLoss(epsilon=config["epsilon"], beta=config["beta"]),
        reward_fn,
        context_size=config["context_size"],
        group_size=config["group_size"],
        max_new_tokens=config["max_new_tokens"],
        micro_batch_size=config["micro_batch_size"],
        max_grad_norm=config["max_grad_norm"],
        scheduler=run.scheduler([optimizer]),
        seed=config["seed"],
    )

    if config["beta"]:
        if not args.resume:
            Checkpoint.save_model(run.path / "reference.npz", model)
        reference_model, _, _ = load_run(run.path, "reference")
        for _, parameter in reference_model.named_parameters(trainable_only=False):
            parameter.require_grad = False
            parameter.grad = None
        trainer.reference_fn = partial(trainer.log_probs, model=reference_model)

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
