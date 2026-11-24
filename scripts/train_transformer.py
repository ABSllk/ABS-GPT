"""Train a Transformer and save its settings, weights and measurements."""

import argparse
import time
from pathlib import Path

import numpy as np

from core.backend import xp
from core.nn import FusedCrossEntropyLoss
from core.optimizer import Optimizer
from dataset.language_model_loader import LanguageModelLoader
from engine.run import ROOT, TrainingRun, parse_config
from engine.trainer import FixedValidation, LanguageModelTrainer
from models.transformer_language_model import TransformerLanguageModel
from tokenizers.bpe_tokenizer import BPETokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/baseline_10M_data100M_s42")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None, help="Total updates, including completed updates")
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--eval-at", type=int, nargs="*", default=[])
    parser.add_argument("--cooldown-steps", type=int, default=None)
    parser.add_argument("--cooldown-shape", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--train-data", dest="settings.data.train_path", type=Path)
    parser.add_argument("--eval-data", dest="settings.data.eval_path", type=Path)
    parser.add_argument("--tokenizer-path", dest="settings.tokenizer_path", type=Path)
    parser.add_argument("--block-num", dest="settings.model.block_num", type=int)
    parser.add_argument("--head-num", dest="settings.model.head_num", type=int)
    parser.add_argument("--kv-head-num", dest="settings.model.kv_head_num", type=int)
    parser.add_argument("--head-size", dest="settings.model.head_size", type=int)
    parser.add_argument("--ffn-type", dest="settings.model.ffn_type", type=str)
    parser.add_argument("--use-mtp", dest="settings.model.use_mtp", action="store_true")
    parser.add_argument("--mtp-loss-weight", dest="settings.model.mtp_loss_weight", type=float)
    parser.add_argument("--mtp-block-num", dest="settings.model.mtp_block_num", type=int)
    parser.add_argument("--use-moe", dest="settings.model.use_moe", action="store_true")
    parser.add_argument("--moe-num-experts", dest="settings.model.moe_num_experts", type=int)
    parser.add_argument("--moe-top-k", dest="settings.model.moe_top_k", type=int)
    parser.add_argument("--moe-aux-weight", dest="settings.model.moe_aux_weight", type=float)
    parser.add_argument("--moe-shared-experts", dest="settings.model.moe_shared_experts", action="store_true")
    parser.add_argument("--context-size", dest="settings.model.context_size", type=int)
    parser.add_argument("--batch-size", dest="settings.batch_size", type=int)
    parser.add_argument("--packing", dest="settings.packing", type=str)
    parser.add_argument("--micro-batch-size", dest="settings.micro_batch_size", type=int)
    parser.add_argument("--release-cuda-cache", dest="settings.release_cuda_cache", action="store_true")
    parser.add_argument("--optimizer", dest="settings.optimizer_name", type=str)
    parser.add_argument("--lr", dest="settings.optimizer.lr", type=float)
    parser.add_argument("--muon-lr", dest="settings.muon_optimizer.lr", type=float)
    parser.add_argument("--betas", dest="settings.optimizer.betas", type=float, nargs=2)
    parser.add_argument("--seed", dest="settings.seed", type=int)
    parser.add_argument("--eval-windows", dest="settings.validation_windows", type=int)
    parser.add_argument("--eval-interval", dest="settings.eval_interval", type=int)
    parser.add_argument("--warmup-steps", dest="settings.warmup_steps", type=int)
    parser.add_argument("--context-schedule", dest="settings.context_schedule", nargs="+",
                        type=lambda value: list(map(int, value.split(":"))))
    parser.add_argument("--checkpoint", dest="settings.activation_checkpointing",
                        action="store_const", const="transformer_blocks")
    return parse_config(parser, "pretrain.json", argv)

def main(validation_callback=None, argv=None):
    args = parse_args(argv)
    run = TrainingRun.transformer(args, clock=time.perf_counter)
    manifest = run.config
    tokenizer = BPETokenizer.load(run.path / "tokenizer.json")
    seed = manifest["seed"]
    if seed is not None:
        np.random.seed(seed)
        xp.random.seed(seed)
    settings, data = manifest["model"], manifest["data"]
    train_loader = LanguageModelLoader(
        data["train_path"], 
        tokenizer, 
        settings["context_size"], 
        manifest["batch_size"], 
        max_length=data["max_length_chars"],
        seed=seed, 
        packing=manifest["packing"]
    )
    eval_loader = LanguageModelLoader(
        data["eval_path"], 
        tokenizer, 
        settings["context_size"], 
        manifest["eval_batch_size"], 
        max_length=data["max_length_chars"],
        seed=manifest["validation_seed"]
    )
    starts = run.validation_starts(train_loader, eval_loader)
    validation = FixedValidation(
        eval_loader.token_ids, 
        starts, 
        settings["context_size"], 
        manifest["eval_batch_size"]
    )
    model = TransformerLanguageModel(**settings)
    model.use_checkpoint = manifest.get("activation_checkpointing") == "transformer_blocks"
    optimizers = Optimizer.build(
        model, 
        manifest["optimizer"], 
        manifest.get("muon_optimizer"), 
        name=manifest.get("optimizer_name")
    )
    run.start(model, optimizers, train_loader, tokenizer)
    trainer = LanguageModelTrainer(
        model, 
        optimizers, 
        FusedCrossEntropyLoss(), 
        manifest["max_grad_norm"], 
        micro_batch_size=manifest.get("micro_batch_size"), 
        scheduler=run.scheduler(optimizers, args),
        return_mtp=model.use_mtp,
    )
    trainer.fit(
        train_loader, 
        validation, 
        manifest["planned_steps"], 
        manifest["eval_interval"], 
        run, 
        validation_callback=validation_callback
    )

if __name__ == "__main__":
    main()
