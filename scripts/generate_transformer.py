import argparse

from pathlib import Path
from engine.run import load_run
from engine.generator import Generator
from dataset.language_model_loader import render_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text with the trained Transformer model."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Text to continue. If omitted, it is read interactively.",
    )
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; lower is more conservative.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Load the best checkpoint and tokenizer from a frozen run")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--greedy", action="store_true", help="Select the most likely token without sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Keep the smallest probability nucleus for sampling")
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer, manifest = load_run(args.run_dir, args.checkpoint)
    context_size = manifest["model"]["context_size"]

    while True:
        if args.prompt is None:
            try:
                prompt = input("Prompt (exit to quit): ")
            except EOFError:
                break
            if prompt.strip().lower() in ("exit", "quit", "退出"):
                break
        else:
            prompt = args.prompt
        instruction = manifest.get("stage") in ("sft", "grpo")
        encoded_prompt = render_prompt(prompt) if instruction else prompt
        start_tokens = [tokenizer.bos_id, *tokenizer.encode(encoded_prompt)]
        generator = Generator(model=model, context_size=context_size, seed=args.seed)
        result = generator.generate(
            start_tokens=start_tokens,
            end_token=tokenizer.eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.greedy,
            top_p=args.top_p,
        )
        generated_text = tokenizer.decode(result["token_ids"], errors="replace")
        print(generated_text if instruction else prompt + generated_text, flush=True)
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
