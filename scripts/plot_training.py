import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.metrics import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    train = read_csv(args.run_dir / "train.csv")
    val = read_csv(args.run_dir / "validation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    tokens = [int(row["tokens_seen"]) for row in train]
    axes[0, 0].plot(tokens, [float(row["train_loss"]) for row in train], label="Train", alpha=0.6)
    axes[0, 0].plot([int(row["tokens_seen"]) for row in val], [float(row["val_loss"]) for row in val], "o-", label="Validation")
    axes[0, 0].set(xlabel="Training tokens seen", ylabel="Cross entropy")
    axes[0, 0].legend()
    axes[0, 1].plot(tokens, [float(row["learning_rate"]) for row in train])
    axes[0, 1].set(xlabel="Training tokens seen", ylabel="Learning rate")
    axes[1, 0].plot([int(row["step"]) for row in train], tokens)
    axes[1, 0].set(xlabel="Optimizer updates", ylabel="Training tokens seen")
    axes[1, 1].plot(tokens, [float(row["tokens_per_second"]) for row in train])
    axes[1, 1].set(xlabel="Training tokens seen", ylabel="Step throughput (tokens/s)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output = args.run_dir / "training_curves.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
