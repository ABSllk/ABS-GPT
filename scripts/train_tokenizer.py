import argparse
import json
from pathlib import Path
import sys
import sysconfig

import pyarrow.parquet as pq

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train byte BPE from a Parquet training split with a text column.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args(argv)

    sys.path.insert(0, sysconfig.get_path("purelib"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps/tokenizer_training"))
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size - 3, 
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    def texts():
        for batch in pq.ParquetFile(args.train_data).iter_batches(batch_size=1024, columns=["text"]):
            yield batch.column(0).to_pylist()

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    ids = {chr(byte): byte for byte in visible}
    ids.update({chr(256 + i): byte for i, byte in enumerate(b for b in range(256) if b not in visible)})
    merges = []
    for left, right in json.loads(tokenizer.to_str())["model"]["merges"]:
        merges.append([ids[left], ids[right]])
        ids[left + right] = 258 + len(merges)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merges, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer.save(str(args.output.with_suffix(".backend.json")))
    print(f"Saved {args.output}; vocab_size={259 + len(merges)}", flush=True)

if __name__ == "__main__":
    main()
