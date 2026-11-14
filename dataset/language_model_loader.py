from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset.token_cache import load_cached_tokens, load_token_dataset, save_cached_tokens

def load_text_tokens(path, tokenizer, max_length=None):
    path = Path(path)
    if path.suffix == ".npy":
        return load_token_dataset(path)
    tokens = load_cached_tokens(path, tokenizer, max_length)
    if tokens is not None:
        return tokens
    values, count = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
        for text in batch.column(0).to_pylist():
            values.extend([tokenizer.bos_id, *tokenizer.encode(text), tokenizer.eos_id])
            count += len(text)
            if max_length is not None and count >= max_length:
                break
        if max_length is not None and count >= max_length:
            break
    tokens = np.asarray(values, dtype=np.int64)
    save_cached_tokens(path, tokenizer, max_length, tokens)
    return tokens

def render_prompt(prompt):
    return f"要求：{prompt}\n\n回答："

def encode_text_example(text, tokenizer, max_length=None):
    tokens = [tokenizer.bos_id, *tokenizer.encode(text), tokenizer.eos_id]
    if max_length is not None and len(tokens) - 1 > max_length:
        raise ValueError("Example exceeds context size")
    return {
        "inputs": np.asarray(tokens[:-1], dtype=np.int64),
        "targets": np.asarray(tokens[1:], dtype=np.int64),
        "loss_mask": np.ones(len(tokens) - 1, dtype=bool)
    }

def encode_sft_example(prompt, completion, tokenizer, max_length):
    prefix = [tokenizer.bos_id, *tokenizer.encode(render_prompt(prompt))]
    suffix = [*tokenizer.encode(completion), tokenizer.eos_id]
    tokens = prefix + suffix
    if len(tokens) - 1 > max_length:
        raise ValueError("Example exceeds context size")
    return {
        "inputs": np.asarray(tokens[:-1], dtype=np.int64),
        "targets": np.asarray(tokens[1:], dtype=np.int64),
        "loss_mask": np.asarray(([False] * len(prefix) + [True] * len(suffix))[1:], dtype=bool)
    }

class TokenDocuments:
    def __init__(self, token_ids, offsets, context_size):
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        counts = (offsets[:, 1] - offsets[:, 0] - 2) // context_size + 1
        document_ids = np.repeat(np.arange(len(offsets)), counts)
        chunk_ids = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        starts = offsets[document_ids, 0] + chunk_ids * context_size
        ends = np.minimum(starts + context_size + 1, offsets[document_ids, 1])
        self.offsets = np.column_stack((starts, ends))
        self.lengths = ends - starts - 1
        self.token_ids = token_ids

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        start, end = self.offsets[index]
        return {
            "inputs": self.token_ids[start:end - 1],
            "targets": self.token_ids[start + 1:end],
            "loss_mask": np.ones(end - start - 1, dtype=bool)
        }

class LanguageModelLoader:
    def __init__(self, dataset, tokenizer, context_size, batch_size, max_length=None, seed=42, *, packing="stream", isolate_samples=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.packing = packing
        self.isolate_samples = isolate_samples
        self.rng = np.random.default_rng(seed)
        if isinstance(dataset, (str, Path)):
            self.token_ids = load_text_tokens(dataset, tokenizer, max_length)
            if packing == "stream":
                dataset = self.token_ids
            else:
                ends = np.flatnonzero(self.token_ids == tokenizer.eos_id) + 1
                if len(ends) == 0 or ends[-1] != len(self.token_ids):
                    ends = np.append(ends, len(self.token_ids))
                starts = np.concatenate(([0], ends[:-1]))
                dataset = TokenDocuments(self.token_ids, np.column_stack((starts, ends)), context_size)
        if packing == "stream":
            if isolate_samples:
                raise ValueError("Attention isolation requires document boundaries")
            self.token_ids = dataset
            self.num_samples = (len(dataset) - 1) // context_size
        else:
            self.examples = dataset
            self.lengths = (np.asarray(dataset.lengths, dtype=np.int64) if hasattr(dataset, "lengths")
                            else np.asarray([len(example["inputs"]) for example in dataset], dtype=np.int64))
            if np.any(self.lengths <= 0) or np.any(self.lengths > context_size):
                raise ValueError("Examples must fit the context size; split long documents before packing")
            self.num_samples = len(dataset)
        self.epoch = 0
        self._start_epoch()

    def _make_packs(self, order):
        packs = []
        buckets = [[] for _ in range(self.context_size + 1)]
        available = 0
        for index in order:
            index = int(index)
            length = int(self.lengths[index])
            candidates = available >> length
            if candidates:
                capacity = length + (candidates & -candidates).bit_length() - 1
                bin_id = buckets[capacity].pop()
                if not buckets[capacity]:
                    available &= ~(1 << capacity)
            else:
                bin_id, capacity = len(packs), self.context_size
                packs.append([])
            packs[bin_id].append(index)
            remaining = capacity - length
            if remaining:
                buckets[remaining].append(bin_id)
                available |= 1 << remaining
        return packs

    def _start_epoch(self):
        self.order = self.rng.permutation(self.num_samples)
        self.epoch += 1
        if self.packing == "documents":
            self.packs = self._make_packs(self.order)
        self.cursor = 0

    def _stream_batch(self, sample_ids):
        positions = sample_ids[:, None] * self.context_size + np.arange(self.context_size)
        return {"inputs": np.asarray(self.token_ids[positions], dtype=np.int64),
                "targets": np.asarray(self.token_ids[positions + 1], dtype=np.int64)}

    def _collate(self, packs):
        shape = (len(packs), self.context_size)
        batch = {"inputs": np.full(shape, self.tokenizer.pad_id, dtype=np.int64),
                 "targets": np.full(shape, self.tokenizer.pad_id, dtype=np.int64),
                 "loss_mask": np.zeros(shape, dtype=bool),
                 "token_mask": np.zeros(shape, dtype=bool),
                 "document_ids": np.full(shape, -1, dtype=np.int64)}
        batch["segment_ids"] = batch["document_ids"] if self.isolate_samples else None
        for row, pack in enumerate(packs):
            offset = 0
            for segment_id, index in enumerate(pack):
                example = self.examples[index]
                end = offset + len(example["inputs"])
                for key in ("inputs", "targets", "loss_mask"):
                    batch[key][row, offset:end] = example[key]
                batch["token_mask"][row, offset:end] = True
                batch["document_ids"][row, offset:end] = segment_id
                offset = end
        return batch

    def next_batch(self):
        units = self.order if self.packing == "stream" else self.packs
        if self.cursor >= len(units):
            self._start_epoch()
            units = self.order if self.packing == "stream" else self.packs
        selected = units[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(selected)
        return self._stream_batch(selected) if self.packing == "stream" else self._collate(selected)

    def iter_eval_batches(self):
        order = np.arange(self.num_samples)
        units = order if self.packing == "stream" else self._make_packs(order)
        for start in range(0, len(units), self.batch_size):
            selected = units[start:start + self.batch_size]
            yield self._stream_batch(selected) if self.packing == "stream" else self._collate(selected)

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "order": self.order.tolist(),
                "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self.order = np.asarray(state["order"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng_state"]
        if self.packing == "documents":
            self.packs = self._make_packs(self.order)

    def __next__(self):
        return self.next_batch()


class BucketedSFTLoader:
    """Pack complete examples by length; visit every pack once per epoch."""

    def __init__(self, examples, tokenizer, contexts, batch_size, seed=42):
        buckets = {length: [] for length in contexts}
        for example in examples:
            length = next(length for length in contexts if len(example["inputs"]) <= length)
            buckets[length].append(example)
        self.loaders = [
            LanguageModelLoader(rows, tokenizer, length, max(1, batch_size * contexts[0] // length),
                                seed=seed + i, packing="documents", isolate_samples=True)
            for i, (length, rows) in enumerate(buckets.items()) if rows
        ]
        self.rng = np.random.default_rng(seed)
        self.epoch = 1
        self._schedule()

    def _schedule(self):
        self.schedule = self.rng.permutation([
            i for i, loader in enumerate(self.loaders)
            for _ in range((len(loader.packs) + loader.batch_size - 1) // loader.batch_size)
        ])
        self.cursor = 0

    def next_batch(self):
        if self.cursor == len(self.schedule):
            self.epoch += 1
            for loader in self.loaders:
                loader._start_epoch()
            self._schedule()
        loader = self.loaders[self.schedule[self.cursor]]
        self.cursor += 1
        return loader.next_batch()

    def iter_eval_batches(self):
        for loader in self.loaders:
            yield from loader.iter_eval_batches()

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "schedule": self.schedule.tolist(),
                "rng_state": self.rng.bit_generator.state,
                "loaders": [loader.state_dict() for loader in self.loaders]}

    def load_state_dict(self, state):
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.schedule = np.asarray(state["schedule"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng_state"]
        for loader, saved in zip(self.loaders, state["loaders"]):
            loader.load_state_dict(saved)


def load_sft_data(path, tokenizer, context_size, batch_size, seed=42, context_buckets=None):
    contexts = context_buckets or [context_size]
    rows = pq.read_table(path, columns=["prompt", "completion"]).to_pylist()
    examples = [encode_sft_example(row["prompt"], row["completion"], tokenizer, contexts[-1]) for row in rows]
    if context_buckets:
        return BucketedSFTLoader(examples, tokenizer, contexts, batch_size, seed)
    return LanguageModelLoader(examples, tokenizer, context_size, batch_size, seed=seed,
                               packing="documents", isolate_samples=True)
