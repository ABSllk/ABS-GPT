import pyarrow.parquet as pq
import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self._merge_rules = [(chr(a) + chr(b), chr(259 + i)) for i, (a, b) in enumerate(merges)]
        self.pad_id = 256
        self.bos_id = 257
        self.eos_id = 258

    @classmethod
    def merge_pair(cls, sequence, pair, new_id):
        i = 0
        new_sequence = []
        while i < len(sequence):
            if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
                new_sequence.append(new_id)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        sequence = new_sequence
        return sequence
    
    def encode(self, text):
        sequence = text.encode('utf-8').decode('latin1')
        for pair, new_id in self._merge_rules:
            sequence = sequence.replace(pair, new_id)
        return [ord(symbol) for symbol in sequence]

    def decode(self, indices, errors='strict'):
        result = []

        for idx in indices:
            if idx not in [self.pad_id, self.bos_id, self.eos_id]:
                byte = self.vocab.get(idx)
                result.append(byte)

        return b"".join(result).decode('utf-8', errors=errors)

    @classmethod
    def from_parquet(cls, parquet_path, max_length=None, batch_size=128, max_vocab_size=1024):
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        size = 259
        parquet = pq.ParquetFile(parquet_path)
        sequences = []
        total_length = 0

        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if max_length is not None:
                    remaining = max_length - total_length
                    if remaining <= 0:
                        break
                    text = text[:remaining]
                sequences.append(text.encode('utf-8').decode('latin1'))
                total_length += len(text)
            if max_length is not None and total_length >= max_length:
                break

        while size < max_vocab_size:
            pair_counts = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[pair] < 2:
                break

            pair_ids = (ord(pair[0]), ord(pair[1]))
            vocab[size] = vocab[pair_ids[0]] + vocab[pair_ids[1]]
            merges.append(pair_ids)
            joined_pair = pair[0] + pair[1]
            merged_symbol = chr(size)
            sequences = [seq.replace(joined_pair, merged_symbol) for seq in sequences]
            size += 1

        return cls(vocab, merges)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            merges = [tuple(pair) for pair in json.load(f)]

        vocab = {i: bytes([i]) for i in range(256)}
        for i, pair in enumerate(merges):
            vocab[259 + i] = vocab[pair[0]] + vocab[pair[1]]
        return cls(vocab, merges)

    def vocab_size(self):
        return 259 + len(self.merges)
