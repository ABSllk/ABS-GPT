import pyarrow.parquet as pq
import json


class CharacterTokenizer:
    def __init__(self, chars):
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.chars = sorted(set(chars) - set(special_tokens)) + special_tokens
        self.stoi = {char: idx for idx, char in enumerate(self.chars)}
        self.itos = {idx: char for idx, char in enumerate(self.chars)}
        self.pad_id = self.stoi['<PAD>']
        self.bos_id = self.stoi['<BOS>']
        self.unk_id = self.stoi['<UNK>']
        self.eos_id = self.stoi['<EOS>']

    def encode(self, text):
        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode(self, indices):
        result = []

        for idx in indices:
            token = self.itos.get(idx, '<UNK>')
            if token not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                result.append(token)

        return ''.join(result)

    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_parquet(cls, parquet_path, max_length=None, batch_size=128):
        parquet = pq.ParquetFile(parquet_path)
        chars = set()
        count = 0
        texts = []
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            texts = batch.column(0).to_pylist()
            for text in texts:
                if max_length is not None:
                    text = text[:max_length]
                count += len(text)
                chars.update(text)
                if max_length is not None and count >= max_length:
                    return cls(chars)

        return cls(chars)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            chars = json.load(f)
        return cls(chars)