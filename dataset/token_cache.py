import hashlib
import json
import uuid
from pathlib import Path

import numpy as np


def tokenizer_signature(tokenizer):
    identity = {"merges": tokenizer.merges, "special_ids": [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_token_dataset(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_paths(parquet_path, tokenizer, max_length):
    source = Path(parquet_path)
    identity = {
        "version": 1,
        "source_sha256": file_sha256(source),
        "merges": tokenizer.merges,
        "special_ids": [tokenizer.bos_id, tokenizer.eos_id],
        "max_length_chars": max_length,
        "boundary_policy": "whole_documents_bos_eos",
        "dtype": "int64",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = source.parent / "token_cache" / key
    return base.with_suffix(".npy"), base.with_suffix(".json"), identity


def load_cached_tokens(parquet_path, tokenizer, max_length):
    array_path, _, _ = cache_paths(parquet_path, tokenizer, max_length)
    if not array_path.exists():
        return None
    return np.load(array_path, mmap_mode="r", allow_pickle=False)


def save_cached_tokens(parquet_path, tokenizer, max_length, tokens):
    array_path, metadata_path, identity = cache_paths(parquet_path, tokenizer, max_length)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, np.asarray(tokens, dtype=np.int64), allow_pickle=False)
    metadata = {"identity": identity, "token_count": len(tokens), "array_sha256": file_sha256(temporary)}
    temporary.replace(array_path)
    metadata_temp = metadata_path.with_name(metadata_path.name + "." + uuid.uuid4().hex + ".tmp")
    metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_temp.replace(metadata_path)
    return array_path
