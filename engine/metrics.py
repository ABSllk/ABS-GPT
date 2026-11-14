"""Small, dependency-free helpers for persisted training measurements."""

import csv
import hashlib
import json
import math
from pathlib import Path

from core.backend import DEVICE, xp


def synchronize():
    if DEVICE == "cuda":
        xp.cuda.Stream.null.synchronize()


def pool_used_mib():
    # Allocated CuPy pool bytes, not total device memory or GPU utilization.
    return xp.get_default_memory_pool().used_bytes() / 2**20 if DEVICE == "cuda" else None


def perplexity(loss):
    return math.exp(loss) if loss < 700 else None


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_csv(path, row):
    path = Path(path)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def truncate_csv(path, completed_steps):
    """Discard rows written after the last committed checkpoint on resume."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["step"]) <= completed_steps]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
