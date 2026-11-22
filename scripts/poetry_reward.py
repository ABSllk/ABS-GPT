import re
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=None)
def is_han(char):
    return unicodedata.name(char, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
    )


def poetry_reward(rollouts):
    rewards = []
    for sample in rollouts:
        text = sample["completion"].strip()
        title, separator, rest = text.partition("\n")
        has_title = bool(separator and re.fullmatch(r"《[^《》\n]+》", title))
        body = rest.strip() if has_title else text
        lines = [part.strip() for part in re.split(r"[，。！？；,!?;\n]+", body) if part.strip()]
        lengths = [sum(is_han(char) for char in line) for line in lines]
        clean = bool(lines) and all(
            length > 0 and all(is_han(char) for char in line.replace("、", ""))
            for line, length in zip(lines, lengths)
        )

        score = 0.1 * has_title + 0.1 * clean
        kind = sample["format_kind"]
        if kind == "uniform":
            expected = sample["format_line_lengths"]
            score += 0.2 * (len(lines) == len(expected))
            correct = sum(actual == target for actual, target in zip(lengths, expected))
            score += 0.6 * correct / max(len(lines), len(expected))
        elif kind == "mixed_length":
            score += 0.8 * (clean and len(set(lengths)) > 1)
        elif kind == "verified_tune":
            tune = sample["verified_tune"]
            score += 0.8 * (clean and has_title and title[1:-1].startswith(tune))
        else:
            score += 0.8 * clean

        if lines:
            score -= 0.2 * (1 - len(set(lines)) / len(lines))
        if sample["finish_reason"] == "length":
            score -= 0.1
        rewards.append(float(score))
    return {"rewards": rewards}
