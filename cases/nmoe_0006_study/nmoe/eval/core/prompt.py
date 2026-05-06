from __future__ import annotations

from typing import Literal

import torch
import tiktoken


def render_prompts_mc(
    item: dict,
    continuation_delimiter: str,
    *,
    fewshot_examples: list[dict] | None = None,
) -> list[str]:
    fewshot_examples = fewshot_examples or []
    parts: list[str] = []
    for ex in fewshot_examples:
        parts.append(
            f"{ex['query']}{continuation_delimiter}{ex['choices'][ex['gold']]}\n\n"
        )
    prefix = "".join(parts)
    return [f"{prefix}{item['query']}{continuation_delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(
    item: dict,
    continuation_delimiter: str,
    *,
    fewshot_examples: list[dict] | None = None,
) -> list[str]:
    fewshot_examples = fewshot_examples or []
    parts: list[str] = []
    for ex in fewshot_examples:
        parts.append(
            f"{ex['context_options'][ex['gold']]}{continuation_delimiter}{ex['continuation']}\n\n"
        )
    prefix = "".join(parts)
    return [
        f"{prefix}{context_option}{continuation_delimiter}{item['continuation']}"
        for context_option in item["context_options"]
    ]


def render_prompts_lm(
    item: dict,
    continuation_delimiter: str,
    *,
    fewshot_examples: list[dict] | None = None,
) -> tuple[str, str]:
    fewshot_examples = fewshot_examples or []
    parts: list[str] = []
    for ex in fewshot_examples:
        parts.append(
            f"{str(ex['context']).strip()}{continuation_delimiter}{ex['continuation']}\n\n"
        )
    prefix = "".join(parts)
    base = f"{prefix}{str(item['context']).strip()}{continuation_delimiter}"
    prompt_without = base.strip()
    prompt_with = base + str(item["continuation"])
    return prompt_without, prompt_with


def find_common_length(token_sequences: list[list[int]], *, direction: Literal["left", "right"]) -> int:
    """Length of common prefix/suffix across token sequences."""
    if not token_sequences:
        return 0
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        tok = token_sequences[0][idx]
        if not all(seq[idx] == tok for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens: list[list[int]], pad_token_id: int) -> torch.Tensor:
    """Stack token sequences into [B,T] padded on the right."""
    if not tokens:
        raise ValueError("empty tokens")
    bsz = len(tokens)
    seq_len = max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), int(pad_token_id), dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def _encode(enc: tiktoken.Encoding, prompt: str, *, bos_id: int) -> list[int]:
    return [int(bos_id)] + enc.encode(prompt)


def batch_sequences_mc(
    enc: tiktoken.Encoding,
    prompts: list[str],
    *,
    bos_id: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = [_encode(enc, p, bos_id=bos_id) for p in prompts]
    answer_start = find_common_length(tokens, direction="left")
    start_idxs = [answer_start] * len(tokens)
    end_idxs = [len(x) for x in tokens]
    return tokens, start_idxs, end_idxs


def batch_sequences_schema(
    enc: tiktoken.Encoding,
    prompts: list[str],
    *,
    bos_id: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = [_encode(enc, p, bos_id=bos_id) for p in prompts]
    suffix_len = find_common_length(tokens, direction="right")
    end_idxs = [len(x) for x in tokens]
    start_idxs = [ei - suffix_len for ei in end_idxs]
    return tokens, start_idxs, end_idxs


def batch_sequences_lm(
    enc: tiktoken.Encoding,
    prompt_without: str,
    prompt_with: str,
    *,
    bos_id: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens_without = _encode(enc, prompt_without, bos_id=bos_id)
    tokens_with = _encode(enc, prompt_with, bos_id=bos_id)
    start_idx = len(tokens_without)
    end_idx = len(tokens_with)
    if not (start_idx < end_idx and tokens_without == tokens_with[:start_idx]):
        raise ValueError("LM prompt_without must be a token-prefix of prompt_with")
    return [tokens_with], [start_idx], [end_idx]


def crop_to_max_len(
    tokens: list[list[int]],
    start_idxs: list[int],
    end_idxs: list[int],
    *,
    max_len: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    if max_len <= 0:
        return tokens, start_idxs, end_idxs
    new_tokens: list[list[int]] = []
    new_starts: list[int] = []
    new_ends: list[int] = []
    for t, s, e in zip(tokens, start_idxs, end_idxs):
        if len(t) > max_len:
            num_to_crop = len(t) - max_len
            new_tokens.append(t[-max_len:])
            new_starts.append(s - num_to_crop)
            new_ends.append(e - num_to_crop)
        else:
            new_tokens.append(t)
            new_starts.append(s)
            new_ends.append(e)
    return new_tokens, new_starts, new_ends

