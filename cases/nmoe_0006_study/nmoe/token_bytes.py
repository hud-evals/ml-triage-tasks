"""Token-byte accounting for tokenizer-agnostic bpb metrics.

This is intentionally small and deterministic. It uses tiktoken when available.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _token_bytes_py(tokenizer: str, vocab_size: int) -> tuple[int, ...]:
  import tiktoken  # local import: keep training import surface small

  enc = tiktoken.get_encoding(str(tokenizer))
  V = int(vocab_size)
  out: list[int] = [0] * V
  for i in range(V):
    try:
      b = enc.decode_single_token_bytes(int(i))
    except Exception:
      # Special tokens and padded IDs (e.g. 50257..50303 for GPT-2 padded vocab)
      out[i] = 0
      continue
    out[i] = int(len(b))
  return tuple(out)


def token_bytes(tokenizer: str, vocab_size: int, *, device: torch.device | None = None) -> torch.Tensor:
  """Returns int32 [vocab_size] tensor of per-token decoded byte lengths.

  Contract:
    - 0 for special tokens and padded IDs.
    - Deterministic across processes.
  """
  tb = torch.tensor(_token_bytes_py(str(tokenizer), int(vocab_size)), dtype=torch.int32)
  if device is not None:
    tb = tb.to(device=device)
  return tb


def loss_nats_to_bpb(loss_sum_nats: torch.Tensor, total_bytes: torch.Tensor) -> torch.Tensor:
  """Convert summed NLL (nats) to bits-per-byte over total_bytes."""
  ln2 = math.log(2.0)
  return loss_sum_nats / (float(ln2) * total_bytes.clamp(min=1.0))

