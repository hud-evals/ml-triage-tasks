"""
Depo v2 (PhysicsLM4-faithful): multi-token "words" + multiple QA pairs per sample.

This follows the structure used in PhysicsLM4's Depo generator:
  - build a cyclic dictionary of N multi-token words
  - append M query/answer pairs, each of the form:
      [HOP(k)] + word(start) + [ANSWER_START] + word(target)
  - labels supervise ONLY the answer segments (including ANSWER_START), so a
    single sample can contain many QA pairs without "answer_only" masking
    accidentally training on intervening query tokens.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from nmoe.research.physics.data.generators import ANSWER_START, BOS, EOS, Sample


@dataclass(frozen=True)
class DepoV2Config:
  # Dictionary
  n_words_max: int = 100
  max_hops: int = 16
  n_qa: int = 10
  separator: bool = False

  # Multi-token word generation
  mini_vocab: int = 3
  min_tlen: int = 5
  max_tlen: int = 7

  # Token ranges (keep disjoint from other tasks)
  word_base: int = 9200  # word tokens use [word_base+1 .. word_base+2*mini_vocab]
  hop_base: int = 8200  # hop tokens use [hop_base+1 .. hop_base+max_hops]
  sep_token: int = 9700

  # PhysicsLM4-style test mode
  qa: bool = False


def _sample_zipf_n(rng: random.Random, *, n_max: int) -> int:
  """
  PhysicsLM4 chooses n ∈ [3, N] with a Zipf-ish prior to vary difficulty.
  """
  if n_max < 3:
    raise ValueError(f"n_words_max must be >= 3 (got {n_max})")
  bias = math.sqrt(float(n_max))
  candidates = list(range(3, int(n_max) + 1))
  weights = [1.0 / (float(i) + bias + 1e-12) for i in candidates]
  # random.Random.choices expects non-normalized weights.
  return int(rng.choices(candidates, weights=weights, k=1)[0])


def _powers_of_two_upto(k: int) -> list[int]:
  out = []
  p = 1
  while p <= k:
    out.append(p)
    p <<= 1
  return out


def _generate_multi_token_words(
  rng: random.Random,
  *,
  n: int,
  mini_vocab: int,
  min_tlen: int,
  max_tlen: int,
  word_base: int,
) -> list[list[int]]:
  if mini_vocab <= 0:
    raise ValueError(f"mini_vocab must be > 0 (got {mini_vocab})")
  if min_tlen <= 0 or max_tlen < min_tlen:
    raise ValueError(f"Require 0 < min_tlen <= max_tlen (got {min_tlen}, {max_tlen})")
  if word_base < 0:
    raise ValueError(f"word_base must be >= 0 (got {word_base})")
  if word_base + 2 * mini_vocab >= int(BOS):
    raise ValueError("word_base+2*mini_vocab must stay below special-token range")

  def sample_word(length: int) -> tuple[int, ...]:
    toks = [rng.randint(1, int(mini_vocab)) for _ in range(int(length))]
    toks[-1] += int(mini_vocab)  # end-of-word marker
    return tuple(int(word_base) + t for t in toks)

  words: set[tuple[int, ...]] = set()
  # Safety: cap attempts so pathological params fail fast.
  max_attempts = max(10_000, 10 * int(n))
  attempts = 0
  while len(words) < int(n):
    if attempts >= max_attempts:
      raise RuntimeError("failed to sample enough unique multi-token words (increase mini_vocab or token length)")
    attempts += 1
    length = rng.randint(int(min_tlen), int(max_tlen))
    words.add(sample_word(length))
  return [list(w) for w in words]


def depo_v2(
  rng: random.Random,
  *,
  n_words_max: int = DepoV2Config.n_words_max,
  max_hops: int = DepoV2Config.max_hops,
  n_qa: int = DepoV2Config.n_qa,
  qa: bool = DepoV2Config.qa,
  separator: bool = DepoV2Config.separator,
  mini_vocab: int = DepoV2Config.mini_vocab,
  min_tlen: int = DepoV2Config.min_tlen,
  max_tlen: int = DepoV2Config.max_tlen,
  word_base: int = DepoV2Config.word_base,
  hop_base: int = DepoV2Config.hop_base,
  sep_token: int = DepoV2Config.sep_token,
) -> Sample:
  """
  Depo v2: multi-token retrieval with multiple QA pairs per sample.
  """
  n_words_max = int(n_words_max)
  max_hops = int(max_hops)
  n_qa = int(n_qa)

  if max_hops <= 0:
    raise ValueError(f"max_hops must be > 0 (got {max_hops})")
  if n_qa <= 0:
    raise ValueError(f"n_qa must be > 0 (got {n_qa})")
  if hop_base < 0:
    raise ValueError(f"hop_base must be >= 0 (got {hop_base})")
  if hop_base + max_hops >= int(BOS):
    raise ValueError("hop_base+max_hops must stay below special-token range")

  # PhysicsLM4 uses a sampled n for train and fixed n for QA/test.
  n = n_words_max if bool(qa) else _sample_zipf_n(rng, n_max=n_words_max)

  vals = _generate_multi_token_words(
    rng,
    n=n,
    mini_vocab=int(mini_vocab),
    min_tlen=int(min_tlen),
    max_tlen=int(max_tlen),
    word_base=int(word_base),
  )
  rng.shuffle(vals)

  # Context: cyclic dictionary edges, presented in random order.
  order = list(range(int(n)))
  rng.shuffle(order)

  tokens: list[int] = [int(BOS)]
  labels: list[int] = [0]

  for idx in order:
    if bool(separator):
      tokens.append(int(sep_token))
      labels.append(0)
    v1 = vals[int(idx)]
    v2 = vals[(int(idx) + 1) % int(n)]
    tokens.extend(v1)
    labels.extend([0] * len(v1))
    tokens.extend(v2)
    labels.extend([0] * len(v2))

  # Queries: multiple QA pairs appended. Supervise only ANSWER_START+answer tokens.
  if bool(qa):
    powers = _powers_of_two_upto(int(max_hops))
    # PhysicsLM4 special-case (not required, but harmless).
    if int(max_hops) == 32 and 24 not in powers:
      powers.append(24)
    if powers and powers[-1] != int(max_hops):
      powers.append(int(max_hops))
  else:
    powers = []

  ks: list[int] = []
  qa_indices = rng.sample(range(int(n)), k=min(int(n), int(n_qa)))
  for idx in qa_indices:
    k = int(rng.choice(powers)) if powers else int(rng.randint(1, int(max_hops)))
    ks.append(k)
    v1 = vals[int(idx)]
    v2 = vals[(int(idx) + k) % int(n)]

    tokens.append(int(hop_base) + k)
    labels.append(0)
    tokens.extend(v1)
    labels.extend([0] * len(v1))

    tokens.append(int(ANSWER_START))
    labels.append(1)  # supervise answer delimiter too (PhysicsLM4-style)
    tokens.extend(v2)
    labels.extend([1] * len(v2))

  tokens.append(int(EOS))
  labels.append(0)

  return Sample(
    tokens=tokens,
    labels=labels,
    task="depo_v2",
    metadata={
      "n_words_max": int(n_words_max),
      "n_words": int(n),
      "max_hops": int(max_hops),
      "n_qa": int(n_qa),
      "ks": ks,
      "mini_vocab": int(mini_vocab),
      "min_tlen": int(min_tlen),
      "max_tlen": int(max_tlen),
      "separator": bool(separator),
      "qa": bool(qa),
      "word_base": int(word_base),
      "hop_base": int(hop_base),
      "sep_token": int(sep_token),
    },
  )

