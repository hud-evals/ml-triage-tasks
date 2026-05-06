"""
Lano-cfg (PhysicsLM4-inspired): a layered CFG with a DP-computable next-token distribution.

This module provides:
  - a small, layered CFG generator (fixed grammar per `graph_seed`)
  - an inside/outside-style DP to compute the ground-truth next-token distribution
    along a *valid* generated sequence (used to report KL(model || DP)).

We intentionally keep this minimal and dependency-free (no xlsxwriter/tqdm).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from nmoe.research.physics.data.generators import BOS, EOS, Sample


_NEG_INF = float("-inf")


def _logsumexp(vals: list[float]) -> float:
  if not vals:
    return _NEG_INF
  m = max(vals)
  if m == _NEG_INF:
    return _NEG_INF
  s = 0.0
  for v in vals:
    if v != _NEG_INF:
      s += math.exp(v - m)
  if s <= 0.0:
    return _NEG_INF
  return m + math.log(s)


def _softmax_from_logs(logits: list[float]) -> list[float]:
  m = max(logits)
  if m == _NEG_INF:
    raise ValueError("softmax over all -inf")
  exps = [0.0 if v == _NEG_INF else math.exp(v - m) for v in logits]
  s = sum(exps)
  if s <= 0.0:
    raise ValueError("softmax sum is zero")
  return [x / s for x in exps]


@dataclass(frozen=True)
class CFGNode:
  depth: int
  sym_id: int
  children: list[list[int]] | None  # indices into next depth, or None for terminals


@dataclass(frozen=True)
class CFGGraph:
  depth: int  # terminal depth
  num_sym: int  # number of terminal symbols (T)
  nodes: list[list[CFGNode]]  # nodes[d][i]

  @property
  def start_id(self) -> int:
    return int(self.nodes[0][0].sym_id)

  @property
  def terminals(self) -> list[int]:
    ids = [int(n.sym_id) for n in self.nodes[self.depth]]
    ids.sort()
    return ids


_GRAPH_CACHE: dict[tuple[int, int, int, int, int, int, int], CFGGraph] = {}


def build_layered_cfg(
  *,
  graph_seed: int,
  depth: int,
  num_sym: int,
  deg_min: int,
  deg_max: int,
  len_min: int,
  len_max: int,
  disallow_duplicate_sym: bool,
  disallow_duplicate_seq: bool,
) -> CFGGraph:
  """
  Build a layered CFG similar in spirit to PhysicsLM4's cfg3* family:
    - depth 0 has a single start symbol
    - depths 1..D each have `num_sym` symbols
    - each nonterminal has `deg` productions, each a concatenation of length 2..3
      of symbols from the next depth
  """
  depth = int(depth)
  num_sym = int(num_sym)
  deg_min = int(deg_min)
  deg_max = int(deg_max)
  len_min = int(len_min)
  len_max = int(len_max)
  if depth <= 0:
    raise ValueError(f"depth must be > 0 (got {depth})")
  if num_sym <= 1:
    raise ValueError(f"num_sym must be > 1 (got {num_sym})")
  if not (1 <= deg_min <= deg_max):
    raise ValueError(f"Require 1 <= deg_min <= deg_max (got {deg_min}, {deg_max})")
  if not (1 <= len_min <= len_max):
    raise ValueError(f"Require 1 <= len_min <= len_max (got {len_min}, {len_max})")
  if len_max > 3:
    raise ValueError("Only production lengths up to 3 are supported in this harness")

  rng = random.Random(int(graph_seed))

  sizes = [1] + [num_sym] * int(depth)
  nodes: list[list[CFGNode]] = [[] for _ in range(int(depth) + 1)]

  # Assign IDs: terminals are 1..num_sym; nonterminals are > num_sym.
  next_id = int(num_sym)
  for d in range(int(depth), -1, -1):
    for _ in range(int(sizes[d])):
      if d == int(depth):
        # terminals
        sym_id = len(nodes[d]) + 1
        nodes[d].append(CFGNode(depth=d, sym_id=int(sym_id), children=None))
      else:
        next_id += 1
        nodes[d].append(CFGNode(depth=d, sym_id=int(next_id), children=[]))

  # Populate productions for nonterminals.
  for d in range(int(depth) - 1, -1, -1):
    next_size = int(sizes[d + 1])
    for i, node in enumerate(nodes[d]):
      deg = rng.randint(int(deg_min), int(deg_max))
      prods: list[list[int]] = []
      seen: set[tuple[int, ...]] = set()
      attempts = 0
      while len(prods) < deg:
        attempts += 1
        if attempts > 10_000:
          raise RuntimeError("failed to sample unique productions (relax duplicate constraints)")
        ln = rng.randint(int(len_min), int(len_max))
        if bool(disallow_duplicate_sym):
          choice = rng.sample(range(next_size), k=min(next_size, ln))
          # If next_size < ln (shouldn't happen for our configs), allow duplicates.
          while len(choice) < ln:
            choice.append(rng.randrange(next_size))
        else:
          choice = [rng.randrange(next_size) for _ in range(ln)]
        key = tuple(int(x) for x in choice)
        if bool(disallow_duplicate_seq) and key in seen:
          continue
        seen.add(key)
        prods.append(list(key))

      # Replace node with children populated.
      nodes[d][i] = CFGNode(depth=node.depth, sym_id=node.sym_id, children=prods)

  return CFGGraph(depth=int(depth), num_sym=int(num_sym), nodes=nodes)


def _expand(
  *,
  graph: CFGGraph,
  rng: random.Random,
  depth: int,
  node_idx: int,
) -> list[int]:
  node = graph.nodes[int(depth)][int(node_idx)]
  if node.children is None:
    return [int(node.sym_id)]
  prod = rng.choice(node.children)
  out: list[int] = []
  for child_idx in prod:
    out.extend(_expand(graph=graph, rng=rng, depth=int(depth) + 1, node_idx=int(child_idx)))
  return out


def generate_cfg_sequence(
  *,
  graph: CFGGraph,
  rng: random.Random,
  max_len: int,
) -> list[int]:
  max_len = int(max_len)
  if max_len <= 0:
    raise ValueError(f"max_len must be > 0 (got {max_len})")
  for _ in range(128):
    seq = _expand(graph=graph, rng=rng, depth=0, node_idx=0)
    if len(seq) <= max_len:
      return seq
  raise RuntimeError("failed to sample a CFG sequence within max_len; reduce depth/branching or increase max_len")


def lano_cfg(
  rng: random.Random,
  *,
  graph_seed: int = 0,
  depth: int = 6,
  num_sym: int = 3,
  deg_min: int = 2,
  deg_max: int = 2,
  len_min: int = 2,
  len_max: int = 3,
  disallow_duplicate_sym: bool = True,
  disallow_duplicate_seq: bool = True,
  max_len: int = 256,
  token_base: int = 9400,
) -> Sample:
  """
  Generate a valid terminal sequence from a fixed CFG and return a masked-LM sample.

  Labels supervise the entire generated sequence (including EOS) so this task can
  be mixed with answer-only QA tasks under `loss_mode=answer_only`.
  """
  token_base = int(token_base)
  if token_base <= 0 or token_base + int(num_sym) >= int(BOS):
    raise ValueError("token_base must keep terminal IDs below special-token range")

  key = (
    int(graph_seed),
    int(depth),
    int(num_sym),
    int(deg_min),
    int(deg_max),
    int(len_min),
    int(len_max),
  )
  graph = _GRAPH_CACHE.get(key)
  if graph is None:
    graph = build_layered_cfg(
      graph_seed=int(graph_seed),
      depth=int(depth),
      num_sym=int(num_sym),
      deg_min=int(deg_min),
      deg_max=int(deg_max),
      len_min=int(len_min),
      len_max=int(len_max),
      disallow_duplicate_sym=bool(disallow_duplicate_sym),
      disallow_duplicate_seq=bool(disallow_duplicate_seq),
    )
    _GRAPH_CACHE[key] = graph

  seq_term_ids = generate_cfg_sequence(graph=graph, rng=rng, max_len=int(max_len))
  seq_tokens = [token_base + (t - 1) for t in seq_term_ids]

  tokens = [int(BOS)] + seq_tokens + [int(EOS)]
  labels = [0] + ([1] * len(seq_tokens)) + [1]  # supervise EOS too; padding EOS stays label=0

  return Sample(
    tokens=tokens,
    labels=labels,
    task="lano_cfg",
    metadata={
      "graph_seed": int(graph_seed),
      "depth": int(depth),
      "num_sym": int(num_sym),
      "deg_min": int(deg_min),
      "deg_max": int(deg_max),
      "len_min": int(len_min),
      "len_max": int(len_max),
      "max_len": int(max_len),
      "token_base": int(token_base),
      "seq_len": int(len(seq_tokens)),
    },
  )


def dp_next_token_distribution(
  *,
  graph: CFGGraph,
  seq_term_ids: list[int],
) -> tuple[list[list[float]], list[float]]:
  """
  Compute the DP ground-truth next-token distribution along a valid terminal sequence.

  Returns:
    probs: length L+1; each row is [p(EOS), p(t1), ..., p(tT)] with Σ=1
    probs_chosen: length L+1; probability of the observed next token at each step (EOS at the end)
  """
  ll = int(len(seq_term_ids))
  if ll <= 0:
    raise ValueError("empty sequence")

  # Precompute productions as child symbol IDs (not indices) for inside/outside.
  prods: dict[int, list[list[int]]] = {}
  log_pp: dict[int, float] = {}
  ids_by_depth: list[list[int]] = [[] for _ in range(int(graph.depth))]
  for d in range(int(graph.depth)):
    for node in graph.nodes[d]:
      if node.children is None:
        raise RuntimeError("nonterminal node has no children")
      cur = int(node.sym_id)
      prod_ids: list[list[int]] = []
      for child in node.children:
        prod_ids.append([int(graph.nodes[d + 1][int(ci)].sym_id) for ci in child])
      prods[cur] = prod_ids
      log_pp[cur] = -math.log(float(len(prod_ids)))
      ids_by_depth[d].append(cur)

  terminals = graph.terminals
  start_id = int(graph.start_id)
  max_id = max([start_id, *terminals, *prods.keys()])

  # ---------------- inside ----------------
  inside: list[list[list[float]]] = [[[ _NEG_INF for _ in range(max_id + 1)] for _ in range(ll)] for _ in range(ll)]
  for k, tok in enumerate(seq_term_ids):
    inside[k][k][int(tok)] = 0.0  # log(1)

  # span length
  for span in range(2, ll + 1):
    for i in range(0, ll - span + 1):
      j = i + span - 1
      # bottom-up over depths (terminals already set)
      for d in reversed(range(int(graph.depth))):
        for cur in ids_by_depth[d]:
          lp = log_pp[cur]
          acc: list[float] = []
          for child in prods[cur]:
            if len(child) == 1:
              v = inside[i][j][child[0]]
              if v != _NEG_INF:
                acc.append(lp + v)
              continue
            if len(child) == 2:
              x, y = child
              mids: list[float] = []
              for mid in range(i, j):
                a = inside[i][mid][x]
                b = inside[mid + 1][j][y]
                if a == _NEG_INF or b == _NEG_INF:
                  continue
                mids.append(a + b)
              v = _logsumexp(mids)
              if v != _NEG_INF:
                acc.append(lp + v)
              continue
            if len(child) == 3:
              x, y, z = child
              mids2: list[float] = []
              for mid1 in range(i, j - 1):
                a = inside[i][mid1][x]
                if a == _NEG_INF:
                  continue
                for mid2 in range(mid1 + 1, j):
                  b = inside[mid1 + 1][mid2][y]
                  c = inside[mid2 + 1][j][z]
                  if b == _NEG_INF or c == _NEG_INF:
                    continue
                  mids2.append(a + b + c)
              v = _logsumexp(mids2)
              if v != _NEG_INF:
                acc.append(lp + v)
              continue
            raise ValueError("production length > 3 is not supported")
          inside[i][j][cur] = _logsumexp(acc)

  # ---------------- outside ----------------
  out: list[list[float]] = [[_NEG_INF for _ in range(max_id + 1)] for _ in range(ll + 1)]
  out[0][start_id] = 0.0  # log(1)

  for k in range(0, ll + 1):
    for d in range(int(graph.depth)):
      for cur in ids_by_depth[d]:
        log_out_cur = out[k][cur]
        if log_out_cur == _NEG_INF:
          continue
        lp = log_pp[cur]
        for child in prods[cur]:
          if len(child) == 1:
            x = child[0]
            out[k][x] = _logsumexp([out[k][x], log_out_cur + lp])
            continue
          if len(child) == 2:
            x, y = child
            out[k][x] = _logsumexp([out[k][x], log_out_cur + lp])
            for mid in range(k, ll):
              a = inside[k][mid][x]
              if a == _NEG_INF:
                continue
              out[mid + 1][y] = _logsumexp([out[mid + 1][y], log_out_cur + lp + a])
            continue
          if len(child) == 3:
            x, y, z = child
            out[k][x] = _logsumexp([out[k][x], log_out_cur + lp])
            for mid1 in range(k, ll - 1):
              a = inside[k][mid1][x]
              if a == _NEG_INF:
                continue
              out[mid1 + 1][y] = _logsumexp([out[mid1 + 1][y], log_out_cur + lp + a])
              for mid2 in range(mid1 + 1, ll):
                b = inside[mid1 + 1][mid2][y]
                if b == _NEG_INF:
                  continue
                out[mid2 + 1][z] = _logsumexp([out[mid2 + 1][z], log_out_cur + lp + a + b])
            continue
          raise ValueError("production length > 3 is not supported")

  # ---------------- convert to next-token probs ----------------
  # probs[0]: distribution for the first token (no EOS).
  row0_term = _softmax_from_logs([out[0][t] for t in terminals])
  probs: list[list[float]] = [[0.0, *row0_term]]

  # chosen probs: actual token prob at each step, plus EOS at end.
  term_to_idx = {t: i for i, t in enumerate(terminals)}
  probs_chosen: list[float] = []
  probs_chosen.append(float(row0_term[term_to_idx[int(seq_term_ids[0])]]))

  for i in range(ll):
    denom = out[i][int(seq_term_ids[i])]
    if denom == _NEG_INF:
      raise RuntimeError("DP outside denom is -inf (sequence not parseable under CFG?)")

    row = []
    # EOS probability (prefix is a complete derivation).
    eos = inside[0][i][start_id]
    row.append(0.0 if eos == _NEG_INF else math.exp(eos - denom))

    for t in terminals:
      v = out[i + 1][t]
      row.append(0.0 if v == _NEG_INF else math.exp(v - denom))

    s = sum(row)
    if s <= 0.0:
      raise RuntimeError("DP next-token row has zero mass")
    row = [x / s for x in row]
    probs.append(row)

    if i + 1 < ll:
      probs_chosen.append(float(row[1 + term_to_idx[int(seq_term_ids[i + 1])]]))
    else:
      probs_chosen.append(float(row[0]))  # EOS after last token

  return probs, probs_chosen
