"""
Synthetic data generators for mechanistic interpretability.

Each task has:
- Controllable difficulty (depth, hops, size)
- Known ground-truth answer
- Rich metadata for post-hoc analysis

Based on PhysicsLM (Allen-Zhu et al.), streamlined for our use.

NOTE: Training is standard LM (predict all tokens). The `labels` field marks
which tokens should be supervised when the physics harness runs with
`loss_mode=answer_only`. (For `loss_mode=full`, the harness supervises all
non-padding targets.)
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


# Token ID conventions (staying within reasonable vocab range)
# We intentionally keep tasks in mostly disjoint token subranges to avoid
# accidental "symbol collision" when mixing tasks.
BOS = 9999
EOS = 9998
QUERY_START = 9997
QUERY_END = 9996
ANSWER_START = 9995

# Task-local token ranges (must remain < SYNTHETIC_VOCAB_SIZE in pack.py)
DEPO_ENTITY_MIN = 1
DEPO_ENTITY_MAX = 999

BREVO_NODE_MIN = 1000
BREVO_NODE_MAX = 3999

MANO_OP_BASE = 4000  # 4000..4003
MANO_VAL_BASE = 5000  # 5000..(5000+mod-1)

# Ngram: Markov order-2 language (2-gram next-token prediction)
NGRAM_SYM_MIN = 6000
NGRAM_SYM_MAX = 7999


@dataclass
class Sample:
    """A single synthetic sample."""
    tokens: list[int]
    labels: list[int]  # 1 = answer token (for eval), 0 = context
    task: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def answer_tokens(self) -> list[int]:
        """Extract just the answer tokens (where label=1)."""
        return [t for t, l in zip(self.tokens, self.labels) if l == 1]


# -----------------------------------------------------------------------------
# Depo: Multi-hop retrieval (single-token entities only)
# -----------------------------------------------------------------------------

def depo(
    rng: random.Random,
    n_entities: int = 100,
    max_hops: int = 8,
    gap: int = 0,
    start_max_idx: int | None = None,
) -> Sample:
    """
    Generate a multi-hop retrieval sample.

    Creates a circular chain of entities: A→B→C→...→A
    Then asks: "Starting from X, go k hops forward. What do you reach?"

    Args:
        rng: Random state
        n_entities: Number of entities in the chain (each is single token 1..n_entities)
        max_hops: Maximum hop distance to query

    Returns:
        Sample with tokens, labels, and metadata including hop distance
    """
    if n_entities <= 0 or n_entities > (DEPO_ENTITY_MAX - DEPO_ENTITY_MIN + 1):
        raise ValueError(
            f"n_entities must be in [1, {DEPO_ENTITY_MAX - DEPO_ENTITY_MIN + 1}] "
            f"to stay within Depo token range {DEPO_ENTITY_MIN}..{DEPO_ENTITY_MAX}, got {n_entities}."
        )

    # Single-token entities: DEPO_ENTITY_MIN..DEPO_ENTITY_MIN+n_entities-1
    entities = list(range(DEPO_ENTITY_MIN, DEPO_ENTITY_MIN + n_entities))

    # Shuffle to create random chain order
    order = list(range(n_entities))
    rng.shuffle(order)

    # Build token sequence: chain definition (pairs of linked entities)
    tokens = [BOS]
    gap = int(gap)
    if gap < 0:
        raise ValueError(f"gap must be >= 0 (got {gap})")
    gap_token = 9100  # reserved filler (unused by existing tasks)
    for i in range(n_entities):
        curr = entities[order[i]]
        next_ = entities[order[(i + 1) % n_entities]]
        tokens.append(curr)
        tokens.append(next_)
        if gap:
            tokens.extend([gap_token] * gap)

    # Query: from random position, go k hops
    k = rng.randint(1, max_hops)
    if start_max_idx is None:
        start_idx = rng.randint(0, n_entities - 1)
    else:
        m = int(start_max_idx)
        if m < 0:
            raise ValueError(f"start_max_idx must be >= 0 (got {m})")
        m = min(m, n_entities - 1)
        start_idx = rng.randint(0, m)
    target_idx = (start_idx + k) % n_entities

    start_entity = entities[order[start_idx]]
    target_entity = entities[order[target_idx]]

    # Add query
    tokens.append(8000 + k)  # Encode hop count in token
    tokens.append(start_entity)
    tokens.append(ANSWER_START)

    labels = [0] * len(tokens)

    # Add answer (single token)
    tokens.append(target_entity)
    labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="depo",
        metadata={
            "n_entities": n_entities,
            "hops": k,
            "answer": target_entity,  # The expected answer token
        },
    )


# -----------------------------------------------------------------------------
# Brevo: Topological sort (single-token nodes only)
# -----------------------------------------------------------------------------

def brevo(
    rng: random.Random,
    n_nodes: int = 50,
    max_parents: int = 4,
) -> Sample:
    """
    Generate a topological sort sample.

    Creates a DAG, picks a query node, asks for valid topological order
    of all nodes reachable from query.

    Args:
        rng: Random state
        n_nodes: Number of nodes in the DAG
        max_parents: Maximum parents per node

    Returns:
        Sample with tokens, labels, and metadata including graph depth
    """
    # Single-token nodes: sample unique IDs from BREVO_NODE_MIN..BREVO_NODE_MAX.
    node_range = range(BREVO_NODE_MIN, BREVO_NODE_MAX + 1)
    if n_nodes <= 0 or n_nodes > len(node_range):
        raise ValueError(
            f"n_nodes must be in [1, {len(node_range)}] to stay within Brevo token range "
            f"{BREVO_NODE_MIN}..{BREVO_NODE_MAX}, got {n_nodes}."
        )
    node_tokens = rng.sample(list(node_range), n_nodes)

    # Build DAG: later nodes can depend on earlier ones
    # dag[child_idx] = [parent_idx1, parent_idx2, ...]
    dag: dict[int, list[int]] = defaultdict(list)
    out_degree: dict[int, int] = defaultdict(int)

    n_leaves = max(1, n_nodes // 5)
    for i in range(n_leaves, n_nodes):
        available = [j for j in range(i) if out_degree[j] < max_parents]
        if not available:
            continue
        n_parents = rng.randint(1, min(len(available), max_parents))
        parents = rng.sample(available, n_parents)
        for p in parents:
            dag[i].append(p)
            out_degree[p] += 1

    # Pick query from nodes with dependencies (not leaves)
    candidates = [i for i in range(n_leaves, n_nodes) if dag[i]]
    if not candidates:
        candidates = list(range(n_nodes))
    query_idx = rng.choice(candidates)

    # Find reachable subgraph (indices)
    reachable = _reachable_from(dag, query_idx)
    subdag = {i: [p for p in dag[i] if p in reachable] for i in reachable}

    # Compute valid topological order (indices)
    topo_order_idx = _topological_sort(subdag, rng)
    depth = _graph_depth(subdag, query_idx)

    # Build token sequence: edges in shuffled order
    edges_idx = [(p, c) for c in subdag for p in subdag[c]]
    rng.shuffle(edges_idx)

    tokens = [BOS]
    for p_idx, c_idx in edges_idx:
        tokens.append(node_tokens[p_idx])
        tokens.append(node_tokens[c_idx])

    # Query
    tokens.append(QUERY_START)
    tokens.append(node_tokens[query_idx])
    tokens.append(ANSWER_START)

    labels = [0] * len(tokens)

    # Answer: topological order in token space
    answer_tokens = [node_tokens[idx] for idx in topo_order_idx]
    for tok in answer_tokens:
        tokens.append(tok)
        labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    # Store idx->token mapping for verifier
    idx_to_token = {idx: node_tokens[idx] for idx in reachable}

    return Sample(
        tokens=tokens,
        labels=labels,
        task="brevo",
        metadata={
            "n_nodes": n_nodes,
            "n_reachable": len(reachable),
            "depth": depth,
            "answer_tokens": answer_tokens,  # Expected answer in token space
            "idx_to_token": idx_to_token,    # For verifier to rebuild DAG in token space
            "dag_token_space": {             # DAG expressed in tokens for verifier
                node_tokens[c]: [node_tokens[p] for p in parents]
                for c, parents in subdag.items()
            },
        },
    )


# -----------------------------------------------------------------------------
# Mano: Arithmetic (prefix notation, mod prime)
# -----------------------------------------------------------------------------

def mano(
    rng: random.Random,
    depth: int = 5,
    ops: str = "asm",
    mod: int = 23,
) -> Sample:
    """
    Generate an arithmetic evaluation sample.

    Creates a prefix-notation expression tree, asks for result mod prime.
    Example: * + 3 4 2 = (3+4)*2 = 14

    Args:
        rng: Random state
        depth: Expression tree depth (number of operations)
        ops: Which operations to use (a=add, s=sub, m=mul, d=div)
        mod: Prime modulus for arithmetic

    Returns:
        Sample with tokens, labels, and metadata including expression structure
    """
    op_map = {"a": "+", "s": "-", "m": "*", "d": "/"}
    op_tokens = {
        "+": MANO_OP_BASE + 0,
        "-": MANO_OP_BASE + 1,
        "*": MANO_OP_BASE + 2,
        "/": MANO_OP_BASE + 3,
    }
    available_ops = [op_map[c] for c in ops if c in op_map]

    def build_expr(d: int) -> tuple[list[int], int | None]:
        """Recursively build expression, return (tokens, value)."""
        if d == 0:
            val = rng.randint(0, mod - 1)
            return [MANO_VAL_BASE + val], val

        op = rng.choice(available_ops)
        left_depth = rng.randint(0, d - 1)
        right_depth = d - 1 - left_depth

        left_toks, left_val = build_expr(left_depth)
        right_toks, right_val = build_expr(right_depth)

        toks = [op_tokens[op]] + left_toks + right_toks

        if left_val is None or right_val is None:
            return toks, None

        if op == "+":
            result = (left_val + right_val) % mod
        elif op == "-":
            result = (left_val - right_val + mod) % mod
        elif op == "*":
            result = (left_val * right_val) % mod
        elif op == "/":
            if right_val == 0:
                return toks, None
            result = (left_val * pow(right_val, -1, mod)) % mod
        else:
            result = None

        return toks, result

    # Generate valid expression (retry if division by zero)
    for _ in range(100):
        expr_toks, answer = build_expr(depth)
        if answer is not None:
            break
    else:
        # Fallback: simple addition
        a, b = rng.randint(0, mod - 1), rng.randint(0, mod - 1)
        expr_toks = [op_tokens["+"], MANO_VAL_BASE + a, MANO_VAL_BASE + b]
        answer = (a + b) % mod

    tokens = [BOS] + expr_toks + [ANSWER_START]
    labels = [0] * len(tokens)

    answer_token = MANO_VAL_BASE + answer
    tokens.append(answer_token)
    labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="mano",
        metadata={
            "depth": depth,
            "ops": ops,
            "mod": mod,
            "answer": answer,
            "answer_token": answer_token,
            "expr_len": len(expr_toks),
        },
    )


# -----------------------------------------------------------------------------
# Ngram: Markov order-2 language modeling (bigram transition function)
# -----------------------------------------------------------------------------

def ngram(
    rng: random.Random,
    *,
    n_symbols: int = 512,
    n_steps: int = 128,
    table_seed: int = 0,
) -> Sample:
    """
    Generate a Markov order-2 sample.

    Sequence is defined by a deterministic bigram transition:
      s_{t+1} = f(s_{t-1}, s_t)

    Format: [BOS, s0, s1, ANSWER_START, s2..s_{n_steps+1}, EOS]
    Labels mark the generated continuation tokens as the "answer" region.
    """
    n_symbols = int(n_symbols)
    n_steps = int(n_steps)
    if n_symbols <= 1 or n_symbols > (NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1):
        raise ValueError(
            f"n_symbols must be in [2, {NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1}] "
            f"to stay within Ngram token range {NGRAM_SYM_MIN}..{NGRAM_SYM_MAX}, got {n_symbols}."
        )
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}.")

    # Deterministic "transition table" via a keyed hash (no per-sample table).
    prng = random.Random(int(table_seed))
    m1 = prng.getrandbits(31) | 1  # odd multiplier
    m2 = prng.getrandbits(31) | 1
    b = prng.getrandbits(31)

    s0 = rng.randrange(n_symbols)
    s1 = rng.randrange(n_symbols)

    states = [s0, s1]
    for _ in range(n_steps):
        a = states[-2]
        c = states[-1]
        nxt = (a * m1 + c * m2 + b) % n_symbols
        states.append(int(nxt))

    tokens = [BOS, NGRAM_SYM_MIN + s0, NGRAM_SYM_MIN + s1, ANSWER_START]
    labels = [0] * len(tokens)

    # Answer region: the generated continuation states s2..
    for s in states[2:]:
        tokens.append(NGRAM_SYM_MIN + int(s))
        labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="ngram",
        metadata={
            "n_symbols": n_symbols,
            "n_steps": n_steps,
            "table_seed": int(table_seed),
        },
    )


# -----------------------------------------------------------------------------
# Ngram Polysemy: same bigram, different answer based on mode token
# -----------------------------------------------------------------------------

# Mode tokens (outside NGRAM symbol range)
NGRAM_MODE_A = 5990
NGRAM_MODE_B = 5991


def ngram_polysemy(
    rng: random.Random,
    *,
    n_symbols: int = 512,
    n_steps: int = 128,
    table_seed: int = 0,
) -> Sample:
    """
    Generate a polysemy sample: same bigram hash, different correct answer by mode.

    Mode A: s_{t+1} = f(s_{t-1}, s_t)  (original transition)
    Mode B: s_{t+1} = g(s_{t-1}, s_t)  (different transition, same hash addresses)

    The mode token appears at sequence start. A content-dependent gate conditioned
    on hidden state (which encodes mode) can learn to suppress the wrong prediction.
    PLE cannot disambiguate because the hash address is identical in both modes.

    Format: [BOS, MODE, s0, s1, ANSWER_START, s2..s_{n_steps+1}, EOS]
    """
    n_symbols = int(n_symbols)
    n_steps = int(n_steps)
    if n_symbols <= 1 or n_symbols > (NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1):
        raise ValueError(
            f"n_symbols must be in [2, {NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1}] "
            f"to stay within Ngram token range {NGRAM_SYM_MIN}..{NGRAM_SYM_MAX}, got {n_symbols}."
        )
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}.")

    # Choose mode randomly
    is_mode_b = rng.random() < 0.5
    mode_token = NGRAM_MODE_B if is_mode_b else NGRAM_MODE_A

    # Build two different transition functions from the same seed base
    # Mode A: uses (m1, m2, b)
    # Mode B: uses (m1', m2', b') derived differently
    prng_a = random.Random(int(table_seed))
    m1_a = prng_a.getrandbits(31) | 1
    m2_a = prng_a.getrandbits(31) | 1
    b_a = prng_a.getrandbits(31)

    # Mode B: different coefficients but SAME hash structure
    # (the memory table uses the same hash, but correct answer differs)
    prng_b = random.Random(int(table_seed) + 999999)
    m1_b = prng_b.getrandbits(31) | 1
    m2_b = prng_b.getrandbits(31) | 1
    b_b = prng_b.getrandbits(31)

    s0 = rng.randrange(n_symbols)
    s1 = rng.randrange(n_symbols)
    states = [s0, s1]

    for _ in range(n_steps):
        a = states[-2]
        c = states[-1]
        if is_mode_b:
            nxt = (a * m1_b + c * m2_b + b_b) % n_symbols
        else:
            nxt = (a * m1_a + c * m2_a + b_a) % n_symbols
        states.append(int(nxt))

    # Format: [BOS, MODE, s0, s1, ANSWER_START, s2..., EOS]
    tokens = [BOS, mode_token, NGRAM_SYM_MIN + s0, NGRAM_SYM_MIN + s1, ANSWER_START]
    labels = [0] * len(tokens)

    for s in states[2:]:
        tokens.append(NGRAM_SYM_MIN + int(s))
        labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="ngram_polysemy",
        metadata={
            "n_symbols": n_symbols,
            "n_steps": n_steps,
            "table_seed": int(table_seed),
            "is_mode_b": bool(is_mode_b),
        },
    )


# -----------------------------------------------------------------------------
# Ngram Mixed: structured vs noise (conditionality test)
# -----------------------------------------------------------------------------

def ngram_mixed(
    rng: random.Random,
    *,
    n_symbols: int = 512,
    n_steps: int = 128,
    table_seed: int = 0,
    is_noise: bool = False,
) -> Sample:
    """
    Generate either a structured Markov sample or a noise sample.

    Structured: s_{t+1} = f(s_{t-1}, s_t) (same as ngram)
    Noise: s_{t+1} ~ Uniform(0, n_symbols-1) (no bigram structure)

    The `is_noise` flag is stored in metadata for split evaluation.
    Token format is identical to ngram so memory modules see the same input.
    """
    n_symbols = int(n_symbols)
    n_steps = int(n_steps)
    if n_symbols <= 1 or n_symbols > (NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1):
        raise ValueError(
            f"n_symbols must be in [2, {NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1}] "
            f"to stay within Ngram token range {NGRAM_SYM_MIN}..{NGRAM_SYM_MAX}, got {n_symbols}."
        )
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}.")

    s0 = rng.randrange(n_symbols)
    s1 = rng.randrange(n_symbols)
    states = [s0, s1]

    if is_noise:
        # Noise: uniform random, no bigram structure
        for _ in range(n_steps):
            states.append(rng.randrange(n_symbols))
    else:
        # Structured: deterministic bigram transition (same as ngram)
        prng = random.Random(int(table_seed))
        m1 = prng.getrandbits(31) | 1
        m2 = prng.getrandbits(31) | 1
        b = prng.getrandbits(31)
        for _ in range(n_steps):
            a = states[-2]
            c = states[-1]
            nxt = (a * m1 + c * m2 + b) % n_symbols
            states.append(int(nxt))

    tokens = [BOS, NGRAM_SYM_MIN + s0, NGRAM_SYM_MIN + s1, ANSWER_START]
    labels = [0] * len(tokens)

    for s in states[2:]:
        tokens.append(NGRAM_SYM_MIN + int(s))
        labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="ngram_mixed",
        metadata={
            "n_symbols": n_symbols,
            "n_steps": n_steps,
            "table_seed": int(table_seed),
            "is_noise": bool(is_noise),
        },
    )


# -----------------------------------------------------------------------------
# Ngram Scrambled: unconditionality test (memory should be off)
# -----------------------------------------------------------------------------

def ngram_scrambled(
    rng: random.Random,
    *,
    n_symbols: int = 512,
    n_steps: int = 128,
    table_seed: int = 0,
) -> Sample:
    """
    Generate an ngram-shaped sample where the continuation is *independent* of the prefix.

    This is the cleanest conditionality test for prompt-level memory gating:
    - The input format and symbol range matches `ngram` / `ngram_polysemy`.
    - But the continuation tokens are uniform random, so hashed bigram memory is useless.

    If a learned stack gate is working, it should assign low memory weight (αE≈0)
    on this task while keeping αE high for `ngram_polysemy`.
    """
    n_symbols = int(n_symbols)
    n_steps = int(n_steps)
    if n_symbols <= 1 or n_symbols > (NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1):
        raise ValueError(
            f"n_symbols must be in [2, {NGRAM_SYM_MAX - NGRAM_SYM_MIN + 1}] "
            f"to stay within Ngram token range {NGRAM_SYM_MIN}..{NGRAM_SYM_MAX}, got {n_symbols}."
        )
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}.")

    # Keep the same ngram-shaped prefix (s0, s1), but make the continuation random.
    s0 = rng.randrange(n_symbols)
    s1 = rng.randrange(n_symbols)

    tokens = [BOS, NGRAM_SYM_MIN + s0, NGRAM_SYM_MIN + s1, ANSWER_START]
    labels = [0] * len(tokens)

    for _ in range(n_steps):
        s = rng.randrange(n_symbols)
        tokens.append(NGRAM_SYM_MIN + int(s))
        labels.append(1)

    tokens.append(EOS)
    labels.append(0)

    return Sample(
        tokens=tokens,
        labels=labels,
        task="ngram_scrambled",
        metadata={
            "n_symbols": n_symbols,
            "n_steps": n_steps,
            "table_seed": int(table_seed),
        },
    )


# -----------------------------------------------------------------------------
# Mixed batch generator
# -----------------------------------------------------------------------------

def _lazy_depo_v2(rng: random.Random, **kwargs) -> Sample:
    # Lazy import to avoid import-time cycles (data.generators is used everywhere).
    from nmoe.research.physics.generators.depo_v2 import depo_v2 as _depo_v2

    return _depo_v2(rng, **kwargs)


def _lazy_lano_cfg(rng: random.Random, **kwargs) -> Sample:
    from nmoe.research.physics.generators.lano_cfg import lano_cfg as _lano_cfg

    return _lano_cfg(rng, **kwargs)


@dataclass
class SyntheticMix:
    """
    Configurable mixture of synthetic tasks.

    Usage:
        mix = SyntheticMix(seed=42)
        mix.add("depo", weight=1.0, n_entities=100, max_hops=8)
        mix.add("brevo", weight=0.5, n_nodes=50)
        mix.add("mano", weight=0.5, depth=5)

        for sample in mix.generate(n=1000):
            ...
    """
    seed: int = 42
    tasks: list[tuple[str, float, dict]] = field(default_factory=list)

    _generators: dict[str, Callable] = field(default_factory=lambda: {
        "depo": depo,
        "depo_v2": _lazy_depo_v2,
        "brevo": brevo,
        "mano": mano,
        "ngram": ngram,
        "ngram_polysemy": ngram_polysemy,
        "ngram_mixed": ngram_mixed,
        "ngram_scrambled": ngram_scrambled,
        "lano_cfg": _lazy_lano_cfg,
    })

    def add(self, task: str, weight: float = 1.0, **kwargs) -> "SyntheticMix":
        """Add a task to the mixture."""
        if task not in self._generators:
            raise ValueError(f"Unknown task: {task}. Available: {list(self._generators)}")
        self.tasks.append((task, weight, kwargs))
        return self

    def generate(self, n: int) -> list[Sample]:
        """Generate n samples from the mixture."""
        if not self.tasks:
            raise ValueError("No tasks configured. Use .add() first.")

        rng = random.Random(self.seed)
        total_weight = sum(w for _, w, _ in self.tasks)
        probs = [w / total_weight for _, w, _ in self.tasks]

        samples = []
        for _ in range(n):
            task_name, _, kwargs = rng.choices(self.tasks, weights=probs)[0]
            gen = self._generators[task_name]
            sample = gen(rng, **kwargs)
            samples.append(sample)

        return samples

    def stream(self, rng: random.Random | None = None):
        """Infinite generator of samples."""
        if not self.tasks:
            raise ValueError("No tasks configured. Use .add() first.")

        if rng is None:
            rng = random.Random(self.seed)

        total_weight = sum(w for _, w, _ in self.tasks)
        probs = [w / total_weight for _, w, _ in self.tasks]

        while True:
            task_name, _, kwargs = rng.choices(self.tasks, weights=probs)[0]
            gen = self._generators[task_name]
            yield gen(rng, **kwargs)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _reachable_from(dag: dict[int, list[int]], start: int) -> set[int]:
    """Find all nodes reachable from start by following parent edges."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(dag.get(node, []))
    return visited


def _topological_sort(dag: dict[int, list[int]], rng: random.Random) -> list[int]:
    """Return a valid topological order (randomized among valid orders)."""
    indegree = {node: 0 for node in dag}
    for node in dag:
        for parent in dag[node]:
            if parent in indegree:
                indegree[parent] += 1

    queue = [n for n in dag if indegree[n] == 0]
    order = []

    while queue:
        # Random choice among available nodes (for variety)
        idx = rng.randint(0, len(queue) - 1)
        node = queue.pop(idx)
        order.append(node)
        for parent in dag.get(node, []):
            if parent in indegree:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)

    order.reverse()
    return order


def _graph_depth(dag: dict[int, list[int]], query: int) -> int:
    """Compute depth from query to leaves."""
    from collections import deque

    dist = {query: 0}
    q = deque([query])
    while q:
        node = q.popleft()
        for parent in dag.get(node, []):
            if parent not in dist:
                dist[parent] = dist[node] + 1
                q.append(parent)

    leaves = [n for n in dag if not dag[n]]
    if not leaves:
        return 0
    return max(dist.get(leaf, 0) for leaf in leaves)


# -----------------------------------------------------------------------------
# CLI for quick testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)

    print("=== Depo (8-hop retrieval) ===")
    s = depo(rng, n_entities=20, max_hops=8)
    print(f"Task: {s.task}, Length: {len(s)}")
    print(f"Metadata: n_entities={s.metadata['n_entities']}, hops={s.metadata['hops']}, answer={s.metadata['answer']}")
    print(f"Answer tokens: {s.answer_tokens()}")
    print()

    print("=== Brevo (topological sort) ===")
    s = brevo(rng, n_nodes=15)
    print(f"Task: {s.task}, Length: {len(s)}")
    print(f"Metadata: n_reachable={s.metadata['n_reachable']}, depth={s.metadata['depth']}")
    print(f"Answer tokens: {s.answer_tokens()}")
    print()

    print("=== Mano (arithmetic mod 23) ===")
    s = mano(rng, depth=5, ops="asm")
    print(f"Task: {s.task}, Length: {len(s)}")
    print(f"Metadata: depth={s.metadata['depth']}, answer={s.metadata['answer']}")
    print(f"Tokens: {s.tokens}")
    print(f"Answer tokens: {s.answer_tokens()}")
    print()

    print("=== Mixed batch ===")
    mix = SyntheticMix(seed=42)
    mix.add("depo", weight=1.0, n_entities=50, max_hops=4)
    mix.add("mano", weight=1.0, depth=3)

    samples = mix.generate(6)
    for s in samples:
        print(f"  {s.task}: len={len(s)}, answer={s.answer_tokens()}")
