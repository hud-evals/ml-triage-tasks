"""
Ground-truth verifiers for synthetic tasks.

Two verification modes:
1. With Sample object (in-memory, has metadata) - verify_sample()
2. From raw token sequence (packed .npy shards) - verify_from_tokens()

All verification happens in TOKEN SPACE (not index space).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nmoe.research.physics.data.generators import (
    Sample,
    BOS, EOS, QUERY_START, QUERY_END, ANSWER_START,
    DEPO_ENTITY_MIN, DEPO_ENTITY_MAX,
    BREVO_NODE_MIN, BREVO_NODE_MAX,
    MANO_OP_BASE, MANO_VAL_BASE,
)


@dataclass
class VerifyResult:
    """Result of verification."""
    correct: bool
    task: str
    expected: list[int] | None = None
    predicted: list[int] | None = None
    details: dict | None = None


# -----------------------------------------------------------------------------
# Token sequence parsing (for packed .npy verification)
# -----------------------------------------------------------------------------

def detect_task(tokens: list[int]) -> str | None:
    """
    Detect task type from token sequence based on token ranges.

    Returns: "depo", "brevo", "mano", or None if unknown
    """
    # Skip BOS, find first content token
    content_tokens = [t for t in tokens if t not in (BOS, EOS, QUERY_START, QUERY_END, ANSWER_START)]

    if not content_tokens:
        return None

    # Check for Mano ops (4000-4003)
    if any(MANO_OP_BASE <= t <= MANO_OP_BASE + 3 for t in content_tokens):
        return "mano"

    # Check for Brevo nodes (1000-3999)
    if any(BREVO_NODE_MIN <= t <= BREVO_NODE_MAX for t in content_tokens):
        return "brevo"

    # Check for Depo entities (1-999) or hop markers (8000+)
    if any(8000 <= t <= 8100 for t in content_tokens):
        return "depo"
    if any(DEPO_ENTITY_MIN <= t <= DEPO_ENTITY_MAX for t in content_tokens):
        return "depo"

    return None


def parse_depo_tokens(tokens: list[int]) -> dict | None:
    """
    Parse Depo problem from token sequence.

    Format: [BOS, (e1,e2), (e2,e3), ..., hop_marker, start_entity, ANSWER_START, answer, EOS]

    Returns dict with: chain (dict e->next), hops, start, expected_answer
    """
    try:
        # Find ANSWER_START position
        ans_idx = tokens.index(ANSWER_START)

        # Tokens before ANSWER_START are: [BOS, chain pairs..., hop_marker, start_entity]
        pre_answer = tokens[1:ans_idx]  # Skip BOS

        # Last two tokens before ANSWER_START: hop_marker, start_entity
        hop_marker = pre_answer[-2]
        start_entity = pre_answer[-1]
        hops = hop_marker - 8000

        # Chain pairs
        chain_tokens = pre_answer[:-2]
        if len(chain_tokens) % 2 != 0:
            return None

        # Build chain: curr -> next
        chain = {}
        for i in range(0, len(chain_tokens), 2):
            curr = chain_tokens[i]
            next_ = chain_tokens[i + 1]
            chain[curr] = next_

        # Trace hops to find expected answer
        current = start_entity
        for _ in range(hops):
            if current not in chain:
                return None
            current = chain[current]

        return {
            "chain": chain,
            "hops": hops,
            "start": start_entity,
            "expected_answer": current,
        }
    except (ValueError, IndexError):
        return None


def parse_brevo_tokens(tokens: list[int]) -> dict | None:
    """
    Parse Brevo problem from token sequence.

    Format: [BOS, (parent,child), ..., QUERY_START, query, ANSWER_START, topo_order..., EOS]

    Returns dict with: dag (child -> [parents]), query_node, expected_nodes
    """
    try:
        # Find QUERY_START and ANSWER_START
        query_idx = tokens.index(QUERY_START)
        ans_idx = tokens.index(ANSWER_START)

        # Edge pairs between BOS and QUERY_START
        edge_tokens = tokens[1:query_idx]
        if len(edge_tokens) % 2 != 0:
            return None

        # Query node between QUERY_START and ANSWER_START
        query_node = tokens[query_idx + 1]

        # Build DAG: child -> [parents]
        dag = defaultdict(list)
        all_nodes = set()
        for i in range(0, len(edge_tokens), 2):
            parent = edge_tokens[i]
            child = edge_tokens[i + 1]
            dag[child].append(parent)
            all_nodes.add(parent)
            all_nodes.add(child)

        # Expected nodes: all reachable from query via parent edges
        reachable = set()
        stack = [query_node]
        while stack:
            node = stack.pop()
            if node not in reachable:
                reachable.add(node)
                stack.extend(dag.get(node, []))

        return {
            "dag": dict(dag),
            "query_node": query_node,
            "expected_nodes": reachable,
        }
    except (ValueError, IndexError):
        return None


def parse_mano_tokens(tokens: list[int], mod: int = 23) -> dict | None:
    """
    Parse and evaluate Mano expression from token sequence.

    Format: [BOS, op/val tokens..., ANSWER_START, answer_token, EOS]

    Returns dict with: expected_answer (token), expected_value (int)
    """
    try:
        ans_idx = tokens.index(ANSWER_START)
        expr_tokens = tokens[1:ans_idx]  # Skip BOS

        # Evaluate prefix expression
        result = _eval_prefix(expr_tokens, mod)
        if result is None:
            return None

        return {
            "expected_value": result,
            "expected_answer": MANO_VAL_BASE + result,
            "mod": mod,
        }
    except (ValueError, IndexError):
        return None


def _eval_prefix(tokens: list[int], mod: int) -> int | None:
    """Evaluate prefix arithmetic expression mod prime."""
    stack = []

    for tok in reversed(tokens):
        if MANO_VAL_BASE <= tok < MANO_VAL_BASE + mod:
            # Value token
            stack.append(tok - MANO_VAL_BASE)
        elif MANO_OP_BASE <= tok <= MANO_OP_BASE + 3:
            # Operation token
            if len(stack) < 2:
                return None
            left = stack.pop()
            right = stack.pop()

            op_idx = tok - MANO_OP_BASE
            if op_idx == 0:  # +
                result = (left + right) % mod
            elif op_idx == 1:  # -
                result = (left - right + mod) % mod
            elif op_idx == 2:  # *
                result = (left * right) % mod
            elif op_idx == 3:  # /
                if right == 0:
                    return None
                result = (left * pow(right, -1, mod)) % mod
            else:
                return None
            stack.append(result)
        else:
            return None  # Unknown token

    return stack[0] if len(stack) == 1 else None


# -----------------------------------------------------------------------------
# Token-only verification (for packed .npy shards)
# -----------------------------------------------------------------------------

def verify_from_tokens(
    input_tokens: list[int],
    predicted_tokens: list[int],
    mod: int = 23,
) -> VerifyResult:
    """
    Verify model output from raw token sequences (no Sample metadata needed).

    Args:
        input_tokens: Full input sequence (includes problem + ANSWER_START)
        predicted_tokens: Model's predicted answer tokens (after ANSWER_START)
        mod: Modulus for Mano (default 23)

    Returns:
        VerifyResult with correctness and details
    """
    task = detect_task(input_tokens)

    if task == "depo":
        parsed = parse_depo_tokens(input_tokens)
        if parsed is None:
            return VerifyResult(
                correct=False, task="depo",
                predicted=predicted_tokens,
                details={"error": "failed to parse depo tokens"},
            )

        expected = [parsed["expected_answer"]]
        correct = predicted_tokens == expected

        return VerifyResult(
            correct=correct,
            task="depo",
            expected=expected,
            predicted=predicted_tokens,
            details={"hops": parsed["hops"], "start": parsed["start"]},
        )

    elif task == "brevo":
        parsed = parse_brevo_tokens(input_tokens)
        if parsed is None:
            return VerifyResult(
                correct=False, task="brevo",
                predicted=predicted_tokens,
                details={"error": "failed to parse brevo tokens"},
            )

        dag = parsed["dag"]
        expected_nodes = parsed["expected_nodes"]

        # Check 1: correct node set
        if set(predicted_tokens) != expected_nodes:
            return VerifyResult(
                correct=False,
                task="brevo",
                predicted=predicted_tokens,
                details={
                    "error": "wrong node set",
                    "missing": list(expected_nodes - set(predicted_tokens)),
                    "extra": list(set(predicted_tokens) - expected_nodes),
                },
            )

        # Check 2: valid topological order
        seen = set()
        for node in predicted_tokens:
            parents = dag.get(node, [])
            for parent in parents:
                if parent in expected_nodes and parent not in seen:
                    return VerifyResult(
                        correct=False,
                        task="brevo",
                        predicted=predicted_tokens,
                        details={"error": f"dependency violated: {parent} before {node}"},
                    )
            seen.add(node)

        return VerifyResult(
            correct=True,
            task="brevo",
            expected=list(expected_nodes),
            predicted=predicted_tokens,
        )

    elif task == "mano":
        parsed = parse_mano_tokens(input_tokens, mod=mod)
        if parsed is None:
            return VerifyResult(
                correct=False, task="mano",
                predicted=predicted_tokens,
                details={"error": "failed to parse/eval mano tokens"},
            )

        if len(predicted_tokens) != 1:
            return VerifyResult(
                correct=False,
                task="mano",
                expected=[parsed["expected_answer"]],
                predicted=predicted_tokens,
                details={"error": f"expected 1 token, got {len(predicted_tokens)}"},
            )

        correct = predicted_tokens[0] == parsed["expected_answer"]

        return VerifyResult(
            correct=correct,
            task="mano",
            expected=[parsed["expected_answer"]],
            predicted=predicted_tokens,
            details={
                "expected_value": parsed["expected_value"],
                "predicted_value": predicted_tokens[0] - MANO_VAL_BASE if MANO_VAL_BASE <= predicted_tokens[0] < MANO_VAL_BASE + mod else None,
                "mod": mod,
            },
        )

    else:
        return VerifyResult(
            correct=False,
            task="unknown",
            predicted=predicted_tokens,
            details={"error": "could not detect task type from tokens"},
        )


def extract_answer_from_sequence(tokens: list[int]) -> list[int]:
    """
    Extract answer tokens from a full sequence.

    Finds tokens between ANSWER_START and EOS.
    """
    try:
        start_idx = tokens.index(ANSWER_START) + 1
    except ValueError:
        return []

    try:
        end_idx = tokens.index(EOS, start_idx)
    except ValueError:
        end_idx = len(tokens)

    return tokens[start_idx:end_idx]


def extract_input_for_verification(tokens: list[int]) -> list[int]:
    """
    Extract input portion (up to and including ANSWER_START) for verification.
    """
    try:
        ans_idx = tokens.index(ANSWER_START)
        return tokens[:ans_idx + 1]
    except ValueError:
        return tokens


# -----------------------------------------------------------------------------
# Sample-based verification (legacy, when Sample metadata is available)
# -----------------------------------------------------------------------------

def verify_sample(sample: Sample, predicted_tokens: list[int]) -> VerifyResult:
    """Verify model output for any sample type (uses Sample.metadata)."""
    verifiers = {
        "depo": verify_depo,
        "brevo": verify_brevo,
        "mano": verify_mano,
        "ngram": verify_ngram,
    }

    if sample.task not in verifiers:
        return VerifyResult(correct=False, task=sample.task, details={"error": "unknown task"})

    return verifiers[sample.task](sample, predicted_tokens)


def verify_depo(sample: Sample, predicted_tokens: list[int]) -> VerifyResult:
    """Verify Depo output using Sample metadata."""
    expected = [sample.metadata["answer"]]
    correct = predicted_tokens == expected

    return VerifyResult(
        correct=correct,
        task="depo",
        expected=expected,
        predicted=predicted_tokens,
        details={
            "hops": sample.metadata.get("hops"),
            "n_entities": sample.metadata.get("n_entities"),
        },
    )


def verify_brevo(sample: Sample, predicted_tokens: list[int]) -> VerifyResult:
    """Verify Brevo output using Sample metadata."""
    dag_token_space = sample.metadata.get("dag_token_space", {})
    expected_answer_tokens = sample.metadata.get("answer_tokens", [])

    if not dag_token_space:
        return VerifyResult(
            correct=False,
            task="brevo",
            predicted=predicted_tokens,
            details={"error": "no dag_token_space in metadata"},
        )

    dag = {}
    for k, v in dag_token_space.items():
        dag[int(k)] = [int(p) for p in v]

    expected_nodes = set(expected_answer_tokens)

    if set(predicted_tokens) != expected_nodes:
        return VerifyResult(
            correct=False,
            task="brevo",
            expected=expected_answer_tokens,
            predicted=predicted_tokens,
            details={
                "error": "wrong node set",
                "missing": list(expected_nodes - set(predicted_tokens)),
                "extra": list(set(predicted_tokens) - expected_nodes),
                "depth": sample.metadata.get("depth"),
            },
        )

    seen = set()
    for node in predicted_tokens:
        parents = dag.get(node, [])
        for parent in parents:
            if parent in expected_nodes and parent not in seen:
                return VerifyResult(
                    correct=False,
                    task="brevo",
                    expected=expected_answer_tokens,
                    predicted=predicted_tokens,
                    details={
                        "error": f"dependency violated: {parent} must come before {node}",
                        "depth": sample.metadata.get("depth"),
                    },
                )
        seen.add(node)

    return VerifyResult(
        correct=True,
        task="brevo",
        expected=expected_answer_tokens,
        predicted=predicted_tokens,
        details={"depth": sample.metadata.get("depth")},
    )


def verify_mano(sample: Sample, predicted_tokens: list[int]) -> VerifyResult:
    """Verify Mano output using Sample metadata."""
    expected_token = sample.metadata.get("answer_token")

    if len(predicted_tokens) != 1:
        return VerifyResult(
            correct=False,
            task="mano",
            expected=[expected_token] if expected_token else None,
            predicted=predicted_tokens,
            details={
                "error": f"expected 1 token, got {len(predicted_tokens)}",
                "depth": sample.metadata.get("depth"),
            },
        )

    correct = predicted_tokens[0] == expected_token

    return VerifyResult(
        correct=correct,
        task="mano",
        expected=[expected_token],
        predicted=predicted_tokens,
        details={
            "depth": sample.metadata.get("depth"),
            "expected_value": sample.metadata.get("answer"),
            "predicted_value": predicted_tokens[0] - MANO_VAL_BASE if predicted_tokens[0] >= MANO_VAL_BASE else None,
            "mod": sample.metadata.get("mod"),
        },
    )


def verify_ngram(sample: Sample, predicted_tokens: list[int]) -> VerifyResult:
    """Verify Ngram output using Sample labels (exact match across answer region)."""
    expected = sample.answer_tokens()
    correct = predicted_tokens == expected
    return VerifyResult(
        correct=correct,
        task="ngram",
        expected=expected,
        predicted=predicted_tokens,
        details={
            "n_symbols": sample.metadata.get("n_symbols"),
            "n_steps": sample.metadata.get("n_steps"),
            "table_seed": sample.metadata.get("table_seed"),
        },
    )


# -----------------------------------------------------------------------------
# Batch evaluation
# -----------------------------------------------------------------------------

def evaluate_batch(
    samples: list[Sample],
    predictions: list[list[int]],
) -> dict:
    """Evaluate a batch of predictions (Sample-based)."""
    results = {
        "total": 0,
        "correct": 0,
        "by_task": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_difficulty": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    for sample, pred in zip(samples, predictions):
        result = verify_sample(sample, pred)

        results["total"] += 1
        results["by_task"][sample.task]["total"] += 1

        if sample.task == "depo":
            diff_key = f"hops_{sample.metadata.get('hops', '?')}"
        elif sample.task == "brevo":
            diff_key = f"depth_{sample.metadata.get('depth', '?')}"
        elif sample.task == "mano":
            diff_key = f"depth_{sample.metadata.get('depth', '?')}"
        else:
            diff_key = "unknown"

        results["by_difficulty"][diff_key]["total"] += 1

        if result.correct:
            results["correct"] += 1
            results["by_task"][sample.task]["correct"] += 1
            results["by_difficulty"][diff_key]["correct"] += 1

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

    for task in results["by_task"]:
        t = results["by_task"][task]
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    for diff in results["by_difficulty"]:
        d = results["by_difficulty"][diff]
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    results["by_task"] = dict(results["by_task"])
    results["by_difficulty"] = dict(results["by_difficulty"])

    return results


def evaluate_token_batch(
    input_sequences: list[list[int]],
    predicted_answers: list[list[int]],
    mod: int = 23,
) -> dict:
    """
    Evaluate a batch from raw token sequences (packed .npy).

    Args:
        input_sequences: List of input token sequences (problem + ANSWER_START)
        predicted_answers: List of predicted answer token sequences
        mod: Modulus for Mano

    Returns:
        Aggregate metrics by task
    """
    results = {
        "total": 0,
        "correct": 0,
        "by_task": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    for inp, pred in zip(input_sequences, predicted_answers):
        result = verify_from_tokens(inp, pred, mod=mod)

        results["total"] += 1
        results["by_task"][result.task]["total"] += 1

        if result.correct:
            results["correct"] += 1
            results["by_task"][result.task]["correct"] += 1

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

    for task in results["by_task"]:
        t = results["by_task"][task]
        t["accuracy"] = t["correct"] / t["total"] if t["total"] > 0 else 0.0

    results["by_task"] = dict(results["by_task"])
    return results


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from nmoe.research.physics.data.generators import depo, brevo, mano

    rng = random.Random(42)

    print("=== Testing Sample-based verifiers ===\n")

    # Depo
    s = depo(rng, n_entities=20, max_hops=4)
    expected = [s.metadata["answer"]]
    r = verify_depo(s, expected)
    print(f"Depo (correct): {r.correct}, hops={r.details['hops']}")

    # Brevo
    s = brevo(rng, n_nodes=10)
    expected = s.metadata["answer_tokens"]
    r = verify_brevo(s, expected)
    print(f"Brevo (correct): {r.correct}, depth={r.details['depth']}")

    # Mano
    s = mano(rng, depth=3)
    expected = [s.metadata["answer_token"]]
    r = verify_mano(s, expected)
    print(f"Mano (correct): {r.correct}, answer={s.metadata['answer']}")

    print("\n=== Testing token-only verification ===\n")

    # Depo from tokens
    s = depo(rng, n_entities=20, max_hops=4)
    input_toks = extract_input_for_verification(s.tokens)
    answer_toks = extract_answer_from_sequence(s.tokens)
    r = verify_from_tokens(input_toks, answer_toks)
    print(f"Depo from tokens: {r.correct}, hops={r.details.get('hops')}")

    # Wrong answer
    r = verify_from_tokens(input_toks, [999])
    print(f"Depo wrong answer: {r.correct}")

    # Brevo from tokens
    s = brevo(rng, n_nodes=10)
    input_toks = extract_input_for_verification(s.tokens)
    answer_toks = extract_answer_from_sequence(s.tokens)
    r = verify_from_tokens(input_toks, answer_toks)
    print(f"Brevo from tokens: {r.correct}")

    # Mano from tokens
    s = mano(rng, depth=3, mod=23)
    input_toks = extract_input_for_verification(s.tokens)
    answer_toks = extract_answer_from_sequence(s.tokens)
    r = verify_from_tokens(input_toks, answer_toks, mod=23)
    print(f"Mano from tokens: {r.correct}, expected={r.expected}, predicted={r.predicted}")

    print("\n=== Token batch evaluation ===")

    samples = [depo(rng, max_hops=h) for h in [2, 4]] + [mano(rng, depth=d) for d in [2, 3]]
    inputs = [extract_input_for_verification(s.tokens) for s in samples]
    answers = [extract_answer_from_sequence(s.tokens) for s in samples]

    results = evaluate_token_batch(inputs, answers)
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"By task: {results['by_task']}")
