"""Modal probe: score MSMARCO 16-way training tuples with two rerankers, dump
distributions. Substrate is the mxbai-edge-colbert-v0 tech report Table 5 +
section 3.1 observation: Qwen3-Reranker as a distillation teacher produces a
worse downstream ColBERT student (NanoBEIR NDCG@10 0.5991) than BGE-M3
reranker (0.6286), despite Qwen3-Reranker being the larger/more recent
model. The paper's diagnosis: Qwen3-Reranker's scores are bimodal at the
extremes, which collapses the KL-Div distillation target after softmax.

This probe reproduces the underlying distribution evidence. For ~250
MSMARCO queries each with 16 candidates (1 positive + 15 negatives, in the
same shape PyLate's ColBERTKDLoss consumes), it computes both rerankers'
raw scores, the per-query softmax distribution, and the softmax entropy.
The case bundle includes these artifacts; the task asks the agent to
reconstruct the diagnosis without running anything live.

Usage:
  modal run modal_probe.py --action probe --n-queries 250
"""
from __future__ import annotations

import modal

APP_NAME = "mxbai-reranker-teacher"
VOLUME_NAME = "mxbai-reranker-teacher"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python=None,
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "transformers==4.51.3",
        "accelerate",
        "datasets",
        "sentencepiece",
        "numpy",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Both rerankers are run on the same input shape (query, 16 candidates per
# query). Qwen3-Reranker is a generative yes/no scorer with the prompt
# format spec'd by the Qwen3-Reranker model card. BGE-Reranker-v2-m3 is a
# standard cross-encoder. Following the mxbai paper's Table 5 setup we
# compare these two as distillation teachers.

QWEN_PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
QWEN_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@app.function(
    gpu="H100:1",
    cpu=8.0,
    timeout=60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])] if False else [],
)
def probe(run_id: str, n_queries: int, batch_size: int) -> dict:
    import json
    import math
    import os
    import subprocess
    import time

    import numpy as np
    import torch
    from datasets import load_dataset

    out_dir = f"/data/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "hardware.txt"), "w") as f:
        for cmd in (["nvidia-smi"], ["uname", "-a"], ["nproc"]):
            try:
                f.write(f"\n$ {' '.join(cmd)}\n")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                f.write(r.stdout)
                if r.stderr:
                    f.write("[stderr]\n" + r.stderr)
            except Exception as exc:
                f.write(f"[error] {exc}\n")
        f.write("\n$ torch.cuda.get_device_properties\n")
        prop = torch.cuda.get_device_properties(0)
        for kk in ("name", "multi_processor_count", "shared_memory_per_block",
                   "warp_size", "regs_per_multiprocessor",
                   "max_threads_per_multi_processor", "total_memory",
                   "major", "minor", "L2_cache_size"):
            f.write(f"{kk} = {getattr(prop, kk, '?')}\n")

    # ------------------------------------------------------------------
    # Load MSMARCO with BM25-mined hard negatives via Tevatron's
    # preprocessed split (same shape PyLate's KD loss consumes — a
    # positive plus a stack of mined hard negatives per query).
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading Tevatron/msmarco-passage ...")
    ds = load_dataset("Tevatron/msmarco-passage", split="train", streaming=True)
    triples = []
    for row in ds:
        pos = row.get("positive_passages") or []
        neg = row.get("negative_passages") or []
        if not pos or len(neg) < 15:
            continue
        pos_text = (pos[0].get("text") or "").strip()
        neg_texts = [(n.get("text") or "").strip() for n in neg[:15]]
        if not pos_text or any(not n for n in neg_texts):
            continue
        triples.append({
            "qid": row.get("query_id", str(len(triples))),
            "query": (row.get("query") or "").strip(),
            "positive": pos_text,
            "negatives": neg_texts,
        })
        if len(triples) >= n_queries:
            break
    n_queries = len(triples)
    print(f"[{time.strftime('%H:%M:%S')}] collected {n_queries} 16-way query groups")

    # ------------------------------------------------------------------
    # Score with BGE-Reranker-v2-m3 first (smaller, fast). Standard XLM-R
    # cross-encoder, called directly via transformers to avoid FlagEmbedding's
    # tokenizer API drift across transformers versions.
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading BAAI/bge-reranker-v2-m3 ...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    bge_id = "BAAI/bge-reranker-v2-m3"
    bge_tok = AutoTokenizer.from_pretrained(bge_id)
    bge = AutoModelForSequenceClassification.from_pretrained(
        bge_id, torch_dtype=torch.float16
    ).cuda().eval()

    def bge_score_batch(query: str, docs: list[str]) -> list[float]:
        pairs = [[query, d] for d in docs]
        enc = bge_tok(
            pairs, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            logits = bge(**enc).logits.view(-1)
        return logits.float().cpu().tolist()

    bge_scores: list[list[float]] = []
    t0 = time.time()
    for i, t in enumerate(triples):
        all_docs = [t["positive"]] + t["negatives"]
        sub = []
        for j in range(0, len(all_docs), batch_size):
            sub.extend(bge_score_batch(t["query"], all_docs[j:j+batch_size]))
        bge_scores.append(sub)
        if (i + 1) % 25 == 0:
            print(f"  bge {i+1}/{n_queries} ({time.time()-t0:.1f}s)")
    print(f"[{time.strftime('%H:%M:%S')}] bge done in {time.time()-t0:.1f}s")
    del bge, bge_tok
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Score with Qwen3-Reranker-8B. Uses the generative yes/no logit-diff
    # protocol from the Qwen3-Reranker model card.
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading Qwen/Qwen3-Reranker-8B ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    qwen_id = "Qwen/Qwen3-Reranker-8B"
    qwen_tok = AutoTokenizer.from_pretrained(qwen_id, padding_side="left")
    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_id, torch_dtype=torch.bfloat16
    ).cuda().eval()
    yes_id = qwen_tok.convert_tokens_to_ids("yes")
    no_id = qwen_tok.convert_tokens_to_ids("no")
    print(f"  yes_id={yes_id} no_id={no_id}")

    def qwen_score_batch(query: str, docs: list[str]) -> list[float]:
        prompts = []
        for d in docs:
            body = (
                f"<Instruct>: {QWEN_INSTRUCTION}\n"
                f"<Query>: {query}\n"
                f"<Document>: {d}"
            )
            prompts.append(QWEN_PROMPT_PREFIX + body + QWEN_PROMPT_SUFFIX)
        enc = qwen_tok(
            prompts, padding=True, truncation=True, max_length=8192,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = qwen(**enc)
        # Last-token logits for the assistant slot
        logits = out.logits[:, -1, :]
        yes = logits[:, yes_id]
        no = logits[:, no_id]
        # Probability of "yes" given the binary {yes, no} distribution
        prob_yes = torch.softmax(torch.stack([no, yes], dim=-1), dim=-1)[:, 1]
        return prob_yes.float().cpu().tolist()

    qwen_scores: list[list[float]] = []
    t0 = time.time()
    for i, t in enumerate(triples):
        all_docs = [t["positive"]] + t["negatives"]
        sub_scores = []
        for j in range(0, len(all_docs), batch_size):
            sub_scores.extend(qwen_score_batch(t["query"], all_docs[j:j+batch_size]))
        qwen_scores.append(sub_scores)
        if (i + 1) % 25 == 0:
            print(f"  qwen {i+1}/{n_queries} ({time.time()-t0:.1f}s)")
    print(f"[{time.strftime('%H:%M:%S')}] qwen done in {time.time()-t0:.1f}s")

    del qwen, qwen_tok
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Persist raw scores + per-query stats + aggregate distribution stats
    # ------------------------------------------------------------------
    with open(os.path.join(out_dir, "teacher_scores.jsonl"), "w") as f:
        for i, t in enumerate(triples):
            row = {
                "row_idx": i,
                "qid": t["qid"],
                "query": t["query"],
                "bge_scores": bge_scores[i],
                "qwen_scores": qwen_scores[i],
            }
            f.write(json.dumps(row) + "\n")

    def softmax(xs: list[float]) -> list[float]:
        m = max(xs)
        ex = [math.exp(x - m) for x in xs]
        s = sum(ex)
        return [e / s for e in ex]

    def entropy(ps: list[float]) -> float:
        return -sum(p * math.log(p + 1e-12) for p in ps)

    bge_pos = [sc[0] for sc in bge_scores]
    bge_neg = [n for sc in bge_scores for n in sc[1:]]
    qwen_pos = [sc[0] for sc in qwen_scores]
    qwen_neg = [n for sc in qwen_scores for n in sc[1:]]

    # PyLate's KD loss uses min-max normalized scores by default, then KL-Div.
    def minmax(xs: list[float]) -> list[float]:
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-9:
            return [0.5] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]

    per_query = []
    for i in range(len(triples)):
        bge_norm = minmax(bge_scores[i])
        qwen_norm = minmax(qwen_scores[i])
        per_query.append({
            "qid": i,
            "bge_softmax_entropy_raw": entropy(softmax(bge_scores[i])),
            "qwen_softmax_entropy_raw": entropy(softmax(qwen_scores[i])),
            "bge_softmax_entropy_minmax": entropy(softmax(bge_norm)),
            "qwen_softmax_entropy_minmax": entropy(softmax(qwen_norm)),
            "bge_pos_minus_max_neg": bge_scores[i][0] - max(bge_scores[i][1:]),
            "qwen_pos_minus_max_neg": qwen_scores[i][0] - max(qwen_scores[i][1:]),
            "bge_pos_rank": 1 + sum(1 for s in bge_scores[i][1:] if s > bge_scores[i][0]),
            "qwen_pos_rank": 1 + sum(1 for s in qwen_scores[i][1:] if s > qwen_scores[i][0]),
        })
    with open(os.path.join(out_dir, "per_query_stats.jsonl"), "w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")

    # Maximum-entropy reference: log(16) for 16-way uniform
    max_ent_16 = math.log(16)

    def describe(name: str, xs: list[float]) -> dict:
        a = np.array(xs, dtype=np.float64)
        return {
            "name": name,
            "n": int(a.size),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p01": float(np.percentile(a, 1)),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        }

    def fraction_in(xs: list[float], lo: float, hi: float) -> float:
        a = np.array(xs)
        return float(((a >= lo) & (a <= hi)).mean())

    qwen_all = qwen_pos + qwen_neg
    bge_all = bge_pos + bge_neg
    qwen_pos_ent = [p["qwen_softmax_entropy_raw"] for p in per_query]
    bge_pos_ent = [p["bge_softmax_entropy_raw"] for p in per_query]
    qwen_pos_ent_mm = [p["qwen_softmax_entropy_minmax"] for p in per_query]
    bge_pos_ent_mm = [p["bge_softmax_entropy_minmax"] for p in per_query]

    summary = {
        "n_queries": int(n_queries),
        "candidates_per_query": 16,
        "datasets": "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b (triplet-hard)",
        "uniform_16way_entropy_nats": max_ent_16,
        "bge": {
            "model": "BAAI/bge-reranker-v2-m3",
            "score_kind": "cross_encoder_logit (no sigmoid)",
            "all": describe("bge_all", bge_all),
            "pos": describe("bge_pos", bge_pos),
            "neg": describe("bge_neg", bge_neg),
            "fraction_pos_in_0p99_1": fraction_in(bge_pos, 0.99, 1.0),
            "fraction_neg_in_0_0p01": fraction_in(bge_neg, 0.0, 0.01),
            "fraction_all_in_top_or_bottom_1pct": fraction_in(bge_all, 0.0, 0.01) + fraction_in(bge_all, 0.99, 1.0),
            "mean_softmax_entropy_raw_nats": float(np.mean(bge_pos_ent)),
            "mean_softmax_entropy_minmax_nats": float(np.mean(bge_pos_ent_mm)),
        },
        "qwen": {
            "model": "Qwen/Qwen3-Reranker-8B",
            "score_kind": "softmax(yes_logit, no_logit)[yes]",
            "all": describe("qwen_all", qwen_all),
            "pos": describe("qwen_pos", qwen_pos),
            "neg": describe("qwen_neg", qwen_neg),
            "fraction_pos_in_0p99_1": fraction_in(qwen_pos, 0.99, 1.0),
            "fraction_neg_in_0_0p01": fraction_in(qwen_neg, 0.0, 0.01),
            "fraction_all_in_top_or_bottom_1pct": fraction_in(qwen_all, 0.0, 0.01) + fraction_in(qwen_all, 0.99, 1.0),
            "mean_softmax_entropy_raw_nats": float(np.mean(qwen_pos_ent)),
            "mean_softmax_entropy_minmax_nats": float(np.mean(qwen_pos_ent_mm)),
        },
        "ranking_agreement": {
            "bge_pos_first_rate": float(np.mean([p["bge_pos_rank"] == 1 for p in per_query])),
            "qwen_pos_first_rate": float(np.mean([p["qwen_pos_rank"] == 1 for p in per_query])),
            "bge_mean_pos_rank": float(np.mean([p["bge_pos_rank"] for p in per_query])),
            "qwen_mean_pos_rank": float(np.mean([p["qwen_pos_rank"] for p in per_query])),
        },
    }
    with open(os.path.join(out_dir, "distribution_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ASCII histograms for direct human-eye inspection of the bimodal claim
    def ascii_hist(xs: list[float], name: str, nbins: int = 20, width: int = 60) -> str:
        a = np.array(xs)
        # Hard bins on [0, 1] for Qwen (already prob); auto for BGE
        if name.startswith("qwen"):
            edges = np.linspace(0.0, 1.0, nbins + 1)
        else:
            edges = np.linspace(a.min(), a.max(), nbins + 1)
        counts, _ = np.histogram(a, bins=edges)
        peak = max(counts.max(), 1)
        out = [f"# {name}  (n={a.size}, range=[{a.min():.4f}, {a.max():.4f}])"]
        for i in range(nbins):
            bar = "#" * int(round(width * counts[i] / peak))
            out.append(f"  [{edges[i]: .4f}, {edges[i+1]: .4f})  {counts[i]:5d}  {bar}")
        return "\n".join(out)

    hist_txt = "\n\n".join([
        ascii_hist(qwen_pos, "qwen_pos"),
        ascii_hist(qwen_neg, "qwen_neg"),
        ascii_hist(bge_pos, "bge_pos"),
        ascii_hist(bge_neg, "bge_neg"),
    ])
    with open(os.path.join(out_dir, "histograms.txt"), "w") as f:
        f.write(hist_txt)

    volume.commit()
    print(f"[{time.strftime('%H:%M:%S')}] wrote outputs to {out_dir}")
    return {"run_id": run_id, "out_dir": out_dir, "summary": summary}


@app.local_entrypoint()
def main(
    action: str = "probe",
    n_queries: int = 250,
    batch_size: int = 16,
    run_id: str = "",
) -> None:
    import time
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    if action == "probe":
        out = probe.remote(run_id, n_queries, batch_size)
        print(f"[probe] {out}")
    else:
        raise SystemExit(f"unknown action: {action}")
