"""Score MSMARCO 16-way training tuples with a cross-encoder reranker, write
the per-tuple scores to a jsonl that the training step will load as the
distillation target.

This is the scoring pass for the KD teacher ablation. One run per teacher
choice; outputs combined into data/teacher_scores.jsonl downstream.

Originally run on a single H100 via Modal — that wrapper is stripped here;
this script is the kernel that ran inside the modal Function.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# Qwen3-Reranker: generative yes/no logit-diff scorer
# ---------------------------------------------------------------------------

QWEN_PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
QWEN_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def load_qwen(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).cuda().eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    return tok, model, yes_id, no_id


def qwen_score(tok, model, yes_id: int, no_id: int, query: str, docs: list[str]) -> list[float]:
    prompts = []
    for d in docs:
        body = f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {d}"
        prompts.append(QWEN_PROMPT_PREFIX + body + QWEN_PROMPT_SUFFIX)
    enc = tok(
        prompts, padding=True, truncation=True, max_length=8192, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[:, -1, :]
    yes = logits[:, yes_id]
    no = logits[:, no_id]
    prob_yes = torch.softmax(torch.stack([no, yes], dim=-1), dim=-1)[:, 1]
    return prob_yes.float().cpu().tolist()


# ---------------------------------------------------------------------------
# BGE-Reranker-v2-m3: XLM-R cross-encoder, raw logit output
# ---------------------------------------------------------------------------

def load_bge(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda().eval()
    return tok, model


def bge_score(tok, model, query: str, docs: list[str]) -> list[float]:
    pairs = [[query, d] for d in docs]
    enc = tok(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits.view(-1)
    return logits.float().cpu().tolist()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["qwen3_8b", "bge_v2_m3"], required=True)
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--n-queries", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

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
        if len(triples) >= args.n_queries:
            break

    if args.teacher == "qwen3_8b":
        tok, model, yes_id, no_id = load_qwen("Qwen/Qwen3-Reranker-8B")
        score = lambda q, ds: qwen_score(tok, model, yes_id, no_id, q, ds)
    else:
        tok, model = load_bge("BAAI/bge-reranker-v2-m3")
        score = lambda q, ds: bge_score(tok, model, q, ds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        t0 = time.time()
        for i, t in enumerate(triples):
            all_docs = [t["positive"]] + t["negatives"]
            scores = []
            for j in range(0, len(all_docs), args.batch_size):
                scores.extend(score(t["query"], all_docs[j:j+args.batch_size]))
            f.write(json.dumps({
                "qid": t["qid"],
                "query": t["query"],
                "scores": scores,
            }) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[{args.teacher}] {i+1}/{len(triples)}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
