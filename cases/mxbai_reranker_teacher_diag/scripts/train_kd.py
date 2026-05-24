"""Our KD training entry point. Forked from
`examples/train/knowledge_distillation.py` (pylate's canonical example).
Two things changed vs the canonical script:

  1. Backbone swapped to our 32M dense-finetuned checkpoint (instead of
     bert-base-uncased).
  2. The `scores` column of the train dataset is OVERRIDDEN with values
     loaded from our own per-teacher score files under
     `data/teacher_scores/`. The canonical example loads `scores` from
     the lightonai/ms-marco-en-bge dataset directly; for the teacher
     ablation we score the same 16-way tuples with whichever reranker
     we're evaluating as teacher (see scripts/score_with_reranker.py)
     and feed THOSE in as the `scores` column.

Optional `--teacher-normalize minmax_per_group` rescales each group's
score vector to [0, 1] before passing it to the loss. The pylate loss
itself (pylate/losses/distillation.py) takes whatever we hand it in
`labels` and does `log_softmax(labels)` to build the KL-Div target.

Run for each ablation cell:

  python scripts/train_kd.py --teacher-scores data/teacher_scores/qwen3_reranker_8b.jsonl
  python scripts/train_kd.py --teacher-scores data/teacher_scores/qwen3_reranker_8b.jsonl --teacher-normalize minmax_per_group
  python scripts/train_kd.py --teacher-scores data/teacher_scores/bge_reranker_v2_m3.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils


def attach_teacher_scores(train_ds, teacher_scores_path: str, normalize: str):
    """Override the `scores` column of the train dataset with values from a
    per-teacher scoring jsonl. Each line: {qid, query, scores}. Group
    order in train_ds and score files must match by query_id."""
    teacher_by_qid: dict[str, list[float]] = {}
    with open(teacher_scores_path) as f:
        for line in f:
            r = json.loads(line)
            teacher_by_qid[r["qid"]] = r["scores"]

    if normalize == "minmax_per_group":
        for qid, sc in teacher_by_qid.items():
            lo, hi = min(sc), max(sc)
            if hi - lo > 1e-9:
                teacher_by_qid[qid] = [(s - lo) / (hi - lo) for s in sc]
    elif normalize != "none":
        raise ValueError(f"unknown --teacher-normalize: {normalize}")

    def _override(row):
        sc = teacher_by_qid.get(row["query_id"])
        if sc is None:
            return None
        row["scores"] = sc
        return row

    out = train_ds.map(_override)
    out = out.filter(lambda r: r is not None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-scores", required=True)
    ap.add_argument("--teacher-normalize", default="none",
                    choices=["none", "minmax_per_group"])
    args = ap.parse_args()

    train = load_dataset(path="lightonai/ms-marco-en-bge", name="train")
    queries = load_dataset(path="lightonai/ms-marco-en-bge", name="queries")
    documents = load_dataset(path="lightonai/ms-marco-en-bge", name="documents")

    train = attach_teacher_scores(
        train, args.teacher_scores, args.teacher_normalize,
    )
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    model_name = "mixedbread-ai/mxbai-edge-colbert-v0-32m-dense-ft"
    batch_size = 128
    run_name = (
        f"kd-{args.teacher_scores.split('/')[-1].rstrip('.jsonl')}"
        f"-{args.teacher_normalize}"
    )

    model = models.ColBERT(model_name_or_path=model_name)
    model = torch.compile(model)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=f"output/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        bf16=True,
        run_name=run_name,
        learning_rate=1e-3,
    )

    train_loss = losses.Distillation(model=model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()


if __name__ == "__main__":
    main()
