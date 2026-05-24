"""Modal probe: measure post-hoc projection-dim truncation effects on the
released mxbai-edge-colbert-v0-17m. Substrate is the tech report Table 8
projection-dim ablation, which shows a cliff between dim 48 and dim 32 when
the projection head is RE-TRAINED at each dim:

    dim 96 -> NDCG@10 0.5991
    dim 64 -> 0.5985
    dim 48 -> 0.5967   (shipped 17M dim)
    dim 32 -> 0.5772   (CLIFF starts here)
    dim 24 -> 0.5423
    dim 16 -> 0.5126

The agent's job is NOT to re-train but to make an edge-deployment call on
the SHIPPED 48-dim model: what does post-hoc truncation actually cost, how
does the SVD of the projection matrix bound it, and how should they
compare against the paper's trained-dim cliff.

Loading without pylate/sentence_transformers (avoids torch/torchvision
version drift in their dependency graph). The model structure on HF is:

    backbone (ModernBERT-17M, 256 hidden)
      -> 1_Dense: Linear(256 -> 512, no bias, identity activation)
      -> 2_Dense: Linear(512 -> 48,  no bias, identity activation)

The two dense layers compose to an effective 256 -> 48 linear projection
(both identity activation), so the "effective W" = W2 @ W1 is 48 x 256.

Usage:
  modal run modal_probe.py --action probe --dataset scifact
"""
from __future__ import annotations

import modal

APP_NAME = "mxbai-projection-dim"
VOLUME_NAME = "mxbai-projection-dim"

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
        "safetensors",
        "huggingface_hub",
        "numpy",
        "scipy",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu="H100:1", cpu=8.0, timeout=60 * 60, volumes={"/data": volume})
def probe(run_id: str, dataset: str, max_doc_tokens: int, max_query_tokens: int) -> dict:
    import json
    import math
    import os
    import subprocess
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoTokenizer, AutoModel

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

    # ------------------------------------------------------------------
    # Download model + load backbone + projection layers
    # ------------------------------------------------------------------
    model_id = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    print(f"[{time.strftime('%H:%M:%S')}] downloading {model_id} ...")
    snap = snapshot_download(model_id)
    print(f"  snapshot at {snap}")

    tok = AutoTokenizer.from_pretrained(snap)
    backbone = AutoModel.from_pretrained(snap, torch_dtype=torch.float32).cuda().eval()
    hidden = backbone.config.hidden_size

    w1 = load_safetensors(os.path.join(snap, "1_Dense", "model.safetensors"))
    w2 = load_safetensors(os.path.join(snap, "2_Dense", "model.safetensors"))
    # Each contains a single tensor "linear.weight"
    W1 = next(iter(w1.values())).cuda().float()   # (512, 256)
    W2 = next(iter(w2.values())).cuda().float()   # (48, 512)
    print(f"  W1 {tuple(W1.shape)}  W2 {tuple(W2.shape)}  hidden={hidden}")

    # Effective single-matrix projection W_eff = W2 @ W1 ∈ (48, 256)
    W_eff = W2 @ W1  # (48, 256)
    proj_out_dim = int(W_eff.shape[0])

    W_eff_np = W_eff.detach().cpu().float().numpy()
    U, S, Vt = np.linalg.svd(W_eff_np, full_matrices=False)
    cum_energy = np.cumsum(S**2) / np.sum(S**2)
    with open(os.path.join(out_dir, "projection_svd.json"), "w") as f:
        json.dump({
            "note": ("W_eff = W2 @ W1, the composition of the two identity-"
                     "activation Dense layers. (48 x 256)"),
            "W1_shape": list(W1.shape),
            "W2_shape": list(W2.shape),
            "W_eff_shape": list(W_eff_np.shape),
            "backbone_hidden_dim": int(hidden),
            "projection_out_dim": proj_out_dim,
            "singular_values": S.tolist(),
            "singular_value_normalized": (S / S.max()).tolist(),
            "cumulative_energy_top_k": cum_energy.tolist(),
            "rank_99_energy": int(np.searchsorted(cum_energy, 0.99) + 1),
            "rank_95_energy": int(np.searchsorted(cum_energy, 0.95) + 1),
            "rank_90_energy": int(np.searchsorted(cum_energy, 0.90) + 1),
            "rank_50_energy": int(np.searchsorted(cum_energy, 0.50) + 1),
            "effective_rank_participation_ratio": float((S.sum())**2 / (S**2).sum()),
        }, f, indent=2)

    # ------------------------------------------------------------------
    # Load BEIR dataset and qrels
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading BeIR/{dataset} ...")
    corpus = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{dataset}-qrels", split="test")
    print(f"  corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    qrels_by_qid: dict[str, dict[str, int]] = {}
    for r in qrels:
        qid = str(r["query-id"])
        did = str(r["corpus-id"])
        qrels_by_qid.setdefault(qid, {})[did] = int(r["score"])
    valid_qids = set(qrels_by_qid.keys())
    queries = [q for q in queries if str(q["_id"]) in valid_qids]
    print(f"  filtered to {len(queries)} queries with qrels")

    # ------------------------------------------------------------------
    # Encode: ColBERT-style means we keep per-token embeddings
    # ------------------------------------------------------------------
    corpus_texts = [(d.get("title") or "") + " " + (d.get("text") or "") for d in corpus]
    corpus_ids = [str(d["_id"]) for d in corpus]
    query_texts = [q["text"] for q in queries]
    query_ids = [str(q["_id"]) for q in queries]

    # ColBERT uses prefix tokens (Q] for query, D] for doc) but for our
    # truncation comparison we hold the prefix fixed across all dims so it
    # cancels out. Using the same plain text is fine; the model was
    # trained without explicit special prefixes (uncased ModernBERT).

    def encode_batch(texts: list[str], max_len: int, is_query: bool, batch_size: int = 32) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                h = backbone(**enc).last_hidden_state  # (B, T, hidden)
                # Apply effective projection: h -> h @ W_eff.T  shape (B, T, 48)
                projected = h @ W_eff.T
            mask = enc["attention_mask"].bool()
            projected_cpu = projected.detach().cpu().float().numpy()
            mask_cpu = mask.cpu().numpy()
            for b in range(projected_cpu.shape[0]):
                valid = mask_cpu[b]
                out.append(projected_cpu[b, valid, :].copy())
            if (i // batch_size) % 20 == 0:
                kind = "query" if is_query else "doc"
                print(f"    {kind} {min(i+batch_size, len(texts))}/{len(texts)}")
        return out

    print(f"[{time.strftime('%H:%M:%S')}] encoding {len(corpus_texts)} docs ...")
    t0 = time.time()
    doc_embs = encode_batch(corpus_texts, max_doc_tokens, is_query=False)
    print(f"  doc encode {time.time()-t0:.1f}s")
    t0 = time.time()
    q_embs = encode_batch(query_texts, max_query_tokens, is_query=True)
    print(f"  query encode {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # PCA basis fit on a sample of doc-token embeddings
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] fitting PCA on doc-token sample ...")
    sample_tokens = np.concatenate(doc_embs[:2000], axis=0)  # (N, 48)
    sample_tokens = sample_tokens - sample_tokens.mean(0, keepdims=True)
    _, pca_S, pca_Vt = np.linalg.svd(sample_tokens, full_matrices=False)
    pca_cum = np.cumsum(pca_S**2) / np.sum(pca_S**2)
    print(f"  PCA energy: top-32={pca_cum[31]:.4f} top-24={pca_cum[23]:.4f} top-16={pca_cum[15]:.4f}")

    # Random orthogonal projection (deterministic seed)
    rng = np.random.default_rng(0)
    full_dim = doc_embs[0].shape[-1]
    R = rng.normal(size=(full_dim, full_dim)).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    R = Q

    def truncate(embs: list[np.ndarray], dim: int, strategy: str) -> list[np.ndarray]:
        if strategy == "naive":
            return [e[..., :dim] for e in embs]
        if strategy == "pca":
            return [(e @ pca_Vt.T)[..., :dim] for e in embs]
        if strategy == "random":
            return [(e @ R)[..., :dim] for e in embs]
        raise ValueError(strategy)

    def renorm(embs: list[np.ndarray]) -> list[np.ndarray]:
        return [e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9) for e in embs]

    def maxsim_scores(q_t: torch.Tensor, d_ts: list[torch.Tensor]) -> np.ndarray:
        scores = np.zeros(len(d_ts), dtype=np.float32)
        for i, d in enumerate(d_ts):
            sim = q_t @ d.T  # (nq, nd)
            scores[i] = float(sim.max(dim=1).values.sum().item())
        return scores

    def ndcg10(ranked_doc_ids: list[str], rels: dict[str, int]) -> float:
        dcg = 0.0
        for i, did in enumerate(ranked_doc_ids[:10]):
            rel = rels.get(did, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(i + 2)
        ideal = sorted(rels.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal[:10]):
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(i + 2)
        return dcg / idcg if idcg > 0 else 0.0

    eval_dims = [48, 40, 32, 24, 16, 8]
    strategies = ["naive", "pca", "random"]
    per_dim_results: dict[str, dict[int, dict]] = {s: {} for s in strategies}

    device = "cuda"

    for strat in strategies:
        for dim in eval_dims:
            if dim > full_dim:
                continue
            t0 = time.time()
            d_trunc = renorm(truncate(doc_embs, dim, strat))
            q_trunc = renorm(truncate(q_embs, dim, strat))
            d_ts = [torch.from_numpy(e).to(device).float() for e in d_trunc]
            ndcgs = []
            for qi, q in enumerate(q_trunc):
                q_t = torch.from_numpy(q).to(device).float()
                scores = maxsim_scores(q_t, d_ts)
                order = np.argsort(-scores)
                ranked = [corpus_ids[i] for i in order]
                rels = qrels_by_qid.get(query_ids[qi], {})
                ndcgs.append(ndcg10(ranked, rels))
            avg = float(np.mean(ndcgs))
            per_dim_results[strat][dim] = {
                "ndcg_at_10": avg,
                "n_queries": len(ndcgs),
                "elapsed_sec": time.time() - t0,
            }
            print(f"  {strat:6s} dim={dim:3d}  NDCG@10={avg:.4f}  ({time.time()-t0:.1f}s)")

    with open(os.path.join(out_dir, "truncation_ndcg.json"), "w") as f:
        json.dump({
            "dataset": f"BeIR/{dataset}",
            "model": model_id,
            "full_dim": int(full_dim),
            "n_queries": len(queries),
            "n_docs": len(corpus),
            "per_strategy": per_dim_results,
            "pca_cumulative_energy_top_k": pca_cum.tolist(),
        }, f, indent=2)

    # ------------------------------------------------------------------
    # Storage / scoring cost analysis
    # ------------------------------------------------------------------
    avg_tokens_per_doc = float(np.mean([e.shape[0] for e in doc_embs]))
    avg_tokens_per_query = float(np.mean([e.shape[0] for e in q_embs]))
    cost = {}
    for dim in eval_dims:
        cost[dim] = {
            "bytes_per_doc_fp16": int(round(avg_tokens_per_doc * dim * 2)),
            "bytes_per_doc_int8": int(round(avg_tokens_per_doc * dim * 1)),
            "bytes_per_query_fp16": int(round(avg_tokens_per_query * dim * 2)),
            "maxsim_ops_per_query_doc_pair": int(round(
                avg_tokens_per_query * avg_tokens_per_doc * dim
            )),
        }
    with open(os.path.join(out_dir, "cost_analysis.json"), "w") as f:
        json.dump({
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "avg_tokens_per_query": avg_tokens_per_query,
            "per_dim_cost": cost,
            "scaling_note": (
                "MaxSim per (query, doc) pair = nq * nd * dim FMA ops. "
                "Storage scales linearly with dim. 48 -> 24 halves both."
            ),
        }, f, indent=2)

    volume.commit()
    return {"run_id": run_id, "out_dir": out_dir}


@app.local_entrypoint()
def main(
    action: str = "probe",
    dataset: str = "scifact",
    max_doc_tokens: int = 220,
    max_query_tokens: int = 32,
    run_id: str = "",
) -> None:
    import time
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    if action == "probe":
        out = probe.remote(run_id, dataset, max_doc_tokens, max_query_tokens)
        print(f"[probe] {out}")
    else:
        raise SystemExit(f"unknown action: {action}")
