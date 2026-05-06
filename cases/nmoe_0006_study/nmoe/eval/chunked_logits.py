from __future__ import annotations

import torch


@torch.no_grad()
def forward_hidden(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Forward model up to final norm (no lm_head), returning hidden [B,T,D]."""
    if not hasattr(model, "embedding") or not hasattr(model, "blocks") or not hasattr(model, "norm"):
        raise TypeError("model must be an nmoe.model.Transformer-like module")
    if not hasattr(model, "rope"):
        raise TypeError("model must have rope buffers (RotaryEmbedding)")

    embed_gain = float(getattr(model, "fp4_embed_gain", getattr(model, "mup_scale_factor", 1.0)))
    x = model.embedding(tokens) * embed_gain
    seqlen = int(tokens.size(1))
    cos = model.rope.cos[:seqlen].to(tokens.device)
    sin = model.rope.sin[:seqlen].to(tokens.device)
    for block in model.blocks:
        x = block(x, cos, sin)
    x = model.norm(x)
    return x


@torch.no_grad()
def logsumexp_vocab_chunked(
    h: torch.Tensor,  # [N,D], float32
    lm_head_weight: torch.Tensor,  # [V,D], bf16/fp16/fp32
    *,
    logits_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute logsumexp over vocab for each row of h without materializing (N,V)."""
    if h.dim() != 2 or lm_head_weight.dim() != 2:
        raise ValueError("expected h=[N,D], lm_head_weight=[V,D]")
    if h.size(1) != lm_head_weight.size(1):
        raise ValueError("hidden dim mismatch with lm_head weight")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    V = int(lm_head_weight.size(0))
    out = None
    for start in range(0, V, chunk_size):
        end = min(V, start + chunk_size)
        w = lm_head_weight[start:end].to(dtype=torch.float32)  # [C,D]
        logits = (h @ w.t()) * logits_scale  # [N,C] fp32
        lse = torch.logsumexp(logits, dim=-1)  # [N]
        out = lse if out is None else torch.logaddexp(out, lse)
    if out is None:
        raise RuntimeError("empty lm_head_weight")
    return out


@torch.no_grad()
def target_logits(
    h: torch.Tensor,  # [N,D], float32
    lm_head_weight: torch.Tensor,  # [V,D], bf16/fp16/fp32
    target_ids: torch.Tensor,  # [N], int64
    *,
    logits_scale: float,
) -> torch.Tensor:
    w = lm_head_weight.index_select(0, target_ids).to(dtype=torch.float32)  # [N,D]
    return (h * w).sum(dim=-1) * logits_scale  # [N]


@torch.no_grad()
def argmax_vocab_chunked(
    h: torch.Tensor,  # [N,D], float32
    lm_head_weight: torch.Tensor,  # [V,D], bf16/fp16/fp32
    *,
    logits_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute argmax over vocab for each row of h without materializing (N,V)."""
    if h.dim() != 2 or lm_head_weight.dim() != 2:
        raise ValueError("expected h=[N,D], lm_head_weight=[V,D]")
    if h.size(1) != lm_head_weight.size(1):
        raise ValueError("hidden dim mismatch with lm_head weight")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    V = int(lm_head_weight.size(0))
    best_val = None
    best_idx = None
    for start in range(0, V, chunk_size):
        end = min(V, start + chunk_size)
        w = lm_head_weight[start:end].to(dtype=torch.float32)  # [C,D]
        logits = (h @ w.t()) * logits_scale  # [N,C] fp32
        cur_val, cur_idx = torch.max(logits, dim=-1)  # [N], [N]
        cur_idx = cur_idx.to(dtype=torch.long) + int(start)
        if best_val is None:
            best_val = cur_val
            best_idx = cur_idx
        else:
            better = cur_val > best_val
            best_val = torch.where(better, cur_val, best_val)
            best_idx = torch.where(better, cur_idx, best_idx)
    if best_idx is None:
        raise RuntimeError("empty lm_head_weight")
    return best_idx
