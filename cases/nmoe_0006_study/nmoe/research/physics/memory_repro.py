"""
Memory mechanisms repro: Engram vs PLE+Ngrammer (PhysicsLM4-style, synthetic).

Goal
  Isolate the "memory" axis and compare:
    1) Baseline transformer (no explicit memory primitive)
    2) Engram-like hashed bigram memory with context-aware gating
    3) PLE+Ngrammer-like per-layer hashed bigram embeddings (no hidden-state gate)

Task
  `ngram`: Markov order-2 language modeling.
    prompt  = [BOS, s0, s1, ANSWER_START]
    answer  = s2..s_{n_steps+1}
  We train standard next-token prediction and report answer-region accuracy.

Run
  python -m nmoe.research.physics.memory_repro --output /tmp/memory_repro --steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmoe.research.physics.data.generators import (
    ANSWER_START,
    EOS,
    NGRAM_SYM_MAX,
    NGRAM_SYM_MIN,
    SyntheticMix,
    ngram_mixed,
)


class _RMSNormF32(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.out = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
        mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        p = att.softmax(dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        y = p @ v  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, inter_dim, bias=False)
        self.fc2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


def _bigram_ids(
    input_ids: torch.Tensor,  # [B,T]
    *,
    table_size: int,
    multiplier: int,
) -> torch.Tensor:
    """
    Hash (prev, cur) token IDs to a table row index.

    We intentionally only hash within the NGRAM symbol range; everything else maps to 0.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"expected input_ids [B,T], got {tuple(input_ids.shape)}")
    B, T = input_ids.shape

    prev = torch.empty_like(input_ids)
    prev[:, 0] = 0
    prev[:, 1:] = input_ids[:, :-1]
    cur = input_ids

    is_prev = (prev >= NGRAM_SYM_MIN) & (prev <= NGRAM_SYM_MAX)
    is_cur = (cur >= NGRAM_SYM_MIN) & (cur <= NGRAM_SYM_MAX)
    ok = is_prev & is_cur

    a = (prev - NGRAM_SYM_MIN).clamp(min=0).to(torch.int64)
    b = (cur - NGRAM_SYM_MIN).clamp(min=0).to(torch.int64)
    h = (a * int(multiplier) + b) % int(table_size)

    out = torch.zeros((B, T), device=input_ids.device, dtype=torch.int64)
    out[ok] = h[ok]
    return out


class MemoryModule(nn.Module):
    def forward(  # pragma: no cover
        self,
        *,
        x: torch.Tensor,
        bigram_ids: torch.Tensor,
        layer_id: int,
        collect_stats: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def layer_stats(self) -> dict[int, dict[str, float]]:
        return {}


class EngramMemory(MemoryModule):
    """
    Engram-like hashed bigram memory:
      mem  = E[hash(prev,cur)]
      gate = sigmoid(<q(x), k(mem)> / sqrt(dim))
      delta = gate * v(mem)
    """

    def __init__(
        self,
        *,
        n_layers: int,
        dim: int,
        eps: float,
        table_size: int = 4096,
        mem_dim: int = 128,
        multiplier: int = 1000003,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.dim = int(dim)
        self.table_size = int(table_size)
        self.mem_dim = int(mem_dim)
        self.multiplier = int(multiplier)

        self.embed = nn.Embedding(self.table_size, self.mem_dim)
        self.k_proj = nn.Linear(self.mem_dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.mem_dim, self.dim, bias=False)
        self.q_proj = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.n_layers)])
        self.rms = _RMSNormF32(self.dim, eps)

        self._last_gate_mean: dict[int, torch.Tensor] = {}

    def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
        layer_id = int(layer_id)
        mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
        k = self.k_proj(mem)
        v = self.v_proj(mem)
        q = self.q_proj[layer_id](self.rms(x))

        gate_logits = (q * k).sum(dim=-1) / math.sqrt(self.dim)
        gate = gate_logits.sigmoid()
        gate = gate * (bigram_ids != 0).to(dtype=gate.dtype)

        if collect_stats:
            self._last_gate_mean[layer_id] = gate.detach().float().mean()
        return gate.unsqueeze(-1) * v

    def layer_stats(self) -> dict[int, dict[str, float]]:
        return {int(k): {"gate_mean": float(v.item())} for k, v in self._last_gate_mean.items()}


class PleNgrammerMemory(MemoryModule):
    """
    PLE+Ngrammer-like memory:
      mem = E[hash(prev,cur)]
      delta_l = proj_l(mem)
    No hidden-state-dependent gating; "addressability" is purely in the hash.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        dim: int,
        table_size: int = 4096,
        mem_dim: int = 128,
        multiplier: int = 1000003,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.dim = int(dim)
        self.table_size = int(table_size)
        self.mem_dim = int(mem_dim)
        self.multiplier = int(multiplier)

        self.embed = nn.Embedding(self.table_size, self.mem_dim)
        self.proj = nn.ModuleList([nn.Linear(self.mem_dim, self.dim, bias=False) for _ in range(self.n_layers)])
        self._last_delta_rms: dict[int, torch.Tensor] = {}

    def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
        layer_id = int(layer_id)
        mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
        delta = self.proj[layer_id](mem)
        delta = delta * (bigram_ids != 0).to(dtype=delta.dtype).unsqueeze(-1)

        # Diagnostic only: relative magnitude of the memory path.
        if collect_stats:
            self._last_delta_rms[layer_id] = delta.detach().float().pow(2).mean().sqrt()
        return delta

    def layer_stats(self) -> dict[int, dict[str, float]]:
        return {int(k): {"delta_rms": float(v.item())} for k, v in self._last_delta_rms.items()}


class Block(nn.Module):
    def __init__(self, *, dim: int, inter_dim: int, n_heads: int, eps: float):
        super().__init__()
        self.norm1 = _RMSNormF32(dim, eps)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = _RMSNormF32(dim, eps)
        self.mlp = MLP(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MemoryTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        inter_dim: int,
        n_layers: int,
        n_heads: int,
        eps: float,
        memory: MemoryModule | None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.eps = float(eps)

        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.unembed = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.layers = nn.ModuleList([Block(dim=dim, inter_dim=inter_dim, n_heads=n_heads, eps=eps) for _ in range(n_layers)])
        self.memory = memory

    def forward(self, input_ids: torch.Tensor, *, collect_stats: bool = False) -> tuple[torch.Tensor, dict[int, dict[str, float]]]:
        x = self.embed(input_ids)

        stats: dict[int, dict[str, float]] = {}
        if self.memory is not None:
            bigram_ids = _bigram_ids(input_ids, table_size=self.memory.table_size, multiplier=self.memory.multiplier)  # type: ignore[attr-defined]
        else:
            bigram_ids = torch.empty((0,), device=input_ids.device, dtype=torch.int64)

        for l, blk in enumerate(self.layers):
            if self.memory is not None:
                x = x + self.memory(x=x, bigram_ids=bigram_ids, layer_id=l, collect_stats=collect_stats)
                if collect_stats:
                    stats.update(self.memory.layer_stats())
            x = blk(x)

        return self.unembed(x), stats


@dataclass
class ReproConfig:
    output: Path
    steps: int = 2000
    seed: int = 42
    # Data
    n_train: int = 20000
    n_valid: int = 2000
    seq_len: int = 256
    n_symbols: int = 512
    n_steps_task: int = 128
    table_seed: int = 0
    # Model
    vocab_size: int = 10240
    dim: int = 256
    inter_dim: int = 512
    n_layers: int = 6
    n_heads: int = 4
    eps: float = 1e-6
    # Memory module
    mem_table_size: int = 4096
    mem_dim: int = 128
    mem_multiplier: int = 1000003
    # Optim
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    log_every: int = 50
    eval_every: int = 200
    # Loss
    loss_mode: str = "answer_only"  # full | answer_only


def _cfg_to_json(cfg: ReproConfig) -> dict:
    d = asdict(cfg)
    d["output"] = str(d["output"])
    return d


def _pad(tokens: list[int], *, target_len: int, pad_token: int) -> list[int]:
    if len(tokens) > target_len:
        raise ValueError(f"Refusing to truncate: len={len(tokens)} > target_len={target_len}.")
    return tokens + [pad_token] * (target_len - len(tokens))


def _build_split(*, n: int, cfg: ReproConfig, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    mix = SyntheticMix(seed=seed)
    mix.add(
        "ngram",
        weight=1.0,
        n_symbols=int(cfg.n_symbols),
        n_steps=int(cfg.n_steps_task),
        table_seed=int(cfg.table_seed),
    )
    samples = mix.generate(int(n))
    doc_len = int(cfg.seq_len) + 1

    toks = torch.full((len(samples), doc_len), EOS, dtype=torch.long)
    labels = torch.zeros((len(samples), doc_len), dtype=torch.uint8)
    for i, s in enumerate(samples):
        tt = _pad(s.tokens, target_len=doc_len, pad_token=EOS)
        ll = _pad(s.labels, target_len=doc_len, pad_token=0)
        toks[i] = torch.tensor(tt, dtype=torch.long)
        labels[i] = torch.tensor(ll, dtype=torch.uint8)
    return toks, labels


def _build_mixed_split(
    *,
    n: int,
    cfg: ReproConfig,
    seed: int,
    noise_frac: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a mixed dataset with both structured and noise samples.

    Returns:
        tokens: [N, doc_len]
        labels: [N, doc_len]
        is_noise: [N] bool tensor indicating which samples are noise
    """
    import random as stdlib_random

    rng = stdlib_random.Random(int(seed))
    doc_len = int(cfg.seq_len) + 1

    toks = torch.full((int(n), doc_len), EOS, dtype=torch.long)
    labels = torch.zeros((int(n), doc_len), dtype=torch.uint8)
    is_noise = torch.zeros(int(n), dtype=torch.bool)

    for i in range(int(n)):
        sample_is_noise = rng.random() < float(noise_frac)
        s = ngram_mixed(
            rng,
            n_symbols=int(cfg.n_symbols),
            n_steps=int(cfg.n_steps_task),
            table_seed=int(cfg.table_seed),
            is_noise=sample_is_noise,
        )
        tt = _pad(s.tokens, target_len=doc_len, pad_token=EOS)
        ll = _pad(s.labels, target_len=doc_len, pad_token=0)
        toks[i] = torch.tensor(tt, dtype=torch.long)
        labels[i] = torch.tensor(ll, dtype=torch.uint8)
        is_noise[i] = sample_is_noise

    return toks, labels, is_noise


def _answer_mask_from_input(x_in: torch.Tensor, *, eos_token_id: int) -> torch.Tensor:
    """
    Build an answer-region supervision mask from the ANSWER_START token.

    x_in: [B,T] aligned with next-token targets y: [B,T] (i.e., targets are x_in shifted by 1 in the original sequence)
    """
    started = (x_in == int(ANSWER_START)).cumsum(dim=1) > 0
    return started & (x_in != int(eos_token_id))


def _loss_and_acc(
    logits: torch.Tensor,  # [B,T,V]
    targets: torch.Tensor,  # [B,T]
    *,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    ignore_index = -100
    masked = targets.clone()
    masked[~mask] = ignore_index
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked.reshape(-1), ignore_index=ignore_index)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = (pred == targets) & mask
        acc = float(correct.sum().item() / max(1, mask.sum().item()))
    return loss, acc


def _iter_minibatches(*, tokens: torch.Tensor, batch_size: int, rng: np.random.Generator):
    n = int(tokens.size(0))
    while True:
        idx = rng.integers(0, n, size=int(batch_size), endpoint=False)
        yield tokens[idx]


def _eval_split(
    model: MemoryTransformer,
    tokens: torch.Tensor,
    is_noise: torch.Tensor,
    *,
    cfg: ReproConfig,
    device: torch.device,
) -> dict:
    """
    Evaluate on a mixed split, reporting separate metrics for structured vs noise.

    Returns dict with keys: structured_acc, noise_acc, structured_gate_mean, noise_gate_mean
    """
    model.eval()
    batch_size = int(cfg.batch_size)

    structured_mask = ~is_noise
    noise_mask = is_noise

    def eval_subset(subset_mask: torch.Tensor) -> tuple[float, dict[int, float]]:
        subset_tokens = tokens[subset_mask]
        if len(subset_tokens) == 0:
            return 0.0, {}

        n_batches = (len(subset_tokens) + batch_size - 1) // batch_size
        total_correct = 0
        total_count = 0
        gate_sums: dict[int, float] = {}
        gate_counts: dict[int, int] = {}

        with torch.no_grad():
            for b in range(n_batches):
                batch = subset_tokens[b * batch_size : (b + 1) * batch_size].to(device)
                x_in = batch[:, :-1]
                y = batch[:, 1:]

                logits, mem_stats = model(x_in, collect_stats=True)

                mask = _answer_mask_from_input(x_in, eos_token_id=int(EOS)) & (y != int(EOS))
                pred = logits.argmax(dim=-1)
                correct = (pred == y) & mask
                total_correct += int(correct.sum().item())
                total_count += int(mask.sum().item())

                for layer_id, stats in mem_stats.items():
                    if "gate_mean" in stats:
                        gate_sums[layer_id] = gate_sums.get(layer_id, 0.0) + float(stats["gate_mean"])
                        gate_counts[layer_id] = gate_counts.get(layer_id, 0) + 1

        acc = float(total_correct) / max(1, total_count)
        gate_means = {k: v / max(1, gate_counts[k]) for k, v in gate_sums.items()}
        return acc, gate_means

    structured_acc, structured_gates = eval_subset(structured_mask)
    noise_acc, noise_gates = eval_subset(noise_mask)

    # Aggregate gate mean across layers (layer 0 is typically most informative)
    structured_gate_mean = float(structured_gates.get(0, 0.0))
    noise_gate_mean = float(noise_gates.get(0, 0.0))

    model.train()
    return {
        "structured_acc": structured_acc,
        "noise_acc": noise_acc,
        "structured_gate_mean": structured_gate_mean,
        "noise_gate_mean": noise_gate_mean,
    }


def _train_one(
    *,
    name: str,
    model: MemoryTransformer,
    train_tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
    cfg: ReproConfig,
    device: torch.device,
) -> dict:
    out_dir = cfg.output / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_cfg_to_json(cfg), indent=2), encoding="utf-8")

    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), betas=(0.9, 0.95), eps=1e-8)

    rng = np.random.default_rng(int(cfg.seed))
    train_iter = _iter_minibatches(tokens=train_tokens, batch_size=int(cfg.batch_size), rng=rng)

    log_path = out_dir / "train.jsonl"
    f = log_path.open("w", encoding="utf-8")

    def log(rec: dict) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()

    for step in range(1, int(cfg.steps) + 1):
        batch = next(train_iter).to(device, non_blocking=True)  # [B, doc_len]
        x_in = batch[:, :-1]
        y = batch[:, 1:]

        do_stats = (step % int(cfg.log_every) == 0) or (step == 1)

        opt.zero_grad(set_to_none=True)
        logits, mem_stats = model(x_in, collect_stats=do_stats)

        if cfg.loss_mode == "full":
            mask = y != int(EOS)
        elif cfg.loss_mode == "answer_only":
            mask = _answer_mask_from_input(x_in, eos_token_id=int(EOS)) & (y != int(EOS))
        else:
            raise ValueError(f"Unknown loss_mode={cfg.loss_mode!r}")

        loss, acc = _loss_and_acc(logits, y, mask=mask)
        loss.backward()
        opt.step()

        if do_stats:
            rec = {
                "step": int(step),
                "loss": float(loss.item()),
                "answer_acc": float(acc),
                "mem": mem_stats,
            }
            log(rec)

        if step % int(cfg.eval_every) == 0:
            model.eval()
            with torch.no_grad():
                vb = valid_tokens[: int(cfg.batch_size)].to(device, non_blocking=True)
                vx = vb[:, :-1]
                vy = vb[:, 1:]
                vlogits, _ = model(vx, collect_stats=False)
                if cfg.loss_mode == "full":
                    vmask = vy != int(EOS)
                else:
                    vmask = _answer_mask_from_input(vx, eos_token_id=int(EOS)) & (vy != int(EOS))
                vloss, vacc = _loss_and_acc(vlogits, vy, mask=vmask)
            log({"step": int(step), "valid_loss": float(vloss.item()), "valid_answer_acc": float(vacc)})
            model.train()

    f.close()
    return {"run_dir": str(out_dir), "train_log": str(log_path)}


def _train_conditionality(
    *,
    name: str,
    model: MemoryTransformer,
    train_tokens: torch.Tensor,
    train_is_noise: torch.Tensor,
    valid_tokens: torch.Tensor,
    valid_is_noise: torch.Tensor,
    cfg: ReproConfig,
    device: torch.device,
) -> dict:
    """
    Train on mixed data, evaluate on structured vs noise splits separately.

    The key metric is the gap in gate activation between structured and noise.
    """
    out_dir = cfg.output / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_cfg_to_json(cfg), indent=2), encoding="utf-8")

    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), betas=(0.9, 0.95), eps=1e-8)

    rng = np.random.default_rng(int(cfg.seed))
    train_iter = _iter_minibatches(tokens=train_tokens, batch_size=int(cfg.batch_size), rng=rng)

    log_path = out_dir / "train.jsonl"
    f = log_path.open("w", encoding="utf-8")

    def log(rec: dict) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()

    for step in range(1, int(cfg.steps) + 1):
        batch = next(train_iter).to(device, non_blocking=True)
        x_in = batch[:, :-1]
        y = batch[:, 1:]

        do_stats = (step % int(cfg.log_every) == 0) or (step == 1)

        opt.zero_grad(set_to_none=True)
        logits, mem_stats = model(x_in, collect_stats=do_stats)

        mask = _answer_mask_from_input(x_in, eos_token_id=int(EOS)) & (y != int(EOS))
        loss, acc = _loss_and_acc(logits, y, mask=mask)
        loss.backward()
        opt.step()

        if do_stats:
            rec = {
                "step": int(step),
                "loss": float(loss.item()),
                "answer_acc": float(acc),
                "mem": mem_stats,
            }
            log(rec)

        if step % int(cfg.eval_every) == 0:
            split_metrics = _eval_split(model, valid_tokens, valid_is_noise, cfg=cfg, device=device)
            log({
                "step": int(step),
                "structured_acc": split_metrics["structured_acc"],
                "noise_acc": split_metrics["noise_acc"],
                "structured_gate_mean": split_metrics["structured_gate_mean"],
                "noise_gate_mean": split_metrics["noise_gate_mean"],
                "gate_gap": split_metrics["structured_gate_mean"] - split_metrics["noise_gate_mean"],
            })

    f.close()

    # Final evaluation
    final_metrics = _eval_split(model, valid_tokens, valid_is_noise, cfg=cfg, device=device)
    return {
        "run_dir": str(out_dir),
        "train_log": str(log_path),
        "final_structured_acc": final_metrics["structured_acc"],
        "final_noise_acc": final_metrics["noise_acc"],
        "final_gate_gap": final_metrics["structured_gate_mean"] - final_metrics["noise_gate_mean"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Memory repro: Engram vs PLE+Ngrammer on synthetic n-gram task.")
    p.add_argument("--output", type=Path, required=True, help="Output directory")
    p.add_argument("--mode", type=str, default="basic", choices=["basic", "conditionality"], help="basic: pure ngram. conditionality: mixed structured+noise")
    p.add_argument("--noise-frac", type=float, default=0.5, help="Fraction of noise samples in conditionality mode")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-valid", type=int, default=2000)
    p.add_argument("--n-symbols", type=int, default=512)
    p.add_argument("--n-steps-task", type=int, default=128)
    p.add_argument("--table-seed", type=int, default=0)
    p.add_argument("--loss-mode", type=str, default="answer_only", choices=["full", "answer_only"])
    args = p.parse_args()

    cfg = ReproConfig(
        output=args.output,
        steps=int(args.steps),
        seed=int(args.seed),
        seq_len=int(args.seq_len),
        n_train=int(args.n_train),
        n_valid=int(args.n_valid),
        n_symbols=int(args.n_symbols),
        n_steps_task=int(args.n_steps_task),
        table_seed=int(args.table_seed),
        loss_mode=str(args.loss_mode),
    )
    cfg.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs: dict[str, dict] = {}

    if args.mode == "basic":
        # Original pure ngram task
        train_tokens, _ = _build_split(n=cfg.n_train, cfg=cfg, seed=int(cfg.seed))
        valid_tokens, _ = _build_split(n=cfg.n_valid, cfg=cfg, seed=int(cfg.seed) + 1_000_000)

        # Baseline
        runs["baseline"] = _train_one(
            name="baseline",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=None,
            ),
            train_tokens=train_tokens,
            valid_tokens=valid_tokens,
            cfg=cfg,
            device=device,
        )

        # Engram-like
        runs["engram"] = _train_one(
            name="engram",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=EngramMemory(
                    n_layers=cfg.n_layers,
                    dim=cfg.dim,
                    eps=cfg.eps,
                    table_size=cfg.mem_table_size,
                    mem_dim=cfg.mem_dim,
                    multiplier=cfg.mem_multiplier,
                ),
            ),
            train_tokens=train_tokens,
            valid_tokens=valid_tokens,
            cfg=cfg,
            device=device,
        )

        # PLE+Ngrammer-like
        runs["ple_ngrammer"] = _train_one(
            name="ple_ngrammer",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=PleNgrammerMemory(
                    n_layers=cfg.n_layers,
                    dim=cfg.dim,
                    table_size=cfg.mem_table_size,
                    mem_dim=cfg.mem_dim,
                    multiplier=cfg.mem_multiplier,
                ),
            ),
            train_tokens=train_tokens,
            valid_tokens=valid_tokens,
            cfg=cfg,
            device=device,
        )

    elif args.mode == "conditionality":
        # Mixed structured + noise: test whether gating learns to be conditional
        noise_frac = float(args.noise_frac)
        train_tokens, _, train_is_noise = _build_mixed_split(
            n=cfg.n_train, cfg=cfg, seed=int(cfg.seed), noise_frac=noise_frac
        )
        valid_tokens, _, valid_is_noise = _build_mixed_split(
            n=cfg.n_valid, cfg=cfg, seed=int(cfg.seed) + 1_000_000, noise_frac=noise_frac
        )

        # Baseline (no memory, no gating - just for reference)
        runs["baseline"] = _train_conditionality(
            name="baseline",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=None,
            ),
            train_tokens=train_tokens,
            train_is_noise=train_is_noise,
            valid_tokens=valid_tokens,
            valid_is_noise=valid_is_noise,
            cfg=cfg,
            device=device,
        )

        # Engram: if gating works, gate_gap should be positive (higher gate on structured)
        runs["engram"] = _train_conditionality(
            name="engram",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=EngramMemory(
                    n_layers=cfg.n_layers,
                    dim=cfg.dim,
                    eps=cfg.eps,
                    table_size=cfg.mem_table_size,
                    mem_dim=cfg.mem_dim,
                    multiplier=cfg.mem_multiplier,
                ),
            ),
            train_tokens=train_tokens,
            train_is_noise=train_is_noise,
            valid_tokens=valid_tokens,
            valid_is_noise=valid_is_noise,
            cfg=cfg,
            device=device,
        )

        # PLE+Ngrammer: no gating, so gate_gap = 0 always
        # The key question: does the always-on memory hurt noise performance?
        runs["ple_ngrammer"] = _train_conditionality(
            name="ple_ngrammer",
            model=MemoryTransformer(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                inter_dim=cfg.inter_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                eps=cfg.eps,
                memory=PleNgrammerMemory(
                    n_layers=cfg.n_layers,
                    dim=cfg.dim,
                    table_size=cfg.mem_table_size,
                    mem_dim=cfg.mem_dim,
                    multiplier=cfg.mem_multiplier,
                ),
            ),
            train_tokens=train_tokens,
            train_is_noise=train_is_noise,
            valid_tokens=valid_tokens,
            valid_is_noise=valid_is_noise,
            cfg=cfg,
            device=device,
        )

    (cfg.output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(runs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
