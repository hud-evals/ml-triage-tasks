"""
mHC paper reproduction in the nmoe physics harness.

Goal
  Reproduce the *mechanism* and *training dynamics* reported in:
    mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880)

Scope (paper-faithful, micro scale)
  - Baseline: standard residual connections
  - HC: Hyper-Connections with n-stream residual + unconstrained H_res
  - mHC: Manifold-Constrained HC with Sinkhorn-projected H_res (doubly stochastic)

We do *not* attempt to reproduce infrastructure claims (TileLang fusion, DualPipe overlap)
inside this harness. This module is about: (i) correct math/parameterization, (ii) training
stability curves, and (iii) propagation diagnostics (Amax gain magnitude).

Execution model
  - Uses the same nmoe synthetic data packer (nmoe.research.physics.data.pack) and runs entirely
    from scripts (terminal-first, artifact-driven).
  - Intended for moonlet/micro scale runs only.

Run
  python -m nmoe.research.physics.mhc_repro --output /tmp/mhc_repro --steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNormF32(nn.Module):
    """
    RMSNorm with float32 weights and float32 compute.

    Used for HC/mHC coefficient computation where the paper writes RMSNorm explicitly.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps)


def _amax_gain_magnitude(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Amax Gain Magnitude (mHC paper):
      forward: max absolute row-sum
      backward: max absolute col-sum

    Args:
      M: [..., n, n]
    Returns:
      (fwd_gain, bwd_gain): [...] tensors
    """
    fwd = M.sum(dim=-1).abs().amax(dim=-1)
    bwd = M.sum(dim=-2).abs().amax(dim=-1)
    return fwd, bwd


def _sinkhorn_knopp(log_M: torch.Tensor, *, iters: int) -> torch.Tensor:
    """
    Entropic projection to the Birkhoff polytope.

    Paper: H_res = Sinkhorn-Knopp(tilde_H_res), with M^(0)=exp(tilde_H_res)
    followed by alternating row/col normalization.

    Args:
      log_M: [..., n, n] (pre-exp)
      iters: iterations (paper uses 20)
    Returns:
      H: [..., n, n] approximately doubly-stochastic.
    """
    # Stabilize exp for numerical robustness (n is tiny, but log_M can drift during training).
    log_M = log_M - log_M.amax(dim=(-2, -1), keepdim=True)
    M = torch.exp(log_M)
    for _ in range(int(iters)):
        M = M / (M.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        M = M / (M.sum(dim=-2, keepdim=True).clamp_min(1e-12))
    return M


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal attention mask handled by SDPA is tricky across versions; do explicit mask.
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


class HCMaps(nn.Module):
    """
    HC coefficient computation (Eq. 5 in the mHC paper preliminary).

    Uses per-stream RMSNorm and per-token dynamic mappings:
      H_pre  = α_pre * tanh(θ_pre x~^T) + b_pre      shape [B,T,n]
      H_post = α_post* tanh(θ_post x~^T) + b_post    shape [B,T,n]
      H_res  = α_res * tanh(θ_res x~^T) + b_res      shape [B,T,n,n]
    """

    def __init__(self, *, n: int, dim: int, eps: float, alpha_init: float = 0.01):
        super().__init__()
        self.n = int(n)
        self.dim = int(dim)
        self.eps = float(eps)

        self.rms = _RMSNormF32(dim, eps)

        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.theta_pre = nn.Parameter(torch.empty((dim,)))
        self.theta_post = nn.Parameter(torch.empty((dim,)))
        self.theta_res = nn.Parameter(torch.empty((n, dim)))

        self.b_pre = nn.Parameter(torch.full((n,), 1.0 / n))
        self.b_post = nn.Parameter(torch.ones((n,)))
        self.b_res = nn.Parameter(torch.eye(n))

        nn.init.normal_(self.theta_pre, mean=0.0, std=0.02)
        nn.init.normal_(self.theta_post, mean=0.0, std=0.02)
        nn.init.normal_(self.theta_res, mean=0.0, std=0.02)

    def forward(self, x_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_stream: [B,T,n,C]
        x_tilde = self.rms(x_stream)  # [B,T,n,C] float32

        # H_pre/H_post: [B,T,n]
        hpre_dyn = torch.einsum("c,btnc->btn", self.theta_pre, x_tilde)
        hpost_dyn = torch.einsum("c,btnc->btn", self.theta_post, x_tilde)
        H_pre = self.alpha_pre * torch.tanh(hpre_dyn) + self.b_pre
        H_post = self.alpha_post * torch.tanh(hpost_dyn) + self.b_post

        # H_res: [B,T,n,n]
        hres_dyn = torch.einsum("ic,btnc->btin", self.theta_res, x_tilde)
        H_res = self.alpha_res * torch.tanh(hres_dyn) + self.b_res

        return H_pre, H_post, H_res


class MHCMaps(nn.Module):
    """
    mHC coefficient computation (Eq. 7-10, as fused in Eq. 5-10).

    Uses flattened residual vector and projects:
      H_pre  = sigmoid(tilde_H_pre)
      H_post = 2*sigmoid(tilde_H_post)
      H_res  = Sinkhorn-Knopp(tilde_H_res)  (doubly stochastic)
    """

    def __init__(
        self,
        *,
        n: int,
        dim: int,
        eps: float,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.n = int(n)
        self.dim = int(dim)
        self.eps = float(eps)
        self.sinkhorn_iters = int(sinkhorn_iters)

        nC = n * dim
        out_dim = n * n + 2 * n

        # Paper: vec(x_l) is RMSNormed (learnable weight absorbed in phi in fused kernels).
        self.rms = _RMSNormF32(nC, eps)

        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.phi = nn.Parameter(torch.empty((nC, out_dim)))
        self.b = nn.Parameter(torch.empty((out_dim,)))

        # Initialize to the fixed mappings used in HC ablations:
        #   H_pre: uniform 1/n
        #   H_post: ones
        #   H_res: identity (as a near-permutation under Sinkhorn)
        b_pre = math.log((1.0 / n) / (1.0 - 1.0 / n))  # logit(1/n)
        b_post = 0.0  # 2*sigmoid(0)=1
        b_res = np.full((n, n), math.log(1e-6), dtype=np.float32)
        np.fill_diagonal(b_res, 0.0)
        b_full = np.concatenate(
            [
                np.full((n,), b_pre, dtype=np.float32),
                np.full((n,), b_post, dtype=np.float32),
                b_res.reshape(-1),
            ],
            axis=0,
        )
        self.b.data.copy_(torch.from_numpy(b_full))
        nn.init.normal_(self.phi, mean=0.0, std=0.02)

    def forward(self, x_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_stream: [B,T,n,C]
        B, T, n, C = x_stream.shape
        if n != self.n or C != self.dim:
            raise ValueError(f"Unexpected shape {tuple(x_stream.shape)} for n={self.n}, dim={self.dim}")

        x_vec = x_stream.reshape(B, T, n * C)  # [B,T,nC]
        x_prime = self.rms(x_vec)  # [B,T,nC] float32

        tt = x_prime @ self.phi  # [B,T,n^2+2n]

        # Split and apply per-map scalars (Eq. 7 in the paper)
        off_pre = 0
        off_post = off_pre + n
        off_res = off_post + n

        pre_raw = (self.alpha_pre * tt[..., off_pre:off_post]) + self.b[off_pre:off_post]
        post_raw = (self.alpha_post * tt[..., off_post:off_res]) + self.b[off_post:off_res]
        res_raw = (self.alpha_res * tt[..., off_res:]).reshape(B, T, n, n) + self.b[off_res:].reshape(n, n)

        H_pre = pre_raw.sigmoid()
        H_post = 2.0 * post_raw.sigmoid()
        H_res = _sinkhorn_knopp(res_raw, iters=self.sinkhorn_iters)
        return H_pre, H_post, H_res


class HyperBlock(nn.Module):
    """
    A single residual layer wrapped with HC/mHC pre/post/res maps.

    The residual state is n-stream: x_stream [B,T,n,C].
    """

    def __init__(
        self,
        *,
        n: int,
        dim: int,
        eps: float,
        maps: nn.Module,  # HCMaps or MHCMaps
        fn: nn.Module,  # residual function F: [B,T,C] -> [B,T,C]
    ):
        super().__init__()
        self.n = int(n)
        self.dim = int(dim)
        self.eps = float(eps)
        self.maps = maps
        self.fn = fn

    def forward(self, x_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_stream: [B,T,n,C]
        H_pre, H_post, H_res = self.maps(x_stream)

        x_in = torch.einsum("btn,btnc->btc", H_pre, x_stream)
        y = self.fn(x_in)  # [B,T,C]
        y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)  # [B,T,n,C]
        x_res = torch.einsum("btij,btjc->btic", H_res, x_stream)
        return x_res + y_stream, H_res


class _PreNormAttnFn(nn.Module):
    def __init__(self, dim: int, n_heads: int, eps: float):
        super().__init__()
        self.norm = _RMSNormF32(dim, eps)
        self.attn = CausalSelfAttention(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x))


class _PreNormFFNFn(nn.Module):
    def __init__(self, dim: int, inter_dim: int, eps: float):
        super().__init__()
        self.norm = _RMSNormF32(dim, eps)
        self.ffn = MLP(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x))


class BaselineTransformer(nn.Module):
    """
    Baseline: standard pre-norm Transformer with single-stream residuals.

    For parity with HC/mHC logging, forward returns (logits, H_res_seq=[]).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        inter_dim: int,
        n_layers: int,
        n_heads: int,
        eps: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.eps = float(eps)

        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.unembed = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.layers_attn = nn.ModuleList([_PreNormAttnFn(self.dim, int(n_heads), self.eps) for _ in range(self.n_layers)])
        self.layers_ffn = nn.ModuleList([_PreNormFFNFn(self.dim, int(inter_dim), self.eps) for _ in range(self.n_layers)])

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(input_ids)
        for l in range(self.n_layers):
            x = x + self.layers_attn[l](x)
            x = x + self.layers_ffn[l](x)
        return self.unembed(x), []


class HCTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        inter_dim: int,
        n_layers: int,
        n_heads: int,
        n_streams: int,
        eps: float,
        maps_kind: str,  # "hc" | "mhc"
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.n_streams = int(n_streams)
        self.eps = float(eps)

        self.embed = nn.Embedding(self.vocab_size, dim)
        self.unembed = nn.Linear(dim, self.vocab_size, bias=False)

        def make_maps() -> nn.Module:
            if maps_kind == "hc":
                return HCMaps(n=n_streams, dim=dim, eps=eps)
            if maps_kind == "mhc":
                return MHCMaps(n=n_streams, dim=dim, eps=eps, sinkhorn_iters=sinkhorn_iters)
            raise ValueError(f"Unknown maps_kind={maps_kind}")

        self.layers_attn = nn.ModuleList(
            [
                HyperBlock(
                    n=n_streams,
                    dim=dim,
                    eps=eps,
                    maps=make_maps(),
                    fn=_PreNormAttnFn(dim, n_heads, eps),
                )
                for _ in range(n_layers)
            ]
        )
        self.layers_ffn = nn.ModuleList(
            [
                HyperBlock(
                    n=n_streams,
                    dim=dim,
                    eps=eps,
                    maps=make_maps(),
                    fn=_PreNormFFNFn(dim, inter_dim, eps),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # input_ids: [B,T]
        x0 = self.embed(input_ids)  # [B,T,C]
        x_stream = x0.unsqueeze(2).expand(-1, -1, self.n_streams, -1).contiguous()

        Hres_seq: list[torch.Tensor] = []
        for l in range(self.n_layers):
            x_stream, Hres_a = self.layers_attn[l](x_stream)
            Hres_seq.append(Hres_a)
            x_stream, Hres_f = self.layers_ffn[l](x_stream)
            Hres_seq.append(Hres_f)

        x = x_stream.mean(dim=2)  # [B,T,C] readout of streams
        logits = self.unembed(x)  # [B,T,V]
        return logits, Hres_seq


@dataclass
class ReproConfig:
    output: Path
    steps: int = 2000
    seed: int = 42
    # Data
    dataset: str = "mhc-physics"
    tasks: tuple[str, ...] = ("depo:1.0:n_entities=50,max_hops=4", "mano:1.0:depth=3", "brevo:1.0:n_nodes=24,max_parents=2")
    n_train: int = 50000
    n_valid: int = 2000
    seq_len: int = 256
    # Model (micro)
    vocab_size: int = 10000
    dim: int = 256
    inter_dim: int = 512
    n_layers: int = 6
    n_heads: int = 4
    n_streams: int = 4
    eps: float = 1e-6
    # Optim
    lr: float = 2e-3
    weight_decay: float = 0.1
    batch_size: int = 16
    grad_accum: int = 1
    log_every: int = 50
    eval_every: int = 200
    # mHC
    sinkhorn_iters: int = 20


def _run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _cfg_to_json(cfg: ReproConfig) -> dict:
    d = asdict(cfg)
    d["output"] = str(d["output"])
    return d


def _maybe_pack(cfg: ReproConfig) -> None:
    data_dir = cfg.output / "data"
    if (data_dir / "train").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            sys.executable,
            "-m",
            "nmoe.research.physics.data.pack",
            "--output",
            str(data_dir),
            "--dataset",
            cfg.dataset,
            "--tasks",
            *cfg.tasks,
            "--n-train",
            str(cfg.n_train),
            "--n-valid",
            str(cfg.n_valid),
            "--seq-len",
            str(cfg.seq_len),
            "--seed",
            str(cfg.seed),
        ]
    )


def _load_one_shard(split_dir: Path) -> np.ndarray:
    shards = sorted(split_dir.glob("*.npy"))
    if not shards:
        raise FileNotFoundError(f"No .npy shards in {split_dir}")
    return np.load(shards[0])


def _iter_minibatches(*, tokens_1d: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator):
    # Each packed doc is fixed length (seq_len+1); reshape accordingly.
    doc_len = int(seq_len) + 1
    n_docs = int(tokens_1d.size // doc_len)
    if n_docs <= 0:
        raise ValueError(f"Shard too small: tokens={tokens_1d.size}, doc_len={doc_len}")
    docs = tokens_1d[: n_docs * doc_len].reshape(n_docs, doc_len)
    while True:
        idx = rng.integers(0, n_docs, size=int(batch_size), endpoint=False)
        batch = docs[idx]  # [B, T+1]
        x = torch.from_numpy(batch[:, :-1].astype(np.int64))
        y = torch.from_numpy(batch[:, 1:].astype(np.int64))
        yield x, y


def _train_one(
    *,
    name: str,
    model: nn.Module,
    train_tokens: np.ndarray,
    valid_tokens: np.ndarray,
    cfg: ReproConfig,
    device: torch.device,
) -> dict:
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)

    rng = np.random.default_rng(cfg.seed)
    train_iter = _iter_minibatches(tokens_1d=train_tokens, seq_len=cfg.seq_len, batch_size=cfg.batch_size, rng=rng)
    valid_iter = _iter_minibatches(tokens_1d=valid_tokens, seq_len=cfg.seq_len, batch_size=cfg.batch_size, rng=rng)

    out_dir = cfg.output / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_cfg_to_json(cfg), indent=2))

    log_path = out_dir / "train.jsonl"
    f = log_path.open("w", encoding="utf-8")

    def log(rec: dict) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()

    for step in range(1, int(cfg.steps) + 1):
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits, Hres_seq = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

        grad_norm = torch.sqrt(
            sum((p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None), torch.tensor(0.0, device=device))
        ).item()

        opt.step()

        if step % int(cfg.log_every) == 0 or step == 1:
            with torch.no_grad():
                # Paper-faithful diagnostics: compute per-layer H_res means and composite gain.
                per_fwd: list[float] = []
                per_bwd: list[float] = []
                comp_fwd: list[float] = []
                comp_bwd: list[float] = []

                if Hres_seq:
                    H_means = [H.detach().float().mean(dim=(0, 1)) for H in Hres_seq]  # each [n,n]
                    comp = torch.eye(H_means[0].size(0), device=device, dtype=torch.float32)
                    for Hm in H_means:
                        fwd, bwd = _amax_gain_magnitude(Hm)
                        per_fwd.append(float(fwd.item()))
                        per_bwd.append(float(bwd.item()))
                        comp = Hm @ comp
                        cfwd, cbwd = _amax_gain_magnitude(comp)
                        comp_fwd.append(float(cfwd.item()))
                        comp_bwd.append(float(cbwd.item()))
                mem_alloc = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else float("nan")
            log(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm),
                    "mhc_per_fwd": per_fwd,
                    "mhc_per_bwd": per_bwd,
                    "mhc_comp_fwd": comp_fwd,
                    "mhc_comp_bwd": comp_bwd,
                    "cuda_max_mem_mib": mem_alloc,
                }
            )

        if step % int(cfg.eval_every) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = next(valid_iter)
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                v_logits, _ = model(vx)
                v_loss = float(F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), vy.reshape(-1)).item())
            log({"step": step, "valid_loss": v_loss})
            model.train()

    f.close()
    return {"run_dir": str(out_dir), "train_log": str(log_path)}


def main() -> int:
    p = argparse.ArgumentParser(description="mHC reproduction (physics harness)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = ReproConfig(output=args.output, steps=args.steps, seed=args.seed)
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "repro_config.json").write_text(json.dumps(_cfg_to_json(cfg), indent=2))

    _maybe_pack(cfg)
    train_tokens = _load_one_shard(cfg.output / "data" / "train")
    valid_tokens = _load_one_shard(cfg.output / "data" / "valid")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    runs: dict[str, dict] = {}

    runs["baseline"] = _train_one(
        name="baseline",
        model=BaselineTransformer(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            inter_dim=cfg.inter_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            eps=cfg.eps,
        ),
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        cfg=cfg,
        device=device,
    )

    runs["hc"] = _train_one(
        name="hc",
        model=HCTransformer(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            inter_dim=cfg.inter_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_streams=cfg.n_streams,
            eps=cfg.eps,
            maps_kind="hc",
        ),
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        cfg=cfg,
        device=device,
    )

    runs["mhc"] = _train_one(
        name="mhc",
        model=HCTransformer(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            inter_dim=cfg.inter_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_streams=cfg.n_streams,
            eps=cfg.eps,
            maps_kind="mhc",
            sinkhorn_iters=cfg.sinkhorn_iters,
        ),
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        cfg=cfg,
        device=device,
    )

    (cfg.output / "runs.json").write_text(json.dumps(runs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
