"""Standard Scaled Dot-Product Attention with RoPE.

Simple attention matching the June 2024 nanogpt baseline architecture.
Used to isolate whether MLA causes training instability at higher LRs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.config import Config


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x. x shape: [B, S, H, D]"""
    # cos/sin: [S, D//2]
    cos = cos[:x.size(1), :].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]
    sin = sin[:x.size(1), :].unsqueeze(0).unsqueeze(2)
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


class SDPA(nn.Module):
    """Standard multi-head attention with RoPE, using F.scaled_dot_product_attention.

    Matches June 2024 nanogpt architecture:
    - Standard Q/K/V projections (no low-rank compression)
    - RoPE on full head dimension
    - Causal attention via SDPA

    Note: This module computes its own RoPE embeddings since it needs full head_dim,
    unlike MLA which uses partial RoPE (qk_rope_head_dim).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        # Standard Q/K/V projections
        self.wq = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)

        self.softmax_scale = self.head_dim ** -0.5

        # Attention scaling factor from June 2024 nanogpt: 1/sqrt(2*n_layer)
        # Applied to attention output (pre-residual)
        n_layer = getattr(config, 'n_layers', 12)
        self.attn_scale = 1.0 / (2 * n_layer) ** 0.5

        # Pre-compute RoPE embeddings for full head_dim
        # Using standard base=10000
        max_seq = getattr(config, 'max_position_embeddings', 4096)
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        self._init_rope(max_seq, rope_theta)

    def _init_rope(self, max_seq: int, base: float):
        """Initialize rotary position embeddings."""
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        self.register_buffer('_cos', cos, persistent=False)
        self.register_buffer('_sin', sin, persistent=False)

    def init_weights(self, init_std: float = 0.02):
        for proj in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    @record_function("attn")
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Forward pass. Note: cos/sin args are ignored - we use our own RoPE."""
        bsz, seqlen, _ = x.size()

        # Project Q, K, V
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K (using our own full-dim RoPE)
        q = _apply_rotary_emb(q, self._cos, self._sin)
        k = _apply_rotary_emb(k, self._cos, self._sin)

        # Transpose for attention: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        with record_function("attn.kernel[sdpa]"):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.softmax_scale,
            )

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(output) * self.attn_scale
