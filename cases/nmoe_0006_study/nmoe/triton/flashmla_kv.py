# SPDX-License-Identifier: Apache-2.0
"""FlashMLA KV cache packing.

Replaces nmoe/csrc/flashmla_kv.cu with pure PyTorch implementation.

KV cache layout per token (656 bytes):
  - [0:512]   FP8 E4M3FN latent (512 bytes)
  - [512:528] float32 scales (4 scales, 16 bytes)
  - [528:656] BF16 rope (64 values, 128 bytes)
"""

import torch


def _ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """Compute UE8M0 scale: pow2_ceil(amax / 448).

    UE8M0 is an 8-bit exponent-only format (no mantissa), so scales are always powers of 2.
    This matches the CUDA implementation's bit manipulation approach.
    """
    FP8_MAX = 448.0
    AMAX_CLAMP = 1e-4

    # Clamp amax to avoid log2(0)
    amax = torch.clamp(amax, min=AMAX_CLAMP)

    # Compute scale = 2^ceil(log2(amax / 448))
    log2_scale = torch.log2(amax / FP8_MAX)
    exp = torch.ceil(log2_scale)
    scale = torch.pow(2.0, exp)

    return scale


def flashmla_pack_kv_fp8_ue8m0_scatter(
    latent: torch.Tensor,  # [T, 512] BF16
    rope: torch.Tensor,    # [T, 64] BF16
    loc: torch.Tensor,     # [T] int64
    kv_out: torch.Tensor,  # [num_slots, 656] uint8
) -> None:
    """Pack KV cache in FP8 format with UE8M0 scales for FlashMLA.

    Args:
        latent: BF16 latent vectors [T, 512]
        rope: BF16 rope vectors [T, 64]
        loc: Destination slot indices [T]
        kv_out: Output KV cache [num_slots, 656] uint8
    """
    T = latent.shape[0]
    assert latent.shape == (T, 512), f"Expected latent shape [T, 512], got {latent.shape}"
    assert rope.shape == (T, 64), f"Expected rope shape [T, 64], got {rope.shape}"
    assert loc.shape == (T,), f"Expected loc shape [T], got {loc.shape}"
    assert kv_out.shape[1] == 656, f"Expected kv_out shape [*, 656], got {kv_out.shape}"

    # Create typed views into the kv_out buffer
    # Layout: [0:512] FP8, [512:528] float32 scales, [528:656] BF16 rope
    kv_out_fp8 = kv_out[:, :512].view(torch.float8_e4m3fn)  # [num_slots, 512] FP8
    kv_out_scale = kv_out[:, 512:528].view(torch.float32)   # [num_slots, 4] float32
    kv_out_rope = kv_out[:, 528:656].view(torch.bfloat16)   # [num_slots, 64] BF16

    # Process each token
    latent_f32 = latent.float()  # [T, 512]

    # Reshape to [T, 4, 128] for tile processing
    latent_tiles = latent_f32.view(T, 4, 128)

    # Compute amax per tile: [T, 4]
    amax = latent_tiles.abs().amax(dim=-1)

    # Compute UE8M0 scales: [T, 4]
    scales = _ue8m0_scale(amax)
    inv_scales = 1.0 / scales

    # Quantize to FP8: scale each tile
    # inv_scales: [T, 4] -> [T, 4, 1] for broadcasting
    scaled_latent = latent_tiles * inv_scales.unsqueeze(-1)

    # Clamp and convert to FP8
    FP8_MAX = 448.0
    scaled_latent = torch.clamp(scaled_latent, -FP8_MAX, FP8_MAX)
    fp8_latent = scaled_latent.to(torch.float8_e4m3fn).view(T, 512)

    # Scatter to output locations
    kv_out_fp8[loc] = fp8_latent
    kv_out_scale[loc] = scales
    kv_out_rope[loc] = rope
