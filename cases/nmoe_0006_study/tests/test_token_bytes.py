import math

import pytest


def test_token_bytes_gpt2_padded_and_special_zero():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tiktoken")

    from nmoe.token_bytes import token_bytes

    tb = token_bytes("gpt2", 50304)
    assert tb.shape == (50304,)
    assert tb.dtype == torch.int32

    # GPT-2 base vocab is 50257; this repo often pads to 50304.
    # Padded IDs must have 0 bytes.
    assert int(tb[50257].item()) == 0
    assert int(tb[50303].item()) == 0

    # Special token (<|endoftext|>) must be treated as 0 bytes for bpb.
    assert int(tb[50256].item()) == 0


def test_loss_nats_to_bpb_math():
    torch = pytest.importorskip("torch")

    from nmoe.token_bytes import loss_nats_to_bpb

    # ln(2)*8 nats over 1 byte -> 8 bits/byte
    loss = torch.tensor(float(math.log(2.0) * 8.0))
    bpb = loss_nats_to_bpb(loss, torch.tensor(1.0))
    assert float(bpb.item()) == pytest.approx(8.0, rel=0.0, abs=1e-6)
