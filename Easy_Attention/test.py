# -*- coding: utf-8 -*-
"""Easy_Attention 模块 smoke test：验证关键性质（独立实现，不依赖 ipynb）。"""
import torch
import torch.nn.functional as F
import math

from core_attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    absolute_position_encoding,
    RotaryPositionEmbedding,
    apply_rotary_emb,
    SwiGLU,
)


def test_softmax_stable():
    x = torch.tensor([0.0, 7.0, 6.0, 12.0, 10.0], dtype=torch.float16)
    m = x.max()
    e = torch.exp(x - m)
    s = e / e.sum()
    assert torch.isfinite(s).all() and abs(s.sum().item() - 1) < 1e-3
    assert torch.allclose(s.float(), F.softmax(x.float(), dim=0), atol=1e-3)


def test_rmsnorm_unit_rms():
    dim = 64
    w = torch.ones(dim)
    x = torch.randn(4, dim)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    rms = out.pow(2).mean(-1).sqrt()
    assert (rms - 1).abs().max() < 1e-4


def test_rope_preserves_norm():
    dim = 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(4).float(), inv_freq)
    cos = torch.cat((freqs, freqs), dim=-1).cos()
    sin = torch.cat((freqs, freqs), dim=-1).sin()
    q = torch.randn(2, 4, 1, dim)
    x1, x2 = q[..., :dim // 2], q[..., dim // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    q_emb = q * cos + rot * sin
    assert torch.allclose(q.norm(), q_emb.norm(), atol=1e-5)


def test_lora_zero_init():
    A = torch.randn(16, 4)
    B = torch.zeros(4, 16)
    assert (A @ B).abs().max() == 0


def test_gqa_repeat_matches_mha_when_g_eq_h():
    # 当 KV 头数 == Q 头数时，GQA 退化为 MHA，repeat 不改变结果
    h, g = 8, 8
    k = torch.randn(1, g, 5, 4)
    k_exp = k.repeat_interleave(h // g, dim=1)
    assert k_exp.shape == (1, h, 5, 4) and torch.equal(k_exp, k)


# ========== core_attention.py 覆盖测试 ==========

def test_sdpa_shape():
    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, S, D), f"SDPA shape: {out.shape}"


def test_sdpa_causal_mask():
    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    causal = torch.tril(torch.ones(S, S)).bool()
    out = scaled_dot_product_attention(q, k, v, mask=causal)
    assert out.shape == (B, H, S, D)


def test_sdpa_different_lengths():
    B, H, Sq, Sk, D = 2, 4, 6, 10, 16
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, Sk, D)
    v = torch.randn(B, H, Sk, D)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, Sq, D)


def test_mha_forward():
    B, S, D = 2, 8, 32
    mha = MultiHeadAttention(D, n_heads=4)
    x = torch.randn(B, S, D)
    out = mha(x)
    assert out.shape == (B, S, D)


def test_mha_gradient():
    B, S, D = 2, 8, 32
    mha = MultiHeadAttention(D, n_heads=4)
    x = torch.randn(B, S, D)
    out = mha(x)
    out.sum().backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"


def test_absolute_pe_shape():
    pe = absolute_position_encoding(20, 64)
    assert pe.shape == (20, 64)
    assert pe[0, 0] == 0.0, "position 0, dim 0 sin should be 0"
    assert abs(pe[0, 1] - 1.0) < 1e-6, "position 0, dim 1 cos should be 1"


def test_absolute_pe_uniqueness():
    pe = absolute_position_encoding(100, 16)
    diffs = (pe[1:] - pe[:-1]).abs().sum(dim=1)
    assert (diffs > 0).all(), "adjacent position encodings should differ"


def test_rope_preserves_norm():
    dim = 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(4).float(), inv_freq)
    cos = torch.cat((freqs, freqs), dim=-1).cos()
    sin = torch.cat((freqs, freqs), dim=-1).sin()
    q = torch.randn(2, 4, 1, dim)
    x1, x2 = q[..., :dim // 2], q[..., dim // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    q_emb = q * cos + rot * sin
    assert torch.allclose(q.norm(), q_emb.norm(), atol=1e-5)


def test_rope_relative_property():
    """同一向量旋转不同角度后范数不变"""
    dim = 4
    rope = RotaryPositionEmbedding(dim, max_seq_len=32)
    x = torch.randn(1, 1, 1, dim)
    cos = rope.cos_cached[:8].unsqueeze(0).unsqueeze(0)
    sin = rope.sin_cached[:8].unsqueeze(0).unsqueeze(0)

    # Rotate at pos=0, 1, 2
    x0 = apply_rotary_emb(x, cos[:, :, :1, :], sin[:, :, :1, :])
    x1 = apply_rotary_emb(x, cos[:, :, 1:2, :], sin[:, :, 1:2, :])
    x2 = apply_rotary_emb(x, cos[:, :, 2:3, :], sin[:, :, 2:3, :])
    # 范数都应相等
    assert torch.allclose(x0.norm(), x.norm(), atol=1e-5)
    assert torch.allclose(x1.norm(), x.norm(), atol=1e-5)
    assert torch.allclose(x2.norm(), x.norm(), atol=1e-5)


def test_rope_module_output():
    rope = RotaryPositionEmbedding(16, max_seq_len=64)
    x = torch.randn(2, 4, 8, 16)
    cos, sin = rope(x, seq_len=8)
    x_rot = apply_rotary_emb(x, cos, sin)
    assert x_rot.shape == x.shape


def test_swiglu_shape():
    B, S, D, H = 2, 8, 64, 128
    swiglu = SwiGLU(D, H)
    x = torch.randn(B, S, D)
    out = swiglu(x)
    assert out.shape == (B, S, D)


def test_swiglu_nonlinear():
    """SwiGLU 应是非线性的：输入翻倍不等于输出翻倍"""
    swiglu = SwiGLU(16, 32)
    x = torch.randn(1, 4, 16)
    y1 = swiglu(x)
    y2 = swiglu(2.0 * x)
    assert not torch.allclose(y2, 2.0 * y1, atol=1e-4), "SwiGLU should be nonlinear"


def test_swiglu_gradient():
    swiglu = SwiGLU(16, 32)
    x = torch.randn(1, 4, 16)
    out = swiglu(x)
    out.sum().backward()
    for name, p in swiglu.named_parameters():
        assert p.grad is not None, f"SwiGLU {name} grad is None"
        assert p.grad.abs().sum() > 0, f"SwiGLU {name} grad is zero"


if __name__ == "__main__":
    all_tests = [
        test_softmax_stable, test_rmsnorm_unit_rms,
        test_lora_zero_init, test_gqa_repeat_matches_mha_when_g_eq_h,
        # core_attention.py tests
        test_sdpa_shape, test_sdpa_causal_mask, test_sdpa_different_lengths,
        test_mha_forward, test_mha_gradient,
        test_absolute_pe_shape, test_absolute_pe_uniqueness,
        test_rope_preserves_norm, test_rope_relative_property, test_rope_module_output,
        test_swiglu_shape, test_swiglu_nonlinear, test_swiglu_gradient,
    ]
    for fn in all_tests:
        torch.manual_seed(42)
        fn()
        print("ok:", fn.__name__)
    print(f"ALL {len(all_tests)} TESTS PASSED")
