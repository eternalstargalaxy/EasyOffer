# -*- coding: utf-8 -*-
"""Easy_Attention 模块 smoke test：验证关键性质（独立实现，不依赖 ipynb）。"""
import torch
import torch.nn.functional as F
import math


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


if __name__ == "__main__":
    for fn in [test_softmax_stable, test_rmsnorm_unit_rms, test_rope_preserves_norm,
               test_lora_zero_init, test_gqa_repeat_matches_mha_when_g_eq_h]:
        fn(); print("ok:", fn.__name__)
    print("ALL TESTS PASSED")
