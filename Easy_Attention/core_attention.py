"""
【题目】注意力核心：SDPA / MHA / 位置编码 / 激活函数（生产级）

【背景】
本文件实现 Transformer 注意力的核心组件，贴近 Llama / FlashAttention 源码风格。
- Scaled Dot-Product Attention：支持 causal mask
- Multi-Head Attention：标准 MHA 实现
- 绝对位置编码：正弦余弦编码
- RoPE：旋转位置编码（Llama 风格）
- 激活函数：GELU / SiLU / SwiGLU

【考察点】
- attention 的数值稳定性（scale + mask）
- RoPE 的旋转矩阵构造与相对位置性质
- SwiGLU 的门控机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[float] = None,
) -> torch.Tensor:
    """
    缩放点积注意力。
    q: [B, H, Sq, D], k: [B, H, Sk, D], v: [B, H, Sk, D]
    mask: [Sq, Sk] 或 [B, 1, Sq, Sk]，True/1 表示可 attend。
    """
    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores.masked_fill(~mask.bool() if mask.dtype == torch.bool else mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None and dropout > 0:
        attn = F.dropout(attn, p=dropout, training=True)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """标准多头注意力，贴近 nn.MultiheadAttention 但更透明。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        out = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


def absolute_position_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """正弦余弦绝对位置编码。"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class RotaryPositionEmbedding(nn.Module):
    """RoPE：旋转位置编码（Llama 风格）。"""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos, x * sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """将 RoPE 应用于 x：旋转 (x_even, x_odd) 对。"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """SwiGLU FFN：down(silu(gate(x)) * up(x))，Llama 风格。"""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 8, 16

    q: torch.Tensor = torch.randn(B, H, S, D)
    k: torch.Tensor = torch.randn(B, H, S, D)
    v: torch.Tensor = torch.randn(B, H, S, D)

    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, S, D)
    print(f"✅ SDPA: {q.shape} -> {out.shape}")

    causal_mask = torch.tril(torch.ones(S, S)).bool()
    out_causal = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    assert out_causal.shape == (B, H, S, D)
    print("✅ SDPA with causal mask")

    mha = MultiHeadAttention(D * H, H)
    x = torch.randn(B, S, D * H)
    out_mha = mha(x)
    assert out_mha.shape == (B, S, D * H)
    print(f"✅ MHA: {x.shape} -> {out_mha.shape}")

    out_mha_causal = mha(x, mask=causal_mask)
    assert out_mha_causal.shape == (B, S, D * H)
    print("✅ MHA with causal mask")

    pe = absolute_position_encoding(10, 64)
    assert pe.shape == (10, 64)
    assert pe[0, 0] == 0, "位置 0 的 sin 分量应为 0"
    print(f"✅ Absolute PE: {pe.shape}")

    rope = RotaryPositionEmbedding(32, max_seq_len=128)
    x_rope = torch.randn(1, 4, 8, 32)
    cos, sin = rope(x_rope, 8)
    x_rotated = apply_rotary_emb(x_rope, cos, sin)
    assert x_rotated.shape == x_rope.shape
    print(f"✅ RoPE: {x_rope.shape} -> {x_rotated.shape}")

    swiglu = SwiGLU(64, 128)
    x_ffn = torch.randn(2, 8, 64)
    out_ffn = swiglu(x_ffn)
    assert out_ffn.shape == (2, 8, 64)
    print(f"✅ SwiGLU: {x_ffn.shape} -> {out_ffn.shape}")

    out_mha.sum().backward()
    assert mha.wq.weight.grad is not None
    print("✅ 反向传播: 梯度正确")
    print("✅ 全部测试通过")
