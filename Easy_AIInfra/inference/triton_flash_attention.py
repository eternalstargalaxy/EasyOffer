"""
【题目】手撕 Triton FlashAttention（简化版）

【背景】
Triton 是 OpenAI 的 GPU kernel 编程语言，用 Python 写高效 GPU kernel。
FlashAttention 的 Triton 实现是学习 GPU 编程的经典案例。
核心: 分块计算 QK^T，online softmax，避免实例化完整 attention matrix。

【输入/输出】
输入: Q, K, V (batch, n_heads, seq_len, d_head)
输出: attention(Q, K, V)

【考察点】
- Triton kernel 的结构（@triton.jit + grid）
- 分块（tiling）策略
- online softmax 的数值技巧
- 本文件提供纯 Python 模拟版（无需 triton 安装）
"""

import torch
import torch.nn.functional as F
import math


def flash_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    """标准 attention 作为参考"""
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    if causal:
        seq_len = scores.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1) @ v


def flash_attention_tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64, causal: bool = False):
    """
    分块 FlashAttention（纯 Python 模拟 Triton kernel 逻辑）
    展示 online softmax + tiling 的核心算法
    """
    batch, n_heads, seq_len, d_head = q.shape
    scale = 1.0 / math.sqrt(d_head)
    output = torch.zeros_like(v)
    # 每个 query block 的全局 max 和 sum（用于 online softmax）
    for b in range(batch):
        for h in range(n_heads):
            # 按 block_size 分块处理 query
            for q_start in range(0, seq_len, block_size):
                q_end = min(q_start + block_size, seq_len)
                q_block = q[b, h, q_start:q_end]  # (q_block_size, d_head)
                # online softmax 状态
                m = torch.full((q_end - q_start,), float('-inf'))  # running max
                l = torch.zeros(q_end - q_start)  # running sum
                acc = torch.zeros(q_end - q_start, d_head)  # running output

                for k_start in range(0, seq_len, block_size):
                    k_end = min(k_start + block_size, seq_len)
                    k_block = k[b, h, k_start:k_end]
                    v_block = v[b, h, k_start:k_end]

                    # 计算 block 内 scores
                    scores = (q_block @ k_block.T) * scale  # (q_block_size, k_block_size)
                    if causal:
                        for i in range(q_end - q_start):
                            for j in range(k_end - k_start):
                                if q_start + i < k_start + j:
                                    scores[i, j] = float('-inf')

                    # Online softmax update
                    m_new = torch.maximum(m, scores.max(dim=-1).values)
                    alpha = torch.exp(m - m_new)  # 旧 block 的缩放
                    p = torch.exp(scores - m_new.unsqueeze(-1))  # 新 block 的 exp
                    l = l * alpha + p.sum(dim=-1)
                    acc = acc * alpha.unsqueeze(-1) + p @ v_block
                    m = m_new

                # 归一化
                output[b, h, q_start:q_end] = acc / l.unsqueeze(-1)
    return output


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, n_heads, seq_len, d_head = 1, 2, 32, 16
    q = torch.randn(batch, n_heads, seq_len, d_head)
    k = torch.randn(batch, n_heads, seq_len, d_head)
    v = torch.randn(batch, n_heads, seq_len, d_head)

    # 非因果
    ref_out = flash_attention_reference(q, k, v, causal=False)
    tiled_out = flash_attention_tiled(q, k, v, block_size=16, causal=False)
    assert torch.allclose(ref_out, tiled_out, atol=1e-5), "分块版应与参考版一致"
    print(f"非因果 max diff: {(ref_out - tiled_out).abs().max():.8f}")

    # 因果
    ref_causal = flash_attention_reference(q, k, v, causal=True)
    tiled_causal = flash_attention_tiled(q, k, v, block_size=16, causal=True)
    assert torch.allclose(ref_causal, tiled_causal, atol=1e-5)
    print(f"因果 max diff: {(ref_causal - tiled_causal).abs().max():.8f}")

    print("✅ Triton FlashAttention 分块版与参考版一致")