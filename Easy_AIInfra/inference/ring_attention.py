"""
【题目】RingAttention：长序列分布式注意力

【背景】
序列长度超过单卡显存时，RingAttention 沿 seq 维切分 Q/K/V 到多卡，
各卡轮流发送 K 块给下一卡，收到后算 attention + 更新 softmax。
通信：环形的 P2P send/recv，每卡需 P-1 轮传输，但每轮只传 1/P 数据。
与 FlashAttention 互补：FA 优化单卡显存，Ring 扩展到多卡。
应用：训练 128K+ 长上下文模型(如 Llama 3 long context)。

【输入/输出】
- 输入：Q,K,V 各卡持 1/P 的 seq chunk, rank, world_size
- 输出：各卡持自己那部分 O [B, L/P, D]

【考察点】
- ring 通信 vs all-to-all 通信 trade-off
- softmax rescale 在多轮传输间维护数值正确性
- 提示：注意梯度也需要 ring 反向传递
"""
import torch
import torch.nn.functional as F


def online_softmax_attention(qi: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor, mi: torch.Tensor, li: torch.Tensor, oi: torch.Tensor, scale: float) -> tuple:
    """online softmax 增量更新：返回更新后的 m, l, O。"""
    s = torch.matmul(qi, ki.transpose(-2, -1)) * scale
    m_block = s.max(dim=-1).values
    m_new = torch.maximum(mi, m_block)
    alpha = torch.exp(mi - m_new)
    beta = torch.exp(s - m_new.unsqueeze(-1))
    l_new = li * alpha + beta.sum(dim=-1)
    o_new = oi * alpha.unsqueeze(-1) + torch.matmul(beta, vi)
    return m_new, l_new, o_new


def ring_attention_block(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rank: int, world_size: int) -> list:
    """
    单机模拟 RingAttention：把 Q/K/V 沿 seq 维切成 world_size 块，
    模拟各卡环形传递 K/V 块并增量更新 attention。
    返回各卡的输出列表。
    """
    N = Q.shape[0]
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    chunk = N // world_size
    Q_chunks = [Q[i*chunk:(i+1)*chunk] for i in range(world_size)]
    K_chunks = [K[i*chunk:(i+1)*chunk] for i in range(world_size)]
    V_chunks = [V[i*chunk:(i+1)*chunk] for i in range(world_size)]

    outputs = []
    for r in range(world_size):
        qi = Q_chunks[r]
        mi = torch.full((chunk,), float('-inf'))
        li = torch.zeros(chunk)
        oi = torch.zeros(chunk, d)
        for step in range(world_size):
            k_src = (r - step) % world_size
            ki = K_chunks[k_src]
            vi = V_chunks[k_src]
            mi, li, oi = online_softmax_attention(qi, ki, vi, mi, li, oi, scale)
        oi = oi / li.unsqueeze(-1)
        outputs.append(oi)
    return outputs


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """朴素全注意力，用于验证。"""
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    s = torch.matmul(Q, K.T) * scale
    p = F.softmax(s, dim=-1)
    return torch.matmul(p, V)


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    N, d = 16, 8
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    for ws in [1, 2, 4]:
        out_chunks = ring_attention_block(Q, K, V, rank=0, world_size=ws)
        out_ring = torch.cat(out_chunks, dim=0)
        out_naive = naive_attention(Q, K, V)
        err = (out_ring - out_naive).abs().max().item()
        assert err < 1e-5, f"world_size={ws}: Ring 与朴素误差 {err} 过大"
        print(f"✅ world_size={ws}: Ring 与朴素结果一致 (误差 {err:.2e})")

    N2, d2 = 32, 16
    Q2 = torch.randn(N2, d2)
    K2 = torch.randn(N2, d2)
    V2 = torch.randn(N2, d2)
    out2 = ring_attention_block(Q2, K2, V2, rank=0, world_size=4)
    assert len(out2) == 4
    assert all(o.shape == (8, d2) for o in out2)
    print("✅ 各卡输出形状正确")
    print("✅ 全部测试通过")
