"""
【题目】FlashAttention（tiling + online softmax）

【背景】
标准 attention 中间矩阵 S=QK^T 占 O(N²) 显存且读写密集。
FlashAttention 把 Q/K/V 切块加载到 SRAM，用 online softmax 增量更新行最大值 m 与行和 l，
不在 HBM 里物化 N² 矩阵，IO 复杂度从 O(N²d) 降到 O(N²d²/M)。
关键：每加入一个新 K/V 块，要用新 m_new 校正已累加的 O 与 l（乘以 exp(m_old - m_new)）。

【输入/输出】
- 输入：Q,K,V: Tensor[N, d]（单头，多头时外层循环 head）
- 输出：O: Tensor[N, d]，与朴素 softmax(QK^T/√d)V 在 fp 误差内一致

【考察点】
- online softmax 的 m/l 校正公式正确性
- tiling 顺序与 SRAM 容量 M 假设
- 因果 mask 在分块下整块跳过 / 部分掩码
- 提示：分块计算避免实例化完整 n*n 矩阵
"""
import torch
import torch.nn.functional as F


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """朴素实现，用作对照。"""
    N, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    S = torch.matmul(Q, K.T) * scale
    if causal:
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
        S = S.masked_fill(~mask, float('-inf'))
    P = F.softmax(S, dim=-1)
    return torch.matmul(P, V)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    block_size: int, causal: bool = False) -> torch.Tensor:
    """
    FlashAttention 分块 + online softmax。
    外层遍历 Q 块 i，维护该块的 m [Bq], l [Bq], O [Bq, d]
    内层遍历 K/V 块 j，增量更新 m/l/O。
    """
    N, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    O = torch.zeros(N, d)
    m = torch.full((N,), float('-inf'))
    l = torch.zeros(N)

    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Qi = Q[i_start:i_end]
        mi = m[i_start:i_end].clone()
        li = l[i_start:i_end].clone()
        Oi = O[i_start:i_end].clone()

        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            Kj = K[j_start:j_end]
            Vj = V[j_start:j_end]

            Sij = torch.matmul(Qi, Kj.T) * scale

            if causal:
                for ii in range(i_end - i_start):
                    for jj in range(j_end - j_start):
                        global_i = i_start + ii
                        global_j = j_start + jj
                        if global_j > global_i:
                            Sij[ii, jj] = float('-inf')

            m_block = Sij.max(dim=-1).values
            m_new = torch.maximum(mi, m_block)
            alpha = torch.exp(mi - m_new)
            beta = torch.exp(Sij - m_new.unsqueeze(-1))
            li = li * alpha + beta.sum(dim=-1)
            Oi = Oi * alpha.unsqueeze(-1) + torch.matmul(beta, Vj)
            mi = m_new

        O[i_start:i_end] = Oi
        m[i_start:i_end] = mi
        l[i_start:i_end] = li

    O = O / l.unsqueeze(-1)
    return O


def io_complexity(N: int, d: int, M: int):
    """返回朴素 vs Flash 的 HBM IO 量"""
    naive_io = N * N * d + N * d
    flash_io = 2 * N * N * d * d / M
    return {"naive": naive_io, "flash": flash_io, "speedup": naive_io / flash_io}


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    N, d = 16, 8
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    out_naive = naive_attention(Q, K, V, causal=False)
    out_flash = flash_attention(Q, K, V, block_size=4, causal=False)
    assert out_flash.shape == (N, d)
    max_err = (out_naive - out_flash).abs().max().item()
    assert max_err < 1e-5, f"Flash 与朴素结果误差过大: {max_err}"
    print(f"✅ 无因果 mask: Flash 与朴素结果一致 (误差 {max_err:.2e})")

    out_naive_c = naive_attention(Q, K, V, causal=True)
    out_flash_c = flash_attention(Q, K, V, block_size=4, causal=True)
    max_err_c = (out_naive_c - out_flash_c).abs().max().item()
    assert max_err_c < 1e-5, f"因果 Flash 误差过大: {max_err_c}"
    print(f"✅ 因果 mask: Flash 与朴素结果一致 (误差 {max_err_c:.2e})")

    for bs in [1, 2, 8, 16]:
        out_bs = flash_attention(Q, K, V, block_size=bs, causal=False)
        err = (out_naive - out_bs).abs().max().item()
        assert err < 1e-5, f"block_size={bs} 误差过大: {err}"
    print("✅ 不同 block_size 结果一致")

    ioc = io_complexity(1024, 64, 1024)
    assert ioc["speedup"] > 1
    print(f"✅ IO 复杂度: naive={ioc['naive']}, flash={ioc['flash']:.0f}, 加速 {ioc['speedup']:.1f}x")
    print("✅ 全部测试通过")
