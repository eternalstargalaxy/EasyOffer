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
"""
import torch


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """朴素实现，用作对照。"""
    raise NotImplementedError


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    block_size: int, causal: bool = False) -> torch.Tensor:
    """
    外层遍历 Q 块 i，维护该块的 m [Bq], l [Bq], O [Bq, d]（初始 m=-inf, l=0, O=0）
    内层遍历 K/V 块 j：
      S = Qi @ Kj^T * scale            # [Bq, Bk]
      m_new = max(m, rowmax(S))
      l = l * exp(m - m_new) + rowsum(exp(S - m_new))
      O = O * exp(m - m_new) + exp(S - m_new) @ Vj
      m = m_new
    causal 时按块位置整块跳过（j 上三角）或部分掩码（对角块）。
    """
    raise NotImplementedError


def io_complexity(N: int, d: int, M: int):
    """返回朴素 vs Flash 的 HBM IO 量"""
    raise NotImplementedError
