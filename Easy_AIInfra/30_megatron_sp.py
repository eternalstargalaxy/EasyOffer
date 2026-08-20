"""
【题目】Megatron：序列并行 (Sequence Parallelism)

【背景】
长序列训练时，LayerNorm 和 Dropout 的激活沿 seq 维存储，显存 O(s*d)。
序列并行沿 seq 维切分到 TP 组各卡：每卡持 seq_len/TP 的激活，
LN 用局部 mean/var 后 all-reduce 得全局统计，Dropout 各卡独立。
与 TP 配合：TP 切权重，SP 切激活，两者正交可同时使用。
Megatron-LM 用 reduce-scatter 替代 all-reduce 优化 LN 通信。

【输入/输出】
- 输入：activation [B,S,D], tp_size, 分布在各卡上
- 输出：LN/Dropout/MLP 后的激活，各卡持 S/tp_size

【考察点】
- SP + TP 的组合通信模式
- LN all-reduce vs reduce-scatter 优化
- 提示：torch.distributed.all_reduce, reduce_scatter
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class SequenceParallelLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, tp_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def scatter_input(x: torch.Tensor, tp_size: int, rank: int):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, S, D = 4, 1024, 64
    print('ℹ' + " 序列并行需 torch.distributed 环境")
    print("验证点：LN 在 SP 下输出应与完整 LN 数值一致(误差<1e-5)")
