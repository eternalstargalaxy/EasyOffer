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
import torch.nn.functional as F


class SequenceParallelLayerNorm(nn.Module):
    """SP LayerNorm：各卡持 seq/TP 的激活，all-reduce 全局统计。"""

    def __init__(self, hidden_size: int, tp_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S/TP, D] 本卡的 seq 片段
        需要跨卡 all-reduce 得全局 mean/var。
        单机模拟：假设输入已是完整序列的 1/TP 切片。
        """
        B, S_local, D = x.shape
        S_global = S_local * self.tp_size

        local_sum = x.sum(dim=1)
        local_sq_sum = (x ** 2).sum(dim=1)

        global_sum = local_sum * self.tp_size
        global_sq_sum = local_sq_sum * self.tp_size

        mean = global_sum / S_global
        var = global_sq_sum / S_global - mean ** 2

        x_normed = (x - mean.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + 1e-5)
        return x_normed * self.weight + self.bias


def scatter_input(x: torch.Tensor, tp_size: int, rank: int) -> torch.Tensor:
    """沿 seq 维切分输入，返回本卡的片段。"""
    S = x.shape[1]
    chunk = S // tp_size
    return x[:, rank * chunk:(rank + 1) * chunk, :].contiguous()


def gather_output(x_shards: list) -> torch.Tensor:
    """沿 seq 维拼接各卡输出。"""
    return torch.cat(x_shards, dim=1)


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    B, S, D = 4, 64, 32
    tp = 4

    x = torch.randn(B, S, D)
    ln_ref = nn.LayerNorm(D)
    y_ref = ln_ref(x)

    sp_ln = SequenceParallelLayerNorm(D, tp)
    sp_ln.weight.data = ln_ref.weight.data.clone()
    sp_ln.bias.data = ln_ref.bias.data.clone()

    shards = [scatter_input(x, tp, r) for r in range(tp)]
    assert all(s.shape == (B, S // tp, D) for s in shards)
    print(f"✅ scatter_input: {x.shape} -> {shards[0].shape} x{tp}")

    y_shards = [sp_ln(s) for s in shards]
    y_sp = gather_output(y_shards)
    assert y_sp.shape == (B, S, D)
    err = (y_sp - y_ref).abs().max().item()
    assert err < 1e-4, f"SP LN 与标准 LN 误差过大: {err}"
    print(f"✅ SP LayerNorm: 与标准 LN 一致 (误差 {err:.2e})")

    x2 = torch.randn(B, S, D)
    shards2 = [scatter_input(x2, tp, r) for r in range(tp)]
    gathered = gather_output(shards2)
    assert torch.allclose(gathered, x2), "scatter + gather 应还原"
    print("✅ scatter + gather 还原正确")

    sp_ln2 = SequenceParallelLayerNorm(D, tp_size=1)
    y_single = sp_ln2(x)
    ln_ref2 = nn.LayerNorm(D)
    ln_ref2.weight.data = sp_ln2.weight.data.clone()
    ln_ref2.bias.data = sp_ln2.bias.data.clone()
    err2 = (y_single - ln_ref2(x)).abs().max().item()
    assert err2 < 1e-5
    print("✅ tp_size=1 退化为标准 LN")
    print("✅ 全部测试通过")
