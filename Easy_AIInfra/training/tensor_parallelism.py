"""
【题目】张量并行（Tensor Parallelism）

【背景】
单层权重太大放不下一卡时，按维度切到多卡，卡间通信藏在层内。Megatron 两个基本积木：
- 列并行（ColumnParallel）：权重按输出维切，Y = X·W = [X·W1, X·W2]，各卡算自己那部分输出；
  后接逐元素非线性（GeLU）时无需通信，可与下一层行并行无缝衔接。
- 行并行（RowParallel）：权重按输入维切，Y = X·W = X1·W1 + X2·W2，各卡部分和需 all-reduce。
组合 ColumnParallel→GeLU→RowParallel（即 MLP）全程只在最后做一次 all-reduce。

【输入/输出】
- 输入：X: Tensor[B, in_dim]，分布在 TP 组各卡（行并行时 X 已按 in_dim 切分）
- 输出：Y: Tensor[B, out_dim]，列并行各卡持部分输出；行并行 all-reduce 后各卡持完整输出

【考察点】
- 切分维度选择与通信点最小化
- 行并行前向 all-reduce ↔ 反向 split 的对称性
- Embedding 按 vocab 维切 + all-reduce
- 提示：torch.distributed.all_reduce 用于行并行输出聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """W 按输出维切：本卡持 W_i [in_dim, out_dim/N]，输出 [B, out_dim/N]"""

    def __init__(self, in_dim, out_dim, tp_size, rank):
        super().__init__()
        assert out_dim % tp_size == 0
        self.out_dim_local = out_dim // tp_size
        self.weight = nn.Parameter(torch.randn(in_dim, self.out_dim_local) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_dim_local))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.t(), self.bias)


class RowParallelLinear(nn.Module):
    """W 按输入维切：本卡持 W_i [in_dim/N, out_dim]，输入 [B, in_dim/N]，输出需 all-reduce"""

    def __init__(self, in_dim, out_dim, tp_size, rank):
        super().__init__()
        assert in_dim % tp_size == 0
        self.in_dim_local = in_dim // tp_size
        self.weight = nn.Parameter(torch.randn(self.in_dim_local, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight.t())
        if self.tp_size > 1:
            partial = self._all_reduce(partial)
        return partial + self.bias

    def _all_reduce(self, t):
        """单机模拟：直接返回（多卡时用 dist.all_reduce）。"""
        return t


class TPMLP(nn.Module):
    """ColumnParallelLinear -> GeLU -> RowParallelLinear，全程一次 all-reduce"""

    def __init__(self, dim, hidden, tp_size, rank):
        super().__init__()
        self.fc1 = ColumnParallelLinear(dim, hidden, tp_size, rank)
        self.fc2 = RowParallelLinear(hidden, dim, tp_size, rank)

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        return self.fc2(h)


class VocabParallelEmbedding(nn.Module):
    """按 vocab 维切 embedding，前向按 token 路由到持有该 vocab 段的卡。"""

    def __init__(self, vocab_size, dim, tp_size, rank):
        super().__init__()
        assert vocab_size % tp_size == 0
        self.vocab_local = vocab_size // tp_size
        self.vocab_start = rank * self.vocab_local
        self.vocab_end = self.vocab_start + self.vocab_local
        self.weight = nn.Parameter(torch.randn(self.vocab_local, dim) * 0.02)
        self.tp_size = tp_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids >= self.vocab_start) & (token_ids < self.vocab_end)
        local_ids = (token_ids - self.vocab_start).clamp(0, self.vocab_local - 1)
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1).float()
        if self.tp_size > 1:
            out = self._all_reduce(out)
        return out

    def _all_reduce(self, t):
        return t


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    B, in_d, out_d, tp = 4, 16, 32, 1

    col = ColumnParallelLinear(in_d, out_d, tp, rank=0)
    x = torch.randn(B, in_d)
    y = col(x)
    assert y.shape == (B, out_d // tp), f"列并行输出形状错误: {y.shape}"
    print(f"✅ ColumnParallelLinear: {x.shape} -> {y.shape}")

    row = RowParallelLinear(in_d, out_d, tp, rank=0)
    x_row = torch.randn(B, in_d // tp)
    y_row = row(x_row)
    assert y_row.shape == (B, out_d), f"行并行输出形状错误: {y_row.shape}"
    print(f"✅ RowParallelLinear: {x_row.shape} -> {y_row.shape}")

    mlp = TPMLP(in_d, 64, tp, rank=0)
    out = mlp(x)
    assert out.shape == (B, in_d), f"TPMLP 输出形状错误: {out.shape}"
    print(f"✅ TPMLP: {x.shape} -> {out.shape}")

    vocab, dim = 100, 16
    emb = VocabParallelEmbedding(vocab, dim, tp, rank=0)
    ids = torch.tensor([0, 50, 99, 30])
    emb_out = emb(ids)
    assert emb_out.shape == (4, dim), f"Embedding 输出形状错误: {emb_out.shape}"
    print(f"✅ VocabParallelEmbedding: {ids.shape} -> {emb_out.shape}")

    tp2 = 2
    col0 = ColumnParallelLinear(in_d, out_d, tp2, 0)
    col1 = ColumnParallelLinear(in_d, out_d, tp2, 1)
    y0 = col0(x)
    y1 = col1(x)
    y_full = torch.cat([y0, y1], dim=-1)
    assert y_full.shape == (B, out_d)
    print(f"✅ tp_size=2 列并行拼接: {y_full.shape}")
    print("✅ 全部测试通过")
