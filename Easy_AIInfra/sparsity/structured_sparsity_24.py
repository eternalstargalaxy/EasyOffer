"""
【题目】2:4 结构化稀疏

【背景】
NVIDIA Ampere+ GPU 支持 2:4 结构化稀疏：每 4 个连续元素中恰好 2 个非零，
硬件自动跳过零元素，2x 矩阵乘加速。要求权重在训练/剪枝时满足 2:4 约束。
方法：对每 4 个权重按绝对值排序，保留 top-2，其余置零。

【输入/输出】
- 输入：dense weight [out, in]
- 输出：2:4 稀疏 weight + mask

【考察点】
- 2:4 约束的施加方法
- 稀疏 GEMM 的加速
- 提示：reshape 到 [..., 4] 取 top-2
"""
import torch
import torch.nn as nn


def apply_24_sparsity(W: torch.Tensor):
    """对权重施加 2:4 稀疏：每 4 个保留 top-2。"""
    out_dim, in_dim = W.shape
    assert in_dim % 4 == 0, "in_dim 必须是 4 的倍数"
    W_reshaped = W.view(out_dim, in_dim // 4, 4)
    abs_val = W_reshaped.abs()
    top2 = abs_val.topk(2, dim=-1).indices
    mask = torch.zeros_like(W_reshaped)
    mask.scatter_(-1, top2, 1.0)
    W_sparse = W_reshaped * mask
    return W_sparse.view(out_dim, in_dim), mask.view(out_dim, in_dim)


def check_24_sparsity(W: torch.Tensor):
    """验证是否满足 2:4 约束。"""
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim, in_dim // 4, 4)
    nonzero_count = (W_reshaped != 0).sum(dim=-1)
    return (nonzero_count <= 2).all().item()


def sparse_24_linear(x: torch.Tensor, W: torch.Tensor, bias: torch.Tensor = None):
    """2:4 稀疏线性层前向。"""
    W_sparse, mask = apply_24_sparsity(W)
    return torch.nn.functional.linear(x, W_sparse, bias)


class Sparse24Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor):
        W_sparse, _ = apply_24_sparsity(self.weight)
        return torch.nn.functional.linear(x, W_sparse, self.bias)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    W = torch.randn(32, 16)
    W_sparse, mask = apply_24_sparsity(W)
    assert W_sparse.shape == W.shape
    assert check_24_sparsity(W_sparse), "应满足 2:4 约束"
    sparsity = (W_sparse == 0).float().mean().item()
    assert 0.45 < sparsity < 0.55, f"稀疏率应约 50%, 实际 {sparsity}"
    print(f"✅ 2:4 稀疏: 稀疏率 {sparsity:.1%}")

    W2 = torch.randn(8, 8)
    assert not check_24_sparsity(W2), "随机权重不应满足 2:4"
    W2_sparse, _ = apply_24_sparsity(W2)
    assert check_24_sparsity(W2_sparse), "施加后应满足"
    print("✅ 约束验证正确")

    x = torch.randn(4, 16)
    y = sparse_24_linear(x, W)
    assert y.shape == (4, 32)
    print(f"✅ Sparse linear: {x.shape} -> {y.shape}")

    layer = Sparse24Linear(16, 32)
    y2 = layer(x)
    assert y2.shape == (4, 32)
    print("✅ Sparse24Linear forward 正确")

    W_reshaped = W_sparse.view(32, 4, 4)
    for i in range(32):
        for j in range(4):
            assert (W_reshaped[i, j] != 0).sum() <= 2
    print("✅ 每 4 元素最多 2 个非零")
    print("✅ 全部测试通过")
