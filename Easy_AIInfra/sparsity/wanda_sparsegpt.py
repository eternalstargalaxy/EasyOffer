"""
【题目】Wanda / SparseGPT 剪枝

【背景】
大模型剪枝到高稀疏率时，Magnitude Pruning（按权重绝对值剪）精度损失大。
- Wanda：按 |W| * ||X|| 剪枝（权重幅度 × 输入激活范数），无需修改权重，一次前向即可
- SparseGPT：用 Hessian 信息逐列剪枝 + 误差补偿（类似 GPTQ），精度更高
两者都支持非结构化和 2:4 结构化稀疏。

【输入/输出】
- 输入：W [out, in], X [n, in], sparsity
- 输出：稀疏 W + mask

【考察点】
- Wanda 的重要性度量 |W| * ||X||
- SparseGPT 的 Hessian 补偿
- 提示：torch.topk 或 sort 选保留权重
"""
import torch
import torch.nn.functional as F


def magnitude_pruning(W: torch.Tensor, sparsity: float = 0.5):
    """Magnitude pruning：按 |W| 剪。"""
    threshold = torch.quantile(W.abs().flatten(), sparsity)
    mask = (W.abs() > threshold).float()
    return W * mask, mask


def wanda_pruning(W: torch.Tensor, X: torch.Tensor, sparsity: float = 0.5):
    """Wanda：按 |W| * ||X|| 剪枝。"""
    X_norm = X.abs().mean(dim=0)
    importance = W.abs() * X_norm.unsqueeze(0)
    threshold = torch.quantile(importance.flatten(), sparsity)
    mask = (importance > threshold).float()
    return W * mask, mask


def sparsegpt_pruning(W: torch.Tensor, X: torch.Tensor, sparsity: float = 0.5):
    """SparseGPT：Hessian 信息 + 误差补偿。"""
    H = X.t() @ X + 0.01 * torch.eye(X.shape[1]) * (X.t() @ X).diag().mean()
    W_pruned = W.clone()
    out_dim, in_dim = W.shape
    num_prune = int(in_dim * sparsity)

    for i in range(out_dim):
        w = W_pruned[i]
        importance = w.abs() * H.diag().sqrt()
        prune_idx = importance.topk(num_prune, largest=False).indices
        keep_idx = importance.topk(in_dim - num_prune, largest=True).indices
        err = w[prune_idx]
        W_pruned[i, prune_idx] = 0
        if len(keep_idx) > 0 and len(prune_idx) > 0:
            H_inv = torch.linalg.inv(H[keep_idx][:, keep_idx])
            compensation = err @ H[prune_idx][:, keep_idx] @ H_inv
            W_pruned[i, keep_idx] -= compensation

    mask = (W_pruned != 0).float()
    return W_pruned, mask


def pruning_error(W: torch.Tensor, W_sparse: torch.Tensor, X: torch.Tensor):
    """计算剪枝前后输出 MSE。"""
    y_orig = X @ W.t()
    y_sparse = X @ W_sparse.t()
    return ((y_orig - y_sparse) ** 2).mean().item()


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    out_dim, in_dim = 32, 64
    W = torch.randn(out_dim, in_dim) * 0.1
    X = torch.randn(16, in_dim)
    sparsity = 0.5

    W_mag, m_mag = magnitude_pruning(W, sparsity)
    assert (m_mag == 0).float().mean().item() == sparsity
    print(f"✅ Magnitude pruning: 稀疏率 {(m_mag==0).float().mean():.1%}")

    W_wanda, m_wanda = wanda_pruning(W, X, sparsity)
    assert (m_wanda == 0).float().mean().item() == sparsity
    print(f"✅ Wanda pruning: 稀疏率 {(m_wanda==0).float().mean():.1%}")

    W_sgpt, m_sgpt = sparsegpt_pruning(W, X, sparsity)
    actual_sparsity = (m_sgpt == 0).float().mean().item()
    print(f"✅ SparseGPT: 稀疏率 {actual_sparsity:.1%}")

    err_mag = pruning_error(W, W_mag, X)
    err_wanda = pruning_error(W, W_wanda, X)
    err_sgpt = pruning_error(W, W_sgpt, X)
    print(f"✅ 误差对比: Magnitude={err_mag:.6f}, Wanda={err_wanda:.6f}, SparseGPT={err_sgpt:.6f}")

    W_wanda_75, _ = wanda_pruning(W, X, 0.75)
    err_75 = pruning_error(W, W_wanda_75, X)
    assert err_75 > err_wanda, "更高稀疏率应有更大误差"
    print(f"✅ 稀疏率 75% 误差 {err_75:.6f} > 50% 误差 {err_wanda:.6f}")
    print("✅ 全部测试通过")
