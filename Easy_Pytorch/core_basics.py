"""
【题目】基础算法：K-means 聚类 / 数值梯度检查

【背景】
面试常考的基础算法，用 PyTorch 实现。
- K-means：经典聚类算法，EM 迭代
- 数值梯度：用中心差分验证自动微分的正确性

【考察点】
- K-means 的 E 步（分配）和 M 步（更新中心）
- 数值梯度与自动微分的一致性
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def kmeans(X: torch.Tensor, k: int, max_iters: int = 100,
           tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means 聚类（PyTorch 版本，可 GPU 加速）。
    X: [N, D]，返回 centroids [k, D] 和 labels [N]。
    """
    N, D = X.shape
    indices = torch.randperm(N)[:k]
    centroids = X[indices].clone()

    for _ in range(max_iters):
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=-1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=X.dtype, device=X.device)
        new_centroids.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(N, dtype=X.dtype, device=X.device))
        new_centroids = new_centroids / counts.unsqueeze(-1).clamp(min=1)

        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break

    dists = torch.cdist(X, centroids)
    labels = dists.argmin(dim=-1)
    return centroids, labels


def numerical_gradient(f, x: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    """
    中心差分数值梯度：(f(x+h) - f(x-h)) / 2h。
    用于验证 autograd 的正确性。
    """
    grad = torch.zeros_like(x)
    flat_x = x.flatten()
    flat_grad = grad.flatten()
    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + h
        f_plus = f(x).item()
        flat_x[i] = orig - h
        f_minus = f(x).item()
        flat_x[i] = orig
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    X = torch.randn(100, 2)
    centroids, labels = kmeans(X, k=3)
    assert centroids.shape == (3, 2)
    assert labels.shape == (100,)
    assert labels.max() < 3 and labels.min() >= 0
    for i in range(3):
        assert (labels == i).sum() > 0, f"簇 {i} 不应为空"
    print(f"✅ K-means: 3 簇, 大小 {[(labels==i).sum().item() for i in range(3)]}")

    x = torch.randn(3, 2, requires_grad=True)
    f = lambda t: (t ** 2).sum()
    grad_num = numerical_gradient(f, x.detach().clone())
    loss = f(x)
    loss.backward()
    assert torch.allclose(grad_num, x.grad, atol=1e-4), f"数值梯度与 autograd 不一致"
    print("✅ numerical_gradient: 与 autograd 一致")

    X_gpu = torch.randn(50, 4)
    c, l = kmeans(X_gpu, k=5, max_iters=50)
    assert c.shape == (5, 4)
    print("✅ K-means: 不同维度正确")
    print("✅ 全部测试通过")
