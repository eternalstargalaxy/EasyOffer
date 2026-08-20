"""
【题目】SmoothQuant：激活-权重联合量化

【背景】
W8A8 量化需在激活和权重间平滑数值分布。LLM 激活有异常值通道(outlier channels)，
这些通道激活值远大于其他通道，导致 token-wise 量化动态范围被拉大、精度下降。
SmoothQuant 用平滑因子 s 拉平: Y = XW = (X diag(s)^-1) * (diag(s) * W)
其中 s_j = max(abs(X_j))^alpha / max(abs(W_j))^(1-alpha)，默认 alpha=0.5。

【输入/输出】
- 输入：weight [out,in], activation X [B,in], alpha
- 输出：量化后 INT8 weight, quant scale

【考察点】
- per-channel smooth 消除激活异常值原理
- alpha 控制迁移强度(alpha=1 等价 weight-only 量化)
- 提示：torch.clamp 做 INT8 范围限制，torch.round 做最近邻量化
"""
import torch
import torch.nn as nn


def compute_smooth_scale(X_absmax: torch.Tensor, W_absmax: torch.Tensor,
                         alpha: float = 0.5) -> torch.Tensor:
    """s_j = max(abs(X_j))^alpha / max(abs(W_j))^(1-alpha)"""
    s = (X_absmax ** alpha) / (W_absmax ** (1 - alpha) + 1e-8)
    return s.clamp(min=1e-8)


def smooth_quantize_weight(weight: torch.Tensor, scale: torch.Tensor,
                           n_bits: int = 8):
    """W_smooth = diag(scale)*W; 对称量化到 INT8"""
    qmax = 2 ** (n_bits - 1) - 1
    W_smooth = weight * scale.unsqueeze(0)
    w_scale = W_smooth.abs().max(dim=1).values / qmax
    w_scale = w_scale.clamp(min=1e-8)
    W_int = torch.round(W_smooth / w_scale.unsqueeze(1)).clamp(-qmax, qmax)
    return W_int.to(torch.int8), w_scale


def smooth_quantize_activation(X: torch.Tensor, scale: torch.Tensor,
                               n_bits: int = 8):
    """X_smooth = X * diag(scale)^-1; 对称量化到 INT8"""
    qmax = 2 ** (n_bits - 1) - 1
    X_smooth = X / scale.unsqueeze(0)
    x_scale = X_smooth.abs().max() / qmax
    x_scale = x_scale.clamp(min=1e-8)
    X_int = torch.round(X_smooth / x_scale).clamp(-qmax, qmax)
    return X_int.to(torch.int8), x_scale


def smooth_quant_forward(X_int, x_scale, W_int, w_scale, scale):
    """反量化 + 前向：验证等价性。"""
    X_smooth = X_int.float() * x_scale
    W_smooth = W_int.float() * w_scale.unsqueeze(1)
    X_orig = X_smooth * scale.unsqueeze(0)
    W_orig = W_smooth / scale.unsqueeze(0)
    return X_orig @ W_orig.t()


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    W = torch.randn(out_dim, in_dim)
    W[:, 0] *= 10
    X_absmax = torch.randn(in_dim).abs()
    X_absmax[0] *= 10

    s = compute_smooth_scale(X_absmax, W.norm(dim=0), alpha=0.5)
    assert s.shape == (in_dim,), f"shape wrong: {s.shape}"
    assert (s > 0).all(), "scale must be positive"
    assert s[0] > 1.0, "outlier channel scale should > 1"
    print(f"✅ compute_smooth_scale: s[0]={s[0]:.3f} (outlier), s[1]={s[1]:.3f}")

    W_q, s_w = smooth_quantize_weight(W, torch.ones(in_dim))
    assert W_q.shape == W.shape
    assert W_q.dtype == torch.int8
    print(f"✅ smooth_quantize_weight: {W.shape} -> int8")

    X = torch.randn(4, in_dim)
    X[:, 0] *= 10
    X_q, s_x = smooth_quantize_activation(X, s)
    assert X_q.shape == X.shape
    assert X_q.dtype == torch.int8
    print(f"✅ smooth_quantize_activation: {X.shape} -> int8")

    y_smooth = smooth_quant_forward(X_q, s_x, W_q, s_w, s)
    y_ref = X @ W.t()
    rel_err = (y_smooth - y_ref).abs().mean().item() / y_ref.abs().mean().item()
    assert rel_err < 0.1, f"平滑量化误差过大: {rel_err}"
    print(f"✅ 等价性: 相对误差 {rel_err:.4f}")

    s_a1 = compute_smooth_scale(X_absmax, W.norm(dim=0), alpha=1.0)
    s_a0 = compute_smooth_scale(X_absmax, W.norm(dim=0), alpha=0.0)
    assert s_a1[0] > s_a0[0], "alpha=1 应更激进"
    print("✅ alpha=1 vs alpha=0: 激进程度正确")
    print("✅ 全部测试通过")
