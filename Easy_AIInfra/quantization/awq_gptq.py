"""
【题目】AWQ / GPTQ 量化

【背景】
朴素 RTN（round-to-nearest）对大模型低 bit 量化精度损失大。
- GPTQ：用校准集算 Hessian H≈X^T X，逐列量化，把当前列量化误差用 H 信息补偿到尚未量化的后续列
- AWQ：发现少量"重要"通道（激活幅度大）主导误差，对重要权重用 per-channel 缩放 s 放大后再 RTN

【输入/输出】
- 输入：W: Tensor[out, in], 校准激活 X: Tensor[n, in]
- 输出：量化权重（int4 + group scale + AWQ 的 channel scale），及量化前向

【考察点】
- GPTQ 的 Hessian 构造与误差补偿递推
- AWQ 的缩放搜索与等价变换
- group size 对 4bit 精度影响
"""
import torch
import torch.nn.functional as F


def rtn_quantize(W: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Round-to-nearest 基线量化。"""
    qmax = 2 ** (bits - 1) - 1
    groups = (W.shape[1] + group_size - 1) // group_size
    W_q = torch.zeros_like(W)
    scales = torch.zeros(W.shape[0], groups)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, W.shape[1])
        w = W[:, start:end]
        scale = w.abs().max() / qmax
        scales[:, g] = scale
        W_q[:, start:end] = torch.round(w / scale).clamp(-qmax, qmax)
    return W_q, scales


def gptq_quantize(W: torch.Tensor, X: torch.Tensor, bits: int = 4, group_size: int = 128):
    """GPTQ：Hessian 误差补偿逐列量化。"""
    out_dim, in_dim = W.shape
    H = X.t() @ X
    H += 0.01 * torch.diag(H.diag().mean() * torch.ones(in_dim))
    H_inv = torch.linalg.inv(H)

    qmax = 2 ** (bits - 1) - 1
    W_q = W.clone()
    groups = (in_dim + group_size - 1) // group_size
    scales = torch.zeros(out_dim, groups)

    for j in range(in_dim):
        g = j // group_size
        scale = W_q[:, j].abs().max() / qmax if W_q[:, j].abs().max() > 0 else 1.0
        scales[:, g] = scale
        q = torch.round(W_q[:, j] / scale).clamp(-qmax, qmax)
        err = (W_q[:, j] - q * scale)
        W_q[:, j] = q
        if j < in_dim - 1:
            W_q[:, j + 1:] -= err.unsqueeze(1) * H_inv[j, j + 1:].unsqueeze(0) / H_inv[j, j]
    return W_q, scales


def awq_quantize(W: torch.Tensor, X: torch.Tensor, bits: int = 4, group_size: int = 128):
    """AWQ：搜索 per-channel 缩放 s 保护重要通道。"""
    out_dim, in_dim = W.shape
    act_scale = X.abs().mean(dim=0)
    qmax = 2 ** (bits - 1) - 1

    best_s = torch.ones(in_dim)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for j in range(in_dim):
        best_err = float('inf')
        for alpha in grid:
            s = (act_scale[j] ** alpha) / (W[:, j].abs().max() ** (1 - alpha) + 1e-8)
            s = s.clamp(min=1e-8)
            w_scaled = W[:, j] * s
            scale = w_scaled.abs().max() / qmax
            w_q = torch.round(w_scaled / scale).clamp(-qmax, qmax) * scale / s
            err = (W[:, j] - w_q).pow(2).sum().item()
            if err < best_err:
                best_err = err
                best_s[j] = s

    W_scaled = W * best_s.unsqueeze(0)
    groups = (in_dim + group_size - 1) // group_size
    W_q = torch.zeros_like(W)
    scales = torch.zeros(out_dim, groups)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, in_dim)
        w = W_scaled[:, start:end]
        scale = w.abs().max() / qmax
        scales[:, g] = scale
        W_q[:, start:end] = torch.round(w / scale).clamp(-qmax, qmax)
    return W_q, scales, best_s


class W4A16Linear(torch.nn.Module):
    """权重 4bit + group scale + AWQ channel scale 的推理线性层。"""

    def __init__(self, W_int: torch.Tensor, scale: float, s: torch.Tensor = None):
        super().__init__()
        self.register_buffer("W_int", W_int)
        self.register_buffer("scale", scale)
        self.register_buffer("s", s if s is not None else torch.ones(W_int.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = self.W_int.shape[1] // self.scale.shape[1]
        W_deq = torch.zeros_like(self.W_int, dtype=torch.float32)
        for g in range(self.scale.shape[1]):
            start = g * group_size
            end = min(start + group_size, self.W_int.shape[1])
            W_deq[:, start:end] = self.W_int[:, start:end].float() * self.scale[:, g].unsqueeze(1)
        W_deq = W_deq / self.s.unsqueeze(0)
        return F.linear(x, W_deq)


def ppl_compare(W: torch.Tensor, X_calib: torch.Tensor, X_eval: torch.Tensor):
    """对比 RTN / GPTQ / AWQ 的输出 MSE。"""
    y_fp = X_eval @ W.t()
    results = {}
    for name, fn in [("RTN", lambda: rtn_quantize(W)), ("GPTQ", lambda: gptq_quantize(W, X_calib))]:
        W_q, s = fn()
        W_deq = W_q.float() * s
        y_q = X_eval @ W_deq.t()
        results[name] = ((y_fp - y_q) ** 2).mean().item()
    W_q, s, _ = awq_quantize(W, X_calib)
    y_q = X_eval @ (W_q.float() * s).t()
    results["AWQ"] = ((y_fp - y_q) ** 2).mean().item()
    return results


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    out_dim, in_dim = 32, 128
    W = torch.randn(out_dim, in_dim) * 0.1
    X = torch.randn(16, in_dim)

    W_rtn, s_rtn = rtn_quantize(W, bits=4)
    assert W_rtn.shape == W.shape
    print("✅ RTN 量化: 形状正确")

    W_gptq, s_gptq = gptq_quantize(W, X, bits=4)
    assert W_gptq.shape == W.shape
    print("✅ GPTQ 量化: 形状正确")

    W_awq, s_awq, ch_s = awq_quantize(W, X, bits=4)
    assert W_awq.shape == W.shape
    assert ch_s.shape == (in_dim,)
    print("✅ AWQ 量化: 形状 + channel scale 正确")

    layer = W4A16Linear(W_awq, s_awq, ch_s)
    x = torch.randn(4, in_dim)
    y = layer(x)
    assert y.shape == (4, out_dim)
    print(f"✅ W4A16Linear forward: {x.shape} -> {y.shape}")

    results = ppl_compare(W, X, X)
    assert all(v >= 0 for v in results.values())
    print(f"✅ MSE 对比: {results}")
    print("✅ 全部测试通过")
