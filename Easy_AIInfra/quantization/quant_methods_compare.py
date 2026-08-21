"""
【题目】量化方法对比：对称/非对称 + Per-tensor/Per-channel/Per-token

【背景】
量化核心：把 FP16 张量映射到低比特整数。三种粒度：
- Per-tensor: 整个张量共享一组 (scale, zero_point)，最快但精度最差
- Per-channel: 沿输出通道维各有一组参数，平衡精度与速度，权重量化常用
- Per-token: 沿序列维各 token 一组参数，激活量化用，精度最高但开销大
对称量化: q = round(x / scale).clamp(-127,127)，zero_point=0
非对称量化: q = round((x - zero)/scale).clamp(0,255)，zero_point != 0

【输入/输出】
- 输入：FP16 张量，量化比特数 n_bits
- 输出：INT 量化张量，scale, zero_point

【考察点】
- 对称 vs 非对称量化适用场景
- Per-tensor/channel/token 粒度 trade-off
- 量化误差：MSE = mean((x - x_hat)^2)
"""
import torch


def symmetric_quantize(x: torch.Tensor, n_bits: int = 8) -> tuple:
    """对称量化：q = round(x/scale).clamp(qmin,qmax), scale = max(abs(x)) / qmax"""
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q.to(torch.int8), scale


def asymmetric_quantize(x: torch.Tensor, n_bits: int = 8) -> tuple:
    """非对称量化：zero = x.min(), scale = (x.max()-x.min())/(qmax-qmin)"""
    qmin = 0
    qmax = 2 ** n_bits - 1
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = x_min
    q = torch.round((x - zero_point) / scale).clamp(qmin, qmax)
    return q.to(torch.uint8), scale, zero_point


def per_channel_quantize(weight: torch.Tensor, n_bits: int = 8) -> tuple:
    """沿 dim=0 各通道独立量化，返回 scale [out,], q [out,in]"""
    qmax = 2 ** (n_bits - 1) - 1
    scale = weight.abs().max(dim=1).values / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(weight / scale.unsqueeze(1)).clamp(-qmax, qmax)
    return q.to(torch.int8), scale


def per_token_quantize(activation: torch.Tensor, n_bits: int = 8) -> tuple:
    """沿 dim=0 各 token 独立量化，返回 scale [B,1], q [B,in]"""
    qmax = 2 ** (n_bits - 1) - 1
    scale = activation.abs().max(dim=1, keepdim=True).values / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(activation / scale).clamp(-qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: float, zero_point: torch.Tensor = None) -> torch.Tensor:
    """反量化：x_hat = scale * q + zero_point"""
    if zero_point is not None:
        return q.float() * scale + zero_point
    return q.float() * scale


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(16, 64)
    x[:, 0] *= 5

    q, s = symmetric_quantize(x)
    dq = dequantize(q, s)
    err = (x - dq).abs().mean().item()
    assert q.dtype == torch.int8
    assert err < 0.1, f"对称量化误差过大: {err}"
    print(f"✅ 对称量化: MAE={err:.6f}")

    q_a, s_a, z_a = asymmetric_quantize(x)
    dq_a = dequantize(q_a, s_a, z_a)
    err_a = (x - dq_a).abs().mean().item()
    assert q_a.dtype == torch.uint8
    assert err_a < 0.1
    print(f"✅ 非对称量化: MAE={err_a:.6f}")

    W = torch.randn(32, 64)
    q_c, s_c = per_channel_quantize(W)
    assert s_c.shape == (32,)
    assert q_c.shape == W.shape
    dq_c = dequantize(q_c, s_c.unsqueeze(1))
    err_c = (W - dq_c).abs().mean().item()
    print(f"✅ Per-channel: MAE={err_c:.6f}")

    q_t, s_t = per_token_quantize(x)
    assert s_t.shape == (16, 1)
    assert q_t.shape == x.shape
    dq_t = dequantize(q_t, s_t)
    err_t = (x - dq_t).abs().mean().item()
    print(f"✅ Per-token: MAE={err_t:.6f}")

    x_pos = torch.relu(torch.randn(16, 64))
    _, s_sym, _ = asymmetric_quantize(x_pos)
    q_sym, s_sym2 = symmetric_quantize(x_pos)
    dq_sym = dequantize(q_sym, s_sym2)
    err_sym = (x_pos - dq_sym).abs().mean().item()
    q_asym, s_asym, z_asym = asymmetric_quantize(x_pos)
    dq_asym = dequantize(q_asym, s_asym, z_asym)
    err_asym = (x_pos - dq_asym).abs().mean().item()
    print(f"✅ ReLU 激活: 对称 MAE={err_sym:.6f}, 非对称 MAE={err_asym:.6f}")

    err_tensor = (x - dequantize(*symmetric_quantize(x))).abs().mean().item()
    err_channel = (x - dequantize(*per_channel_quantize(x))).abs().mean().item()
    err_token = (x - dequantize(*per_token_quantize(x))).abs().mean().item()
    print(f"✅ 粒度对比: tensor={err_tensor:.6f}, channel={err_channel:.6f}, token={err_token:.6f}")
    print("✅ 全部测试通过")
