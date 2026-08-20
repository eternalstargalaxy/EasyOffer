"""
【题目】量化方法对比：对称/非对称 + Per-tensor/Per-channel/Per-token

【背景】
量化核心：把 FP16 张量映射到低比特整数。三种粒度：
- Per-tensor: 整个张量共享一组 (scale, zero_point)，最快但精度最差
- Per-channel: 沿输出通道维各有一组参数，平衡精度与速度，权重量化常用
- Per-token: 沿序列维各 token 一组参数，激活量化用，精度最高但开销大
对称量化: q = round(x / scale).clamp(-127,127)，zero_point=0
非对称量化: q = round((x - zero)/scale).clamp(0,255)，zero_point != 0
对称适合正态分布数据(权重)，非对称适合偏态分布(ReLU激活)。
反量化：x_hat = scale * q + zero_point

【输入/输出】
- 输入：FP16 张量，量化比特数 n_bits
- 输出：INT 量化张量，scale, zero_point

【考察点】
- 对称 vs 非对称量化适用场景
- Per-tensor/channel/token 粒度 trade-off
- 量化误差：MSE = mean((x - x_hat)^2)
- 提示：torch.aminmax 获取 min/max，torch.round 做量化映射
"""
import torch
import torch.nn as nn


def symmetric_quantize(x: torch.Tensor, n_bits: int = 8):
    """对称量化：q = round(x/scale).clamp(qmin,qmax), scale = max(abs(x)) / qmax"""
    raise NotImplementedError


def asymmetric_quantize(x: torch.Tensor, n_bits: int = 8):
    """非对称量化：zero = x.min(), scale = (x.max()-x.min())/(qmax-qmin)"""
    raise NotImplementedError


def per_channel_quantize(weight: torch.Tensor, n_bits: int = 8):
    """沿 dim=0(out_channels) 各通道独立量化，返回 scale [out,], q [out,in]"""
    raise NotImplementedError


def per_token_quantize(activation: torch.Tensor, n_bits: int = 8):
    """沿 dim=0(tokens) 各 token 独立量化，返回 scale [B,1], q [B,in]"""
    raise NotImplementedError


def dequantize(q: torch.Tensor, scale, zero_point=None):
    """反量化：x_hat = scale * q + zero_point"""
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == "__main__":
    x = torch.randn(16, 64)
    x[:, 0] *= 5  # 制造异常通道

    try:
        q, s = symmetric_quantize(x)
        dq = dequantize(q, s)
        err = (x - dq).abs().mean().item()
        print(f"对称量化 MAE: {err:.6f}")
        assert q.dtype in (torch.int8, torch.float32)
        print("\u2705 对称量化通过")
    except NotImplementedError:
        print("\u2139 待实现")

    try:
        q, s, z = asymmetric_quantize(x)
        dq = dequantize(q, s, z)
        err = (x - dq).abs().mean().item()
        print(f"非对称量化 MAE: {err:.6f}")
        print("\u2705 非对称量化通过")
    except NotImplementedError:
        print("\u2139 待实现")

    W = torch.randn(32, 64)
    try:
        q_c, s_c = per_channel_quantize(W)
        assert s_c.shape == (32,), f"scale shape: {s_c.shape}"
        assert q_c.shape == W.shape
        print("\u2705 Per-channel 量化通过")
    except NotImplementedError:
        print("\u2139 待实现")
