"""
【题目】W8A16 线性层量化推理

【背景】
推理时把权重压成 int8（或 int4），激活保持 fp16，反量化后做 fp16 GEMM。
省显存、减权重带宽，对 decode（memory-bound，权重读一次只算 1 token）特别友好。
推理只量化权重不量化激活，因为激活逐 token 变化且对精度敏感，量化收益小风险大。
对称量化：W_int = round(W/scale)，scale = max(|W|)/127；非对称再加 zero-point。

【输入/输出】
- 输入：W: Tensor[out_dim, in_dim] (fp16/fp32), x: Tensor[B, in_dim] (fp16)
- 输出：W8A16Linear(x) ≈ x @ W^T + bias，权重以 int8 + scale 存储

【考察点】
- scale 维度选择（per-tensor/channel/group）对精度影响
- 反量化时机（是否融合进 GEMM epilogue）
- 对称 vs 非对称、int8 vs int4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_w8a16(W: torch.Tensor, granularity: str = "per_channel",
                   group_size: int = 128):
    """W: [out_dim, in_dim] -> W_int8, scale"""
    qmax = 127
    if granularity == "per_tensor":
        scale = W.abs().max() / qmax
        W_int = torch.round(W / scale).clamp(-qmax, qmax).to(torch.int8)
        return W_int, scale
    elif granularity == "per_channel":
        scale = W.abs().max(dim=1).values / qmax
        scale = scale.clamp(min=1e-8)
        W_int = torch.round(W / scale.unsqueeze(1)).clamp(-qmax, qmax).to(torch.int8)
        return W_int, scale
    elif granularity == "per_group":
        out_dim, in_dim = W.shape
        groups = (in_dim + group_size - 1) // group_size
        W_int = torch.zeros_like(W, dtype=torch.int8)
        scales = torch.zeros(out_dim, groups)
        for g in range(groups):
            start = g * group_size
            end = min(start + group_size, in_dim)
            w = W[:, start:end]
            s = w.abs().max(dim=1).values / qmax
            s = s.clamp(min=1e-8)
            scales[:, g] = s
            W_int[:, start:end] = torch.round(w / s.unsqueeze(1)).clamp(-qmax, qmax).to(torch.int8)
        return W_int, scales


class W8A16Linear(nn.Module):
    """权重 int8 + scale 的推理线性层。"""

    def __init__(self, in_dim: int, out_dim: int, granularity: str = "per_channel",
                 group_size: int = 128):
        super().__init__()
        self.granularity = granularity
        self.group_size = group_size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.register_buffer("W_int", torch.zeros(out_dim, in_dim, dtype=torch.int8))
        if granularity == "per_tensor":
            self.register_buffer("scale", torch.tensor(1.0))
        elif granularity == "per_channel":
            self.register_buffer("scale", torch.ones(out_dim))
        elif granularity == "per_group":
            groups = (in_dim + group_size - 1) // group_size
            self.register_buffer("scale", torch.ones(out_dim, groups))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def from_weight(self, W: torch.Tensor, bias: torch.Tensor = None):
        W_int, scale = quantize_w8a16(W, self.granularity, self.group_size)
        self.W_int = W_int
        self.scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
        if bias is not None:
            self.bias.data = bias
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.granularity == "per_tensor":
            W = self.W_int.float() * self.scale
        elif self.granularity == "per_channel":
            W = self.W_int.float() * self.scale.unsqueeze(1)
        elif self.granularity == "per_group":
            groups = self.scale.shape[1]
            W = torch.zeros_like(self.W_int, dtype=torch.float32)
            for g in range(groups):
                start = g * self.group_size
                end = min(start + self.group_size, self.in_dim)
                W[:, start:end] = self.W_int[:, start:end].float() * self.scale[:, g].unsqueeze(1)
        return F.linear(x, W, self.bias)


def compare_error(W: torch.Tensor, x: torch.Tensor):
    """返回各粒度下与 fp16 的最大误差。"""
    y_ref = x @ W.t()
    results = {}
    for g in ["per_tensor", "per_channel", "per_group"]:
        W_int, scale = quantize_w8a16(W, g)
        if g == "per_tensor":
            W_deq = W_int.float() * scale
        elif g == "per_channel":
            W_deq = W_int.float() * scale.unsqueeze(1)
        else:
            groups = scale.shape[1]
            gs = 128
            W_deq = torch.zeros_like(W_int, dtype=torch.float32)
            for gi in range(groups):
                start = gi * gs
                end = min(start + gs, W.shape[1])
                W_deq[:, start:end] = W_int[:, start:end].float() * scale[:, gi].unsqueeze(1)
        y_q = x @ W_deq.t()
        results[g] = (y_ref - y_q).abs().max().item()
    return results


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    out_dim, in_dim = 64, 128
    W = torch.randn(out_dim, in_dim) * 0.1
    x = torch.randn(4, in_dim)

    for g in ["per_tensor", "per_channel", "per_group"]:
        W_int, scale = quantize_w8a16(W, g)
        assert W_int.dtype == torch.int8
        assert W_int.shape == W.shape
        print(f"✅ {g}: W_int {W_int.shape} + scale {scale.shape if hasattr(scale, 'shape') else 'scalar'}")

    layer = W8A16Linear(in_dim, out_dim, "per_channel").from_weight(W)
    y = layer(x)
    y_ref = x @ W.t()
    err = (y - y_ref).abs().max().item()
    assert err < 0.01, f"per_channel 误差过大: {err}"
    print(f"✅ W8A16Linear forward: 误差 {err:.6f}")

    errs = compare_error(W, x)
    assert errs["per_channel"] <= errs["per_tensor"]
    print(f"✅ 误差对比: {errs}")

    mem_fp = W.numel() * 4
    mem_int8 = W.numel() * 1
    print(f"✅ 显存: fp32 {mem_fp} -> int8 {mem_int8} ({mem_fp/mem_int8:.0f}x 压缩)")
    print("✅ 全部测试通过")
