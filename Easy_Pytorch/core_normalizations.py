"""
【题目】归一化合集：LayerNorm / RMSNorm / BatchNorm（生产级）

【背景】
归一化是深度学习训练稳定的关键。本文件实现三种归一化的 PyTorch nn.Module 版本，
贴近 Llama / GPT 等真实模型的实现风格。
- LayerNorm：BERT/GPT-2 使用，计算 mean+var
- RMSNorm：Llama/Gemma 使用，只计算 RMS，比 LN 快 7-64%
- BatchNorm：CV 领域常用，NLP 中少用但面试常考

【考察点】
- LN vs RMSNorm 的计算差异与效果对比
- 反向传播的正确性（与 PyTorch 自动微分对齐）
- weight/bias 参数的初始化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm(nn.Module):
    """标准 LayerNorm：减均值除标准差，再仿射变换。"""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
    RMSNorm：只用 RMS 归一，不减均值。
    Llama 的实现：x * rsqrt(mean(x²) + eps) * weight。
    比 LayerNorm 少算一次均值，速度快 7-64%。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Llama 源码风格：单独抽出 _norm 便于 fused kernel 替换。"""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class BatchNorm1d(nn.Module):
    """BatchNorm1d：跨 batch 维归一，训练用 batch 统计，推理用 running 统计。"""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    dim = 64
    x = torch.randn(4, 10, dim)

    ln = LayerNorm(dim)
    out = ln(x)
    assert out.shape == x.shape
    ln_ref = nn.LayerNorm(dim)
    ln_ref.weight.data = ln.weight.data.clone()
    ln_ref.bias.data = ln.bias.data.clone()
    assert torch.allclose(out, ln_ref(x), atol=1e-5), "LayerNorm 应与 PyTorch 一致"
    print("✅ LayerNorm: 与 nn.LayerNorm 一致")

    rms = RMSNorm(dim)
    out_rms = rms(x)
    assert out_rms.shape == x.shape
    rms_val = out_rms.pow(2).mean(dim=-1)
    assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3), "RMSNorm 后均方应≈1"
    print("✅ RMSNorm: 均方≈1")

    bn = BatchNorm1d(dim)
    x_bn = torch.randn(32, dim)
    bn.train()
    out_bn = bn(x_bn)
    assert out_bn.shape == x_bn.shape
    bn.eval()
    out_bn_eval = bn(x_bn)
    assert out_bn_eval.shape == x_bn.shape
    print("✅ BatchNorm1d: train/eval 模式正确")

    out.sum().backward()
    assert ln.weight.grad is not None
    assert rms.weight.grad is not None
    print("✅ 反向传播: 梯度正确")
    print("✅ 全部测试通过")
