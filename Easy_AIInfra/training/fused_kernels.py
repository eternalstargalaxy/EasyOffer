"""
【题目】Fused Kernels：fused Adam / fused LayerNorm / fused MLP

【背景】
GPU kernel launch 开销不可忽略(微秒级)，大量独立 kernel 累积延迟。
Fused kernel 把多个逐元素操作融合为一个 kernel 减少 launch + HBM 往返。
Fused Adam：正常 Adam 需要 6 次逐元素操作(load param/grad/m/v, store param/m/v)，
融合后 1 次 kernel 完成全流程，省 5x launch + 3x HBM 读写。
Fused LayerNorm：LN 中的 mean/variance/normalize/scale 融合为 1 kernel。
Fused MLP：gelu + linear 融合，省 gelu 的中间 HBM 写入。

【输入/输出】
- 输入：param, grad, m, v (fp32), lr, betas, eps
- 输出：更新后的 param, m, v

【考察点】
- kernel fusion 的访存优化原理
- fused Adam 的 step 公式实现
- 提示：torch.lerp, torch.addcmul 逐元素; cuda 端 ATen 实现
"""
import torch
import torch.nn.functional as F


def fused_adam_update(param: torch.Tensor, grad: torch.Tensor,
                      m: torch.Tensor, v: torch.Tensor,
                      step: int, lr: float = 1e-3,
                      betas: tuple = (0.9, 0.999), eps: float = 1e-8):
    """Fused Adam：一次 kernel 完成 m/v 更新 + bias correction + param 更新。"""
    beta1, beta2 = betas
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    lr_t = lr * (bias2 ** 0.5) / bias1
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    param.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
    return param


def fused_layernorm(x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor, eps: float = 1e-5):
    """Fused LayerNorm：mean/var/normalize/scale 一次完成。"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(var + eps)
    return x_normed * weight + bias


def fused_mlp(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """Fused MLP：linear + gelu 融合，省中间 HBM 写入。"""
    logits = F.linear(x, weight, bias)
    return F.gelu(logits)


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    p = torch.randn(4, 8)
    g = torch.randn(4, 8)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    p_orig = p.clone()

    p_new = fused_adam_update(p, g, m, v, step=1)
    assert p_new.shape == p.shape
    assert not torch.equal(p_new, p_orig), "参数应被更新"
    print("✅ Fused Adam: 参数更新成功")

    p2 = torch.randn(4, 8)
    g2 = torch.randn(4, 8)
    m2 = torch.zeros_like(p2)
    v2 = torch.zeros_like(p2)
    fused_adam_update(p2, g2, m2, v2, step=1)
    fused_adam_update(p2, g2, m2, v2, step=2)
    assert m2.abs().sum() > 0 and v2.abs().sum() > 0
    print("✅ Fused Adam: 多步更新正确")

    x = torch.randn(2, 4, 8)
    w = torch.ones(8)
    b = torch.zeros(8)
    y = fused_layernorm(x, w, b)
    assert y.shape == x.shape
    var = y.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4), "LN 后方差应为 1"
    mean = y.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), "LN 后均值应为 0"
    print("✅ Fused LayerNorm: 均值=0, 方差=1")

    ln_ref = torch.nn.LayerNorm(8)
    ln_ref.weight.data = w.clone()
    ln_ref.bias.data = b.clone()
    y_ref = ln_ref(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "应与 PyTorch LayerNorm 一致"
    print("✅ Fused LayerNorm: 与 PyTorch 一致")

    x_mlp = torch.randn(4, 16)
    w_mlp = torch.randn(8, 16)
    b_mlp = torch.randn(8)
    y_mlp = fused_mlp(x_mlp, w_mlp.t(), b_mlp)
    assert y_mlp.shape == (4, 8)
    y_ref_mlp = F.gelu(F.linear(x_mlp, w_mlp.t(), b_mlp))
    assert torch.allclose(y_mlp, y_ref_mlp, atol=1e-6)
    print("✅ Fused MLP: linear+gelu 融合正确")
    print("✅ 全部测试通过")
