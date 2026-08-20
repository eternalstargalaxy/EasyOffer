"""
【题目】Fused Kernels：fused Adam / fused LayerNorm / fused MLP

【背景】
GPU kernel launch 开销不可忽略(微秒级)，大量独立 kernel 累积延迟。
Fused kernel 把多个逐元素操作融合为一个 kernel 减少 launch + HBM 往返。
Fused Adam：正常 Adam 需要 6 次逐元素操作(load param/grad/m/v, store param/m/v)，
融合后 1 次 kernel 完成全流程，省 5x launch + 3x HBM 读写。
Fused LayerNorm：LN 中的 mean/variance/normalize/scale 融合为 1 kernel。
Fused MLP：gelu + linear 融合，省 gelu 的中间 HBM 写入。
Megatron-LM / flash-attn 都内置 fused LN, PyTorch 2.0 内置 fused Adam(torch.optim.AdamW fused=True)。

【输入/输出】
- 输入：param, grad, m, v (fp32), lr, betas, eps
- 输出：更新后的 param, m, v

【考察点】
- kernel fusion 的访存优化原理
- fused Adam 的 step 公式实现
- 提示：torch.lerp, torch.addcmul 逐元素; cuda 端 ATen 实现
"""
import torch


def fused_adam_update(param: torch.Tensor, grad: torch.Tensor,
                       m: torch.Tensor, v: torch.Tensor,
                       step: int, lr: float = 1e-3,
                       betas=(0.9, 0.999), eps: float = 1e-8):
    raise NotImplementedError


def fused_layernorm(x: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor, eps: float = 1e-5):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    p = torch.randn(4, 8)
    g = torch.randn(4, 8)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    try:
        p2 = fused_adam_update(p.clone(), g, m.clone(), v.clone(), 1)
        assert p2.shape == p.shape
        assert not torch.equal(p2, p)
        print('✅' " Fused Adam 测试通过")
    except NotImplementedError:
        print('ℹ' " 待实现")

    x = torch.randn(2, 4, 8)
    w = torch.ones(8)
    b = torch.zeros(8)
    try:
        y = fused_layernorm(x, w, b)
        assert y.shape == x.shape
        var = y.pow(2).mean(dim=-1)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
        print('✅' " Fused LN 测试通过")
    except NotImplementedError:
        print('ℹ' " 待实现")
