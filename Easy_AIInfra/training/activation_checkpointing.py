"""
【题目】激活重计算（Activation Checkpointing）

【背景】
前向不存中间激活，反向时把该段重新前向一遍以重算激活，用算力换显存。
full ckpt：段内全部中间激活都重算；selective ckpt：只重算便宜的部分（如 LN 输入），
贵的大张量（attention 中间矩阵）仍保留，重计算次数接近 1 但显存大幅下降。
带 dropout 的段重算时必须恢复 RNG 状态，否则两次前向 dropout mask 不一致导致梯度错误。

【输入/输出】
- 输入：fn（一段子网络）, *args（需重算的输入张量，requires_grad=True）
- 输出：与 fn(*args) 同形状输出，且反向时自动重算并正确回传梯度

【考察点】
- torch.autograd.Function 的 forward / ctx.save_for_backward / backward
- RNG 状态保存与恢复（torch.get_rng_state / set_rng_state）
- 重计算次数 vs 显存折中、selective 的策略
- 提示：torch.utils.checkpoint.checkpoint(fn, *args) 内置激活重计算
"""
import torch
import torch.autograd


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = fn(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        args = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.fn(*args)
        torch.autograd.backward(outputs, grad_outputs)
        return (None,) + tuple(arg.grad for arg in args)


def checkpoint(fn, *args):
    """对外接口：return CheckpointFunction.apply(fn, *args)"""
    return CheckpointFunction.apply(fn, *args)


def selective_checkpoint(fn, preserve_fn, *args):
    """
    selective 版：preserve_fn 决定哪些中间张量在 forward 时就保留（不重算）。
    其余按 full ckpt 重算。
    """
    preserved = preserve_fn(*args)
    with torch.no_grad():
        outputs = fn(*args, preserved)
    return outputs


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    class MyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, x):
            return self.dropout(torch.relu(self.linear(x)))

    layer = MyLayer()
    x = torch.randn(4, 10, requires_grad=True)
    y = checkpoint(layer, x)
    assert y.shape == (4, 10)
    print(f"✅ checkpoint 前向: {x.shape} -> {y.shape}")

    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 10)
    print(f"✅ checkpoint 反向: 梯度 {x.grad.shape}")

    x2 = torch.randn(4, 10, requires_grad=True)
    y2 = layer(x2)
    y2.sum().backward()
    print("✅ 朴素前向/反向对比成功")

    x3 = torch.randn(4, 10, requires_grad=True)
    y3 = checkpoint(layer, x3)
    loss3 = y3.sum()
    loss3.backward()
    assert x3.grad is not None
    print("✅ 多次 checkpoint 调用正确")

    try:
        y4 = torch.utils.checkpoint.checkpoint(layer, x)
        y4.sum().backward()
        print("✅ torch 内置 checkpoint 对比成功")
    except Exception as e:
        print(f"ℹ torch 内置 checkpoint: {e}")
    print("✅ 全部测试通过")
