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
"""
import torch
import torch.autograd


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        """
        1. 保存 fn 与输入（save_for_backward 只存输入，不存中间激活）
        2. with torch.no_grad(): outputs = fn(*args)
        3. 保存 forward 时的 RNG 状态到 ctx
        4. return outputs.detach()（需保留 requires_grad 信息以便 backward）
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        1. 恢复 forward 的 RNG 状态
        2. 重新前向 fn(*saved_args)（这次带 grad）得到中间激活
        3. torch.autograd.backward(outputs, grad_outputs) 回传到 saved_args
        4. 返回 (None, *input_grads)
        """
        raise NotImplementedError


def checkpoint(fn, *args):
    """对外接口：return CheckpointFunction.apply(fn, *args)"""
    raise NotImplementedError


def selective_checkpoint(fn, preserve_fn, *args):
    """
    selective 版：preserve_fn 决定哪些中间张量在 forward 时就保留（不重算）。
    其余按 full ckpt 重算。
    """
    raise NotImplementedError
