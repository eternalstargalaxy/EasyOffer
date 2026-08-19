"""
【题目】混合精度训练 AMP + 动态 loss scaling

【背景】
fp16 尾数位少、动态范围窄，小梯度反向时易下溢为 0。做法：前向用 fp16 copy 权重计算，
维护 fp32 master weight；用 scale 放大 loss 再反向（梯度随之放大），反向后检测 inf/nan：
溢出则跳过本步更新并把 scale 衰减；连续若干步未溢出则 scale 增长。更新前对梯度 unscale=grad/scale。
bf16 动态范围与 fp32 同（指数位相同），不需要 loss scaling，但精度更低。

【输入/输出】
- 输入：model(fp32 master), optimizer, dataloader, dtype∈{fp16,bf16}
- 输出：每步前向用 fp16 权重、更新 fp32 master；scale 动态调整

【考察点】
- fp16 vs bf16 差异（为何 bf16 不需 scaling）
- inf/nan 检测与"跳过更新"的正确性
- unscale 时机 vs 梯度裁剪顺序（先 unscale 再 clip 再 step）
"""
import torch
import torch.nn as nn


class AMPScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.backoff_factor = backoff_factor
        self.growth_factor = growth_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0   # 连续未溢出步数

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """返回 loss * scale（用于 backward）"""
        return loss * self.scale

    def unscale_(self, grads):
        """原地 grad /= scale；若发现 inf/nan 标记溢出"""
        for grad in grads:
            grad /= self.scale
        return any((torch.isnan(grad).any() or torch.isinf(grad).any()) for grad in grads)

    def step(self, optimizer, grads):
        """
        检测溢出 -> 跳过更新、scale *= backoff
        否则 unscale -> optimizer.step -> 连续未溢出计数达 growth_interval 则 scale *= growth
        """
        if self.unscale_(grads):
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            optimizer.step()
        
        if self._growth_tracker >= self.growth_interval:
            self.scale *= self.growth_factor
            self._growth_tracker = 0


def train_step(model: nn.Module, optimizer, scaler: AMPScaler,
               x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype):
    """
    1. 用 fp16/bf16 copy master 权重做前向
    2. loss = scaler.scale_loss(criterion(...)); loss.backward()
    3. scaler.step(optimizer, grads)  # 内含 unscale + 更新 + scale 调整
    """
    model.train()
    x = x.to(device)
    y = y.to(device)

    with torch.autocast("cuda", dtype=dtype):
        pred = model(x)
        loss = criterion(pred, y)

    if dtype == torch.bfloat16:
        loss.backward()
        optimizer.step()

    else:
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        scaler.step(optimizer, grads)
        
    optimizer.zero_grad()

