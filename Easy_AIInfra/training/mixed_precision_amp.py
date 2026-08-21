"""
【题目】混合精度训练 AMP + 动态 loss scaling

【背景】
fp16 尾数位少、动态范围窄，小梯度反向时易下溢为 0。做法：前向用 fp16 copy 权重计算，
维护 fp32 master weight；用 scale 放大 loss 再反向（梯度随之放大），反向后检测 inf/nan：
溢出则跳过本步更新并把 scale 衰减；连续若干步未溢出则 scale 增长。更新前对梯度 unscale=grad/scale。
bf16 动态范围与 fp32 同（指数位相同），不需要 loss scaling，但精度更低。

【输入/输出】
- 输入：model(fp32 master), optimizer, scaler, criterion, x, y, dtype∈{fp16,bf16}, device
- 输出：每步前向用 fp16 权重、更新 fp32 master；scale 动态调整

【考察点】
- fp16 vs bf16 差异（为何 bf16 不需 scaling）
- inf/nan 检测与"跳过更新"的正确性
- unscale 时机 vs 梯度裁剪顺序（先 unscale 再 clip 再 step）
- 提示：torch.autocast(device_type="cuda", dtype=...) 用于自动混合精度前向
"""
import torch
import torch.nn as nn


class AMPScaler:
    """动态 loss scaling，模拟 torch.cuda.amp.GradScaler。"""

    def __init__(self, init_scale: float = 2.0 ** 16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.backoff_factor = backoff_factor
        self.growth_factor = growth_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_(self, grads: torch.Tensor):
        """原地 grad /= scale；若发现 inf/nan 返回 True（溢出）。"""
        overflow = False
        for grad in grads:
            if grad is None:
                continue
            grad.div_(self.scale)
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                overflow = True
        return overflow

    def step(self, optimizer, grads: torch.Tensor):
        """检测溢出 -> 跳过更新、scale *= backoff；否则更新 + 可能增长 scale。"""
        if self.unscale_(grads):
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            return False
        else:
            self._growth_tracker += 1
            optimizer.step()
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            return True


def train_step(model: nn.Module, optimizer, scaler: AMPScaler,
               x: torch.Tensor, y: torch.Tensor,
               criterion: nn.Module = None,
               dtype: torch.dtype = torch.float16,
               device: torch.device = None):
    """AMP 训练一步：前向 -> scaled loss -> backward -> unscale -> step。"""
    model.train()
    if criterion is None:
        criterion = nn.MSELoss()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)

    use_autocast = device.type == "cuda"
    if use_autocast:
        with torch.autocast("cuda", dtype=dtype):
            pred = model(x)
            loss = criterion(pred, y)
    else:
        pred = model(x)
        loss = criterion(pred, y)

    if dtype == torch.bfloat16 and use_autocast:
        loss.backward()
        optimizer.step()
    else:
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        scaler.step(optimizer, grads)

    optimizer.zero_grad()
    return loss.item()


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = AMPScaler(init_scale=1024.0)
    scaled = scaler.scale_loss(torch.tensor(2.0))
    assert abs(scaled.item() - 2048.0) < 1e-6
    print("✅ scale_loss 正确")

    scaler2 = AMPScaler(init_scale=1024.0)
    grads_normal = [torch.randn(4, 4) * 10]
    overflow = scaler2.unscale_(grads_normal)
    assert overflow == False
    print("✅ unscale_ 正常梯度无溢出")

    scaler3 = AMPScaler(init_scale=1024.0)
    grads_nan = [torch.randn(4, 4), torch.tensor([float("nan")])]
    overflow3 = scaler3.unscale_(grads_nan)
    assert overflow3 == True
    print("✅ unscale_ 检测 NaN 溢出")

    scaler4 = AMPScaler(init_scale=1024.0, growth_interval=3)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)

    loss = train_step(model, optimizer, scaler4, x, y, dtype=torch.float16, device=device)
    assert scaler4.scale > 0
    print(f"✅ train_step fp16: loss={loss:.4f}, scale={scaler4.scale}")

    scaler5 = AMPScaler(init_scale=1024.0)
    model2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss2 = train_step(model2, optimizer2, scaler5, x, y, dtype=torch.bfloat16, device=device)
    print(f"✅ train_step bf16: loss={loss2:.4f}")

    scaler6 = AMPScaler(init_scale=1024.0, backoff_factor=0.5)
    scale_before = scaler6.scale
    grads_inf = [torch.tensor([[float("inf")]])]
    scaler6.step(torch.optim.SGD([nn.Parameter(torch.randn(1, 1))], lr=0.01), grads_inf)
    assert scaler6.scale == scale_before * 0.5, "溢出后 scale 应衰减"
    print(f"✅ 溢出处理: scale {scale_before} -> {scaler6.scale}")
    print("✅ 全部测试通过")
