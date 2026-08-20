"""
【题目】梯度累积（Gradient Accumulation）

【背景】
显存装不下大 batch 时，把一个大 batch 拆成 accumulation_steps 个 micro-batch，
依次前向/反向累积梯度，累积满后再 step，数学上等价于一次大 batch 更新。
关键细节：(1) 每个 micro-batch 的 loss 要除以 accumulation_steps，否则累积出的是"和"而非"均值"；
(2) zero_grad 只在真实 step 后做一次；(3) 与 DDP/AMP 组合时，梯度同步/unscale 只在真实 step 的那个 micro-batch 触发，否则重复通信、重复缩放。

【输入/输出】
- 输入：model, optimizer, dataloader（产出 (x, y) micro-batch）, criterion, accumulation_steps, device
- 输出：训练若干步，每个 accumulation_steps 个 micro-batch 完成一次权重更新

【考察点】
- loss 缩放与等价性、zero_grad 时机
- 与 DDP（何时同步梯度）、AMP（何时 unscale）的边界
- 最后一个不满 accumulation_steps 的 batch 处理
- 提示：需要梯度裁剪时用 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)（在 optimizer.step 前调用）
"""
import copy
import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,            # yields (x: Tensor[B,...], y: Tensor[B,...])
    accum_steps: int,
    criterion: nn.Module = None,  # 损失函数，默认 MSELoss
    device: torch.device = None,
):
    """
    用梯度累积完成一个 epoch。
    要求：
    1. 每个 micro-batch: loss = criterion(model(x), y) / accum_steps; loss.backward()
    2. 每 accum_steps 个 micro-batch 才 optimizer.step() + optimizer.zero_grad()
    3. 末尾不足 accum_steps 时也要 step（兜底）
    """
    if criterion is None:
        criterion = nn.MSELoss()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for micro_step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        raw_loss = criterion(pred, y)
        scaled_loss = raw_loss / accum_steps
        scaled_loss.backward()
        if (micro_step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.step()
    optimizer.zero_grad()


# ===== 等价性自检（可选）=====
def equivalence_check():
    """
    对比：accum_steps=K 的小 batch 累积  vs  一次拼成大 batch
    断言两者单步更新后的权重在 fp 误差内一致。
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    accum_steps = 4
    input_dim = 10
    output_dim = 1

    x_list = [torch.randn(batch_size, input_dim, device=device) for _ in range(accum_steps)]
    y_list = [torch.randn(batch_size, output_dim, device=device) for _ in range(accum_steps)]

    x_big = torch.cat(x_list, dim=0)
    y_big = torch.cat(y_list, dim=0)

    # 注意要用 eval 模式，避免 BatchNorm / Dropout 随机性干扰
    model_a = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    ).to(device).eval()

    model_b = copy.deepcopy(model_a).eval()

    criterion = nn.MSELoss()
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.01)

    optimizer_a.zero_grad()
    pred_big = model_a(x_big)
    loss_big = criterion(pred_big, y_big)
    loss_big.backward()
    optimizer_a.step()

    optimizer_b.zero_grad()
    for x_small, y_small in zip(x_list, y_list):
        pred_small = model_b(x_small)
        loss_small = criterion(pred_small, y_small) / accum_steps
        loss_small.backward()
    optimizer_b.step()

    max_diff = 0.0
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        diff = (p_a - p_b).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"两模型权重的最大绝对误差: {max_diff:.8f}")
    assert max_diff < 1e-5, f"梯度累积不等价！误差 {max_diff} 过大"
    print("✅ 等价性验证通过")


if __name__ == "__main__":
    equivalence_check()
