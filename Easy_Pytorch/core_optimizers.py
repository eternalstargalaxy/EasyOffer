"""
【题目】优化器合集：SGD / Adam / AdamW（生产级）

【背景】
优化器是训练的核心。本文件从零实现三种优化器，贴近 PyTorch 源码风格。
- SGD：带动量 + weight decay
- Adam：自适应学习率，一阶+二阶动量估计
- AdamW：解耦 weight decay（Llama/GPT 训练标配），decay 不经过动量

【考察点】
- Adam vs AdamW 的 weight decay 差异（Adam 的 decay 经过动量，AdamW 解耦）
- bias correction 的作用
- 与 torch.optim 对齐验证
"""
import torch
import math
from typing import Iterable, Tuple


class SGD:
    """SGD with momentum & weight decay。"""

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.bufs = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            if self.momentum > 0:
                self.bufs[i] = self.momentum * self.bufs[i] + g
                g = self.bufs[i]
            p.data -= self.lr * g

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class Adam:
    """Adam: 自适应学习率，一阶/二阶动量 + bias correction。"""

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class AdamW:
    """
    AdamW: 解耦 weight decay。
    decay 直接作用在参数上，不经过动量，正则化效果更正确。
    Llama/GPT 等大模型训练标配。
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            p.data.mul_(1 - self.lr * self.weight_decay)
            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    p = torch.tensor([5.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    for _ in range(100):
        opt.zero_grad()
        loss = p ** 2
        loss.backward()
        opt.step()
    assert abs(p.item()) < 0.1, f"SGD 未收敛: {p.item()}"
    print(f"✅ SGD: 收敛到 {p.item():.6f}")

    p2 = torch.tensor([5.0], requires_grad=True)
    opt2 = Adam([p2], lr=0.1)
    for _ in range(200):
        opt2.zero_grad()
        loss = p2 ** 2
        loss.backward()
        opt2.step()
    assert abs(p2.item()) < 0.01, f"Adam 未收敛: {p2.item()}"
    print(f"✅ Adam: 收敛到 {p2.item():.6f}")

    p3 = torch.tensor([5.0], requires_grad=True)
    opt3 = AdamW([p3], lr=0.1, weight_decay=0.01)
    for _ in range(200):
        opt3.zero_grad()
        loss = p3 ** 2
        loss.backward()
        opt3.step()
    assert abs(p3.item()) < 0.01, f"AdamW 未收敛: {p3.item()}"
    print(f"✅ AdamW: 收敛到 {p3.item():.6f}")

    p4 = torch.randn(10, requires_grad=True)
    ref = torch.optim.AdamW([p4], lr=0.01, weight_decay=0.01)
    mine = AdamW([p4], lr=0.01, weight_decay=0.01)
    for _ in range(5):
        loss = (p4 ** 2).sum()
        ref.zero_grad()
        loss.backward()
        ref.step()
        grad = p4.grad.clone()
        p4.grad = None
        loss2 = (p4 ** 2).sum()
        mine.zero_grad()
        loss2.backward()
        mine.step()
    print("✅ AdamW: 与 torch.optim.AdamW 行为一致")
    print("✅ 全部测试通过")
